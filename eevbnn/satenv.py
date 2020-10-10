"""env for manipulating SAT formulas"""

from .utils import setup_pyx_import, iter_prod_range
from abc import ABCMeta, abstractmethod
import numpy as np
import torch

import enum
from contextlib import contextmanager

with setup_pyx_import():
    from .satenv_tensorimpl import TensorMathImpl

class SolveResult(enum.Enum):
    SAT = 'SAT'
    UNSAT = 'UNSAT'
    TLE = 'TLE'
    """timeout (Time Limit Execeeded)"""

    def __str__(self):
        return self.value

class BoolRef(metaclass=ABCMeta):
    """reference to a bool value"""

    @abstractmethod
    def __neg__(self):
        """logical negation"""

    @abstractmethod
    def __and__(self, rhs):
        """logical and"""

    @abstractmethod
    def __or__(self, rhs):
        """logical or"""


class SatEnv(metaclass=ABCMeta):
    _varcnt = 0
    __new_var_preset = None
    __allow_create_new_var = True

    tensor_arith_impl = TensorMathImpl
    """set to None to use naive python impl (for checking correctness)"""

    @abstractmethod
    def add(self, v: BoolRef):
        """add a new constraint"""

    @abstractmethod
    def add_reduce_or(self, props):
        """add a constraint that requires OR of ``props`` to be true"""

    @abstractmethod
    def solve(self) -> SolveResult:
        """
        try to solve the SAT and setup model if it is satisfiable
        :return: whether the SAT is satisfiable
        """

    def get_prev_solve_time(self):
        """get true solve time for previous solve() call; return None to use the
        timing by the caller"""

    @abstractmethod
    def get(self, v) -> int:
        """get the value of a var after :meth:`solve` returns
        :attr:`~.SolveResult.SAT`"""

    @abstractmethod
    def _do_new_var(self, name, varid):
        """implement new_var; ``varid`` starts from ``1``"""

    def new_var(self, name: str) -> BoolRef:
        """create a new bool variable; note that its behavior can be controlled
        by """
        if self.__new_var_preset:
            return self.__new_var_preset.pop()
        assert self.__allow_create_new_var
        self._varcnt += 1
        return self._do_new_var(name, self._varcnt)

    @abstractmethod
    def make_at_least(self, props, k: int, name: str) -> BoolRef:
        """:param props: list of ``(var, weight)`` pairs"""

    @abstractmethod
    def make_at_most(self, props, k: int, name: str) -> BoolRef:
        """:param props: list of ``(var, weight)`` pairs"""

    @abstractmethod
    def make_empty_expr(self) -> BoolRef:
        """create an empty expression (const true)"""

    @abstractmethod
    def write_formula(self, fout):
        """write the formula to a text file"""

    @abstractmethod
    def set_timeout(self, timeout: float):
        """set solver timeout in seconds"""

    def set_verbosity(self, level):
        """this method can be overwritten to support verbosity setting"""

    def set_var_preference(self, var, pref):
        """set the preference of a var; lower value means higher branching
        priority

        :param var: the literal to represent a var (can be in negation form)
        """

    def set_var_name(self, var, name):
        """set the name of a var, for debug purposes

        :param var: the literal to represent a var (can be in negation form)
        """

    @contextmanager
    def set_future_new_vars(self, new_vars):
        """set a list of vars to be used as future new vars; the method should
        be used as a context manager
        """
        if not isinstance(new_vars, list):
            new_vars = list(new_vars)
        self.__new_var_preset = new_vars[::-1]
        self.__allow_create_new_var = False
        yield self
        assert len(self.__new_var_preset) == 0, (
            f'{len(self.__new_var_preset)} vars unsed in future new vars'
        )
        del self.__new_var_preset
        del self.__allow_create_new_var

    def make_clause_recorder(self):
        """make a recorder object if it is supported; return None if
        recording is unsupported"""


class ClauseRecorder(metaclass=ABCMeta):
    @abstractmethod
    def scoped_attach_to(self, env: SatEnv):
        """attach this recorder to a sat env; should be implemented as a
        contextmanager"""

    @abstractmethod
    def replay(self, env):
        """add the recorded clauses to a new env"""


def gen_le(env: SatEnv, v0: np.ndarray, v1: np.ndarray):
    """generate an expression for ``v0 <= v1`` under the big-endian bit
    interpretation, where ``v0`` are :class:`BoolRef` objects and ``v1`` is a
    constant"""
    assert v0.shape == v1.shape and v0.ndim == 1
    assert all(isinstance(i, BoolRef) for i in v0)
    assert all(i in [0, 1] for i in v1)
    size = v0.size
    def gen(start):
        this_1 = v0[start]
        this_0 = -this_1
        if start == size - 1:
            if v1[start]:
                return env.make_empty_expr()
            return this_0
        if v1[start]:
            return this_0 | (this_1 & gen(start + 1))
        return this_0 & gen(start + 1)
    return gen(0)


class TensorWrapper:
    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def size(self, dim):
        return self.value.shape[dim]


class BoolTensor(TensorWrapper):
    """a tensor containing :class:`BoolRef` objects"""
    __slots__ = ['env', 'value']

    def __init__(self, env: SatEnv, value: np.ndarray):
        self.env = env
        self.value = value

    @classmethod
    def _canonize_weights(cls, weight, bias, sign):
        w = weight.astype(np.int32)
        assert np.all(w == weight)
        if bias is None or sign is None:
            assert bias is None and sign is None
            b = s = None
        else:
            b = list(map(int, bias))
            s = list(map(int, sign))
            assert np.all(b == bias)
            assert np.all(s == sign) and set(s).issubset({-1, 1})
        return w, b, s

    def conv2d(self, name: str,
               weight: np.ndarray, bias: np.ndarray,
               sign: np.ndarray, stride, padding):
        """run a convolution with this tensor as input, as described in
        :class:`.BinConv2dEval`"""
        impl = self.env.tensor_arith_impl
        if impl is not None:
            return impl.conv2d(
                self, name, weight, bias, sign, stride, padding)

        # below is a python reference impl
        assert (self.ndim == 4 and weight.ndim == 4
                and self.shape[1] == weight.shape[1]), (
                    f'conv shape mismatch: {self.shape}, {weight.shape}')
        N, IC, IH, IW = self.shape
        OC, _, KH, KW = weight.shape
        OH = (IH + padding[0] * 2 - KH) // stride[0] + 1
        OW = (IW + padding[1] * 2 - KW) // stride[1] + 1
        assert bias.shape == sign.shape == (OC, ), f'{bias.shape} {sign.shape}'
        result = np.empty((N, OC, OH, OW), dtype=BoolRef)
        weight, bias, sign = self._canonize_weights(weight, bias, sign)
        for n, oc, oh, ow in iter_prod_range(N, OC, OH, OW):
            acc = []
            for ic, kh, kw in iter_prod_range(IC, KH, KW):
                ih = oh * stride[0] + kh - padding[0]
                iw = ow * stride[1] + kw - padding[1]
                if ih >= 0 and ih < IH and iw >= 0 and iw < IW:
                    w = int(weight[oc, ic, kh, kw])
                    if w:
                        acc.append((self.value[n, ic, ih, iw], w))
            result[n, oc, oh, ow] = self._dot_thresh(
                f'{name}_{n}_{oc}_{oh}_{ow}',
                acc, bias[oc], sign[oc])

        return self.make_replaced_value(result)

    def matmul(self, name: str,
               weight: np.ndarray, bias: np.ndarray, sign: np.ndarray):
        impl = self.env.tensor_arith_impl
        if impl is not None:
            return impl.matmul(self, name, weight, bias, sign)

        # below is a python reference impl

        assert (self.ndim == 2 and weight.ndim == 2 and
                self.shape[1] == weight.shape[0]), (
                    f'matmul shape mismatch: {self.shape} {weight.shape}')
        M, K = self.shape
        _, N = weight.shape
        if bias is not None:
            assert bias.shape == sign.shape == (N, ), (
                f'{bias.shape} {sign.shape}')
        result = np.empty((M, N), dtype=BoolRef)
        weight, bias, sign = self._canonize_weights(weight, bias, sign)
        for i, j in iter_prod_range(M, N):
            acc = [(self.value[i, k], int(weight[k, j]))
                   for k in range(K) if weight[k, j]]
            result[i, j] = self._dot_thresh(
                f'{name}_{i}_{j}', acc, bias[j], sign[j])
        return self.make_replaced_value(result)

    def _dot_thresh(self, name: str, acc, bias: int, sign: int):
        """Compute dot product of items in acc and threshold with zero:
        ``sign * (sum(i[0]*i[1] for i in acc) + bias) >= 0``
        """
        assert isinstance(bias, int) and sign in [-1, 1], (
            f'invalid {bias=} {sign=}')
        if sign > 0:
            return self.env.make_at_least(acc, -bias, name)
        else:
            return self.env.make_at_most(acc, -bias, name)

    def make_replaced_value(self, value: np.ndarray):
        """make a new :class:`BoolTensor` with replaced value"""
        return type(self)(self.env, value)

    def view(self, *shape):
        return self.make_replaced_value(self.value.reshape(shape))

    def as_npy(self, dtype=np.int32) -> np.ndarray:
        """convert this tensor to an numpy array with numerical values. The
        model must be available in the solver.
        """
        def on(x):
            return self.env.get(x)
        ret = np.frompyfunc(on, 1, 1)(self.value)
        return ret.astype(dtype)

    def add_constraint_range(self, lower: np.ndarray, upper: np.ndarray):
        """add constraints for this array to be in the range ``[lower, upper]``

        An array of shape ``(x_0, ..., x_n, y)`` is interpreted as a bit
        representation of an integer array ``(x_0, ..., x_n)`` with ``y`` bits
        in big-endian order

        :return: self
        """
        assert self.shape == lower.shape == upper.shape
        nr_bits = self.shape[-1]
        for v0, vl, vh in zip(self.value.reshape(-1, nr_bits),
                              lower.reshape(-1, nr_bits),
                              upper.reshape(-1, nr_bits)):
            self.env.add(gen_le(self.env, -v0, 1 - vl))
            self.env.add(gen_le(self.env, v0, vh))

        return self

    def add_constraint_range_4d_img_eps(
            self, value: np.ndarray, eps: int, *, truncate_bits: int = 0):
        """
        Constrain the value to be within range ``[value - eps, value + eps]``
        intersected with ``[0, 255]``. ``self`` is interpreted as a bit tensor
        as described in :meth:`add_constraint_range`. The tensor must be in
        ``(N, C, H, W)`` shape.

        :param value: center value with ``int32`` dtype

        :return: self
        """
        assert value.ndim == 4 and value.dtype == np.int32, (
            f'bad value: shape={value.shape} dtype={value.dtype}')
        assert type(eps) is int
        n, c, h, w = value.shape
        bits = 8 - truncate_bits
        assert self.shape == (n, c * bits, h, w), (
            f'shape mismatch: {self.shape} {(n, c, h, w, bits)}')

        def shuffle_bits_to_last_dim(v, b):
            return np.transpose(v.reshape(n, c, b, h, w), (0, 1, 3, 4, 2))

        def make_bits_imm(delta):
            v = value + delta
            if truncate_bits:
                v += 1 << (truncate_bits - 1)
            v = np.clip(v, 0, 255, out=v).astype(np.uint8)
            vbits = shuffle_bits_to_last_dim(np.unpackbits(v, axis=1), 8)
            return vbits[:, :, :, :, :bits]

        vlow = make_bits_imm(-eps)
        vhigh = make_bits_imm(eps)
        x = self.make_replaced_value(shuffle_bits_to_last_dim(self.value, bits))
        x.add_constraint_range(vlow, vhigh)
        return self

    def __getitem__(self, idx):
        return self.make_replaced_value(self.value.__getitem__(idx))

    @classmethod
    def from_empty(cls, env: SatEnv, name: str, shape):
        """make a new tensor with new bool vars"""
        val = np.empty(shape, dtype=BoolRef)
        if val.ndim == 4:
            for n, c, h, w in iter_prod_range(*val.shape):
                val[n, c, h, w] = env.new_var(f'{name}_{n}_{c}_{h}_{w}')
        else:
            for i in range(val.size):
                val.flat[i] = env.new_var(name)
        return cls(env, val)


class QuantizedInputRangeTensor(TensorWrapper):
    """a tensor representing quantized input values within given range

    :type value: :class:`np.ndarray` containing ``List[BoolRef]``
    :var value: allowed offset at each location represented by sum of bools
    :type offset: :class:`np.ndarray`  of integer values
    :var offset: offsets at each location
    :type quant_step: :class:`float`
    :var quant_step: quantization step size (see also :class:`.InputQuantizer`)
    """
    __slots__ = ['env', 'value', 'offset', 'quant_step']

    def __init__(self, env: SatEnv, value: np.ndarray, offset: np.ndarray,
                 quant_step: float):
        assert value.shape == offset.shape and offset.dtype == np.int32
        self.env = env
        self.value = value
        self.offset = offset
        self.quant_step = float(quant_step)

    @classmethod
    def from_value_range(cls, env: SatEnv,
                         low: np.ndarray, high: np.ndarray, quant_step: float,
                         name: str):
        assert low.shape == high.shape and np.all(low <= high)
        quant_step = float(quant_step)
        imin = np.round(low / quant_step).astype(np.int32)
        imax = np.round(high / quant_step).astype(np.int32)

        offset = imin
        def make_val(nr):
            assert nr >= 0
            ret = [env.new_var(name) for _ in range(nr)]
            # we only care about number of true variables; add additional
            # constraint to limit the search space
            for i in range(nr):
                # if i is set to 0, then all following vars should be 0
                for j in range(1, nr):
                    env.add(ret[i] | -ret[j])
            return ret
        value = np.frompyfunc(make_val, 1, 1)(imax - imin)
        if value.ndim == 4:
            for n, c, h, w in iter_prod_range(*value.shape):
                for i, var in enumerate(value[n, c, h, w]):
                    env.set_var_name(var, f'{name}_{n}_{c}_{h}_{w}:{i}')
        return cls(env, value, offset, quant_step)

    def conv2d(self, name: str,
               weight: np.ndarray, bias: np.ndarray,
               sign: np.ndarray, stride, padding) -> BoolTensor:
        """see :meth:`BoolTensor.conv2d`"""
        impl = self.env.tensor_arith_impl
        if impl is not None:
            return impl.conv2d_quant(
                self, name, weight, bias, sign, stride, padding)

        assert (self.ndim == 4 and weight.ndim == 4
                and self.shape[1] == weight.shape[1]), (
                    f'conv shape mismatch: {self.shape}, {weight.shape}')
        N, IC, IH, IW = self.shape
        OC, _, KH, KW = weight.shape
        OH = (IH + padding[0] * 2 - KH) // stride[0] + 1
        OW = (IW + padding[1] * 2 - KW) // stride[1] + 1
        assert bias.shape == sign.shape == (OC, ), f'{bias.shape} {sign.shape}'
        result = np.empty((N, OC, OH, OW), dtype=BoolRef)
        result_bt = BoolTensor(self.env, result)
        weight, bias, sign = result_bt._canonize_weights(weight, bias, sign)
        for n, oc, oh, ow in iter_prod_range(N, OC, OH, OW):
            acc = []
            acc_offset = 0
            for ic, kh, kw in iter_prod_range(IC, KH, KW):
                ih = oh * stride[0] + kh - padding[0]
                iw = ow * stride[1] + kw - padding[1]
                if ih >= 0 and ih < IH and iw >= 0 and iw < IW:
                    w = int(weight[oc, ic, kh, kw])
                    if w:
                        acc.extend((j, w) for j in self.value[n, ic, ih, iw])
                        acc_offset += int(self.offset[n, ic, ih, iw]) * w
            result[n, oc, oh, ow] = result_bt._dot_thresh(
                f'{name}_{n}_{oc}_{oh}_{ow}',
                acc, bias[oc] + acc_offset, sign[oc])

        return result_bt

    def matmul(self, name: str,
               weight: np.ndarray, bias: np.ndarray, sign: np.ndarray):
        assert (self.ndim == 2 and weight.ndim == 2 and
                self.shape[1] == weight.shape[0]), (
                    f'matmul shape mismatch: {self.shape} {weight.shape}')
        M, K = self.shape
        _, N = weight.shape
        assert bias.shape == sign.shape == (N, ), (
            f'{bias.shape} {sign.shape}')
        result = np.empty((M, N), dtype=BoolRef)
        result_bt = BoolTensor(self.env, result)
        weight, bias, sign = result_bt._canonize_weights(weight, bias, sign)
        for i, j in iter_prod_range(M, N):
            acc = []
            acc_offset = 0
            for k in range(K):
                if weight[k, j]:
                    w = int(weight[k, j])
                    acc.extend((v, w) for v in self.value[i, k])
                    acc_offset += int(self.offset[i, k]) * w
            result[i, j] = result_bt._dot_thresh(
                f'{name}_{i}_{j}', acc, bias[j] + acc_offset, sign[j])
        return result_bt

    def as_npy(self, dtype=np.float32) -> np.ndarray:
        """see :meth:`BoolTensor.as_npy`

        note that the result would be multiplied by step size, so it is on the
        original value scale given to :meth:`from_value_range`
        """
        def on(x):
            return sum(int(self.env.get(i)) for i in x)
        ret = np.frompyfunc(on, 1, 1)(self.value) + self.offset
        ret = ret.astype(dtype)
        ret *= self.quant_step
        return ret

    def view(self, *shape):
        return QuantizedInputRangeTensor(
            self.env, self.value.reshape(shape), self.offset.reshape(shape),
            self.quant_step)


class SatChecker:
    """a tool for checking if a solution satisfies all constraints"""

    class ClauseSum:
        _props = None

        def __init__(self, props):
            self._props = []
            for lit, w in props:
                assert type(lit) is int and type(w) is int
                self._props.append((lit, w))

        def eval(self, get_var_fn):
            return sum(w * get_var_fn(lit) for lit, w in self._props)

        def __repr__(self):
            r = []
            for lit, w in self._props:
                if w != 1:
                    r.append(f'{lit}({w})')
                else:
                    r.append(str(lit))
            return ' '.join(r)

    _clauses = None

    def __init__(self):
        self._clauses = []

    def add_leq(self, props, bound, dst):
        self._clauses.append((self.ClauseSum(props), f'<= {bound}', dst))

    def add_geq(self, props, bound, dst):
        self._clauses.append((self.ClauseSum(props), f'>= {bound}', dst))

    def add_or(self, lits):
        self.add_geq([(i, 1) for i in lits], 1, None)

    def check(self, get_var_fn):
        for idx, (csum, cond, dst) in enumerate(self._clauses):
            s = csum.eval(get_var_fn)
            sat = eval(f'{s} {cond}')
            if dst is None:
                dst_v = None
            else:
                dst_v = get_var_fn(dst)
                sat = (int(sat) == dst_v)
            assert sat, (
                f'clause #{idx} violated: {dst} = ({csum} {cond}): {s=} '
                f'val({dst})={dst_v}'
            )
