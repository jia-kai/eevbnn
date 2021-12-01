"""evaluation for a 0/1 binarized network"""

from .utils import (torch_as_npy, ModelHelper, default_dataset_root, ensure_dir,
                    get_nr_correct, get_rng_if_none, npy_chw_f32_to_cv2_uint8,
                    iter_prod_range)
from .satenv import BoolTensor, QuantizedInputRangeTensor, SolveResult

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import time
import functools
import sys
import gc
import itertools
import json
import pickle
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

VAR_PREFERENCE_FIRST_COMPATIBLE = ['none', 'first', 'first-sp']

def setup_weight_bias(layer, weight, sign_bias):
    weight = weight.detach()
    assert (weight.min() >= -1 and weight.max() <= 1 and
            torch.all(weight == torch.floor(weight)))
    layer.register_buffer('weight', weight)
    if sign_bias is None:
        layer.bias = None
        layer.sign = 1
    else:
        sign, bias = sign_bias
        if weight.ndim == 4:
            sign = sign.view(1, sign.size(0), 1, 1)
        else:
            assert weight.ndim == 2
            sign = sign.view(1, sign.size(0))
        assert (sign.min() >= -1 and sign.max() <= 1 and
                torch.all(sign.abs() == 1))
        layer.register_buffer('bias', bias.detach())
        layer.register_buffer('sign', sign.detach())

def get_sign_bias_from_bn(prev: nn.Module, bn: nn.BatchNorm2d, scale=None):
    """get equivalent ``(s, b)`` for this bn such that
    ``bn(prev(x * scale)) >= 0`` iff ``s * (x + b) >= 0``, where
    ``s in [-1, 1]`` and ``b`` is an integer"""
    from .net_bin import PositiveInputCombination
    mean = bn.running_mean

    if isinstance(prev, PositiveInputCombination):
        mean = mean - prev.get_bias()

    k, b = bn._get_scale_bias(bn.running_var, mean)
    # cond: k * x + b > 0
    if scale is not None:
        k *= scale

    sign = k.sign()
    assert torch.all(sign.abs() > 1e-4)
    bias = b / k
    bias = torch.floor_(bias) + (sign < 0).to(bias.dtype)
    return sign, bias

def run_fwd(x, layer, fn):
    with torch.no_grad():
        y = fn(x, layer.weight, layer.bias)
        if layer.rounding or layer.quant_step:
            # also round input layer (with only quant_step) because quant_step
            # has been absorbed into BN scale and the conv should be purely
            # integral
            y = torch.round_(y)
        if layer.bias is not None:
            return (y.mul_(layer.sign) >= 0).to(dtype=x.dtype)
        return y

class BinConv2dEval(nn.Module):
    """input in {0, 1}, weight in {-1, 0, 1}, and sign in {-1, 1}. Output is 1
    iff sign * (conv result + bias) >= 0 """

    layer_name = ''

    def __init__(self, weight: torch.Tensor, sign_bias, rounding,
                 stride, padding, quant_step):
        """:param sign_bias: a tuple ``(sign, bias)`` to binarize the
            output
        :param quant_step: if given, a :class:`QuantizedInputRangeTensor` with
            identical quantization step could be received as the input
        """
        super().__init__()
        setup_weight_bias(self, weight, sign_bias)
        assert isinstance(rounding, bool)
        assert quant_step is None or not rounding
        self.rounding = rounding
        self.stride = stride
        self.padding = padding
        self.quant_step = quant_step

    @property
    def full_layer_name(self):
        return f'conv{self.layer_name}'

    def forward(self, x):
        if isinstance(x, (BoolTensor, QuantizedInputRangeTensor)):
            if isinstance(x, QuantizedInputRangeTensor):
                assert x.quant_step == self.quant_step, (
                    f'quant step mismatch: {x.quant_step} vs {self.quant_step}')
            return x.conv2d(
                self.full_layer_name,
                torch_as_npy(self.weight), torch_as_npy(self.bias),
                torch_as_npy(self.sign).flatten(),
                self.stride, self.padding
            )
        if self.quant_step is not None:
            x = torch.round_(x / self.quant_step)
        return run_fwd(x, self, functools.partial(F.conv2d, stride=self.stride,
                                                  padding=self.padding))

    def infer_out_shape(self, inp_shape):
        assert len(inp_shape) == 4
        n, c, ih, iw = inp_shape
        oc, ic, kh, kw = self.weight.shape
        assert c == ic
        oh = (ih + self.padding[0] * 2 - kh) // self.stride[0] + 1
        ow = (iw + self.padding[1] * 2 - kw) // self.stride[1] + 1
        return n, oc, oh, ow

    @classmethod
    def from_conv_bn(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, quant_step=None):
        """make a conv layer from conv+bn+Binarize01"""
        assert conv.bias is None
        return cls(conv.weight_bin,
                   get_sign_bias_from_bn(conv, bn, scale=quant_step),
                   conv.rounding, conv.stride, conv.padding, quant_step)


class BinLinearEval(nn.Module):
    """see :class:`BinConv2dEval`; note that if ``sign_bias`` is None, output
    would not be thresholded"""

    output_numeric_scale = 1
    layer_name = ''

    def __init__(self, weight: torch.Tensor, sign_bias, scale_bias, rounding, *,
                 quant_step=None):
        """
        :param scale_bias: a tuple of ``(scale, bias)`` for the final
            classification score layer. Note that ``sign_bias`` and
            ``scale_bias`` shoule not be both given
        """
        super().__init__()
        setup_weight_bias(self, weight, sign_bias)
        self.output_numeric = sign_bias is None
        assert isinstance(rounding, bool)
        assert quant_step is None or not rounding
        self.rounding = rounding
        self.quant_step = quant_step

        if scale_bias is not None:
            assert (sign_bias is None and
                    self.bias is None and self.sign == 1)
            del self.bias
            del self.sign
            scale, bias = scale_bias
            scale = float(scale.item())
            assert isinstance(bias, torch.Tensor)
            self.register_buffer('bias', bias / scale)
            self.register_buffer(
                'sign',
                torch.from_numpy(
                    np.array([1 if scale > 0 else -1] * weight.size(0),
                             dtype=np.float32),
                ).to(device=weight.device))
            self.output_numeric_scale = scale

    @property
    def full_layer_name(self):
        return f'lin{self.layer_name}'

    def forward(self, x):
        if isinstance(x, (BoolTensor, QuantizedInputRangeTensor)):
            if isinstance(x, QuantizedInputRangeTensor):
                assert x.quant_step == self.quant_step, (
                    f'quant step mismatch: {x.quant_step} vs {self.quant_step}')
            assert not self.output_numeric and self.bias is not None
            bias, sign = (torch_as_npy(self.bias),
                          torch_as_npy(self.sign).flatten())
            return x.matmul(
                self.full_layer_name,
                torch_as_npy(self.weight.T), bias, sign)
        if self.quant_step is not None:
            x = torch.round_(x / self.quant_step)
        if self.output_numeric:
            return (F.linear(x, self.weight, self.bias) *
                    self.output_numeric_scale)
        return run_fwd(x, self, F.linear)

    def infer_out_shape(self, inp_shape):
        assert len(inp_shape) == 2, f'bad input shape {inp_shape}'
        n, _ = inp_shape
        return n, self.output_dim

    def forward_sym_lt(self, x: BoolTensor, idx0: int, idx1: int):
        """get a bool expression that ``output[idx0] < output[idx1]``"""
        assert self.bias is not None and self.output_numeric
        weight = torch_as_npy(self.weight.T)
        bias, sign = torch_as_npy(self.bias), torch_as_npy(self.sign).flatten()

        s0 = sign[idx0]
        s1 = sign[idx1]
        w0 = weight[:, idx0] * s0
        w1 = weight[:, idx1] * s1
        b0 = bias[idx0] * s0
        b1 = bias[idx1] * s1

        w = w1 - w0
        b = np.floor(b1 - b0 - 1e-6)

        return x.matmul('clsfy', w[:, np.newaxis], b.reshape(1),
                        np.array(1, dtype=np.int32).reshape(1))

    @classmethod
    def from_lin_bn(cls, lin: nn.Linear, bn: nn.BatchNorm2d, *,
                    quant_step=None):
        """make a conv layer from conv+bn+Binarize01"""
        assert lin.bias is None
        return cls(lin.weight_bin,
                   get_sign_bias_from_bn(lin, bn, scale=quant_step),
                   scale_bias=None, rounding=lin.rounding,
                   quant_step=quant_step)

    @property
    def output_dim(self):
        return self.weight.size(0)


def run_model_on_npy(model: nn.Module, data: np.ndarray, *, add_batch=False):
    device = next(iter(model.buffers())).device
    if add_batch:
        data = data[np.newaxis]
    ret = torch_as_npy(
        model(torch.from_numpy(data.astype(np.float32)).to(device)))
    if add_batch:
        ret, = ret
    return ret

def meanpooling_nhw(inp, scale):
    assert inp.ndim == 3, f'bad shape: {inp.shape}'
    n, h, w = inp.shape
    coeff = np.ones_like(inp)
    pad_h = (-inp.shape[1]) % scale
    pad_w = (-inp.shape[2]) % scale
    pw = ((0, 0), (0, pad_h), (0, pad_w))
    h1 = (h + pad_h) // scale
    w1 = (w + pad_w) // scale

    def pool(x):
        x = np.pad(x, pw)
        x = x.reshape(n, h1, scale, w1, scale)
        x = np.transpose(x, (0, 1, 3, 2, 4)).reshape(n, h1, w1, scale * scale)
        return x.sum(axis=3)

    return pool(inp) / pool(coeff)

def reorder_output_coords(model, inp_sym, downscale=2, sample_size=16,
                          rng=None):
    """return a generator for output coordinates according to neuron
    stability"""
    rng = get_rng_if_none(rng)
    assert isinstance(inp_sym, QuantizedInputRangeTensor)
    inp_upper = np.frompyfunc(len, 1, 1)(inp_sym.value) + 1
    n, c, h, w = inp_sym.shape
    inp = np.floor(rng.random((n, sample_size, c, h, w)) *
                   inp_upper[:, np.newaxis])
    inp += inp_sym.offset[:, np.newaxis]
    inp = inp.reshape(n * sample_size, c, h, w)
    out = run_model_on_npy(model, inp)
    assert (out.min() >= 0 and out.max() <= 1 and
            np.all(np.abs(out - 0.5) >= 0.49))
    out = out.reshape(n, sample_size, *out.shape[1:])
    prob = out.mean(axis=1)
    entropy = -(prob * np.log2(np.maximum(prob, 1e-9)) +
               (1 - prob) * np.log2(np.maximum(1 - prob, 1e-9)))
    avg_entropy = meanpooling_nhw(entropy.mean(axis=1), downscale)

    for i_n in range(n):
        # from stable (lower entropy) to unstable (higher entropy)
        ind = np.unravel_index(np.argsort(avg_entropy[i_n], axis=None),
                               avg_entropy.shape[1:])
        for x, y in zip(*ind):
            yield i_n, x * downscale, y * downscale

def setup_var_preference_first(env, inp_sym):
    vpf = env.main_args.var_preference
    if vpf == 'none':
        return

    if vpf == 'first':
        for i in inp_sym.value.flat:
            for j in i:
                env.set_var_preference(j, -1)
        return

    if vpf == 'first-sp':
        v = inp_sym.value
        N, C, H, W = v.shape
        cnt = 0
        r4x4nc = list(iter_prod_range(4, 4, N, C))
        for h0, w0 in itertools.product(range(0, H, 4), range(0, W, 4)):
            for i, j, n, c in r4x4nc:
                h = h0 + i
                w = w0 + j
                if h < H and w < W:
                    cnt += 1
                    for x in v[n, c, h, w]:
                        env.set_var_preference(x, -cnt)
        assert cnt == v.size
        return

    raise RuntimeError(f'bad var pref setting {vpf}')

def setup_var_preference(env, model, ftr_syms):
    """setup var branching preference by iterating over 2x2 output features and
    the corresponding receiptive fields"""
    assert len(model) + 1 == len(ftr_syms)

    if env.main_args.var_preference in VAR_PREFERENCE_FIRST_COMPATIBLE:
        return setup_var_preference_first(env, ftr_syms[0])

    inp_sym: np.ndarray = ftr_syms[0].value
    out_sym: np.ndarray = ftr_syms[-1].value

    if env.main_args.var_preference == 'mid-first':
        for i in out_sym.flat:
            env.set_var_preference(i, -2)
        for i in inp_sym.flat:
            for j in i:
                env.set_var_preference(j, -1)
        return

    def map_back_x(omin, omax):
        return omin, omax + 1
    map_back_y = map_back_x

    def make_map_back(prev_fn, prev_size, stride, padding, kern):
        def map_back(omin, omax):
            imin = max(omin * stride - padding, 0)
            imax = min(omax * stride - padding + kern - 1, prev_size - 1)
            return prev_fn(imin, imax)
        return map_back

    for idx, layer in enumerate(model):
        assert isinstance(layer, BinConv2dEval), (
            f'unsupported early stage layer {layer}')
        sx, sy = layer.stride
        px, py = layer.padding
        kx, ky = layer.weight.shape[2:]
        ix, iy = ftr_syms[idx].shape[2:]
        map_back_x = make_map_back(map_back_x, ix, sx, px, kx)
        map_back_y = make_map_back(map_back_y, iy, sy, py, ky)

    pref = -(np.frompyfunc(len, 1, 1)(inp_sym).sum() + out_sym.size + 1)
    inp_mark = np.zeros(inp_sym[:, 0].shape, dtype=np.uint8)
    IC = inp_sym.shape[1]
    N, OC, OH, OW = out_sym.shape

    def set_out(n, oh, ow, oc):
        nonlocal pref
        if oh < OH and ow < OW:
            env.set_var_preference(out_sym[n, oc, oh, ow], pref)
            pref += 1

    if env.main_args.var_preference == 'z':
        output_coord_iter = itertools.product(
            range(N), range(0, OH, 2), range(0, OW, 2))
    else:
        assert env.main_args.var_preference == 'z-ent'
        output_coord_iter = reorder_output_coords(model, ftr_syms[0])

    for n, oh, ow in output_coord_iter:
        for oc in range(OC):
            set_out(n, oh, ow, oc)
            set_out(n, oh + 1, ow, oc)
            set_out(n, oh, ow + 1, oc)
            set_out(n, oh + 1, ow + 1, oc)

        for ih, iw in itertools.product(
                range(*map_back_x(oh, min(oh + 1, OH - 1))),
                range(*map_back_y(ow, min(ow + 1, OW - 1)))):

            if inp_mark[n, ih, iw]:
                continue

            inp_mark[n, ih, iw] = 1
            for ic in range(IC):
                for i in inp_sym[n, ic, ih, iw]:
                    env.set_var_preference(i, pref)
                    pref += 1

    assert np.all(inp_mark)
    assert pref == -1

def make_sat_env(args):
    if args.sat_solver == 'z3':
        from .sat_z3 import Z3SatEnv as SatEnv
    elif args.sat_solver == 'roundingsat':
        from .sat_roundingsat import RoundingSatEnv as SatEnv
    elif args.sat_solver == 'pysat':
        from .sat_pysat import PySatEnv
        SatEnv = lambda: PySatEnv(args.pysat_name,
                                  need_record=bool(args.write_formula))
    else:
        assert args.sat_solver == 'pysat_stat'
        from .sat_pysat import PySatStatEnv
        SatEnv = lambda: PySatStatEnv(args.pysat_name)

    env = SatEnv()
    if args.verbosity:
        env.set_verbosity(args.verbosity)
    if args.timeout:
        env.set_timeout(args.timeout)
    assert env.tensor_arith_impl is not None
    if args.disable_fast_tensor_arith:
        env.tensor_arith_impl = None
    env.main_args = args
    return env

def encode_input_linf(env, quant_step: float, inp_num: np.ndarray,
                      eps: float, *, range_min=0, range_max=1):
    """encode for L-inf bounded input perturbations"""
    assert 0 < quant_step < 1
    inp_min = np.clip(inp_num - eps, range_min, range_max)
    inp_max = np.clip(inp_num + eps, range_min, range_max)
    inp_sym = QuantizedInputRangeTensor.from_value_range(
        env, inp_min, inp_max, quant_step, 'inp'
    )
    return inp_sym

def encode_model(env, model: nn.Module, inp_sym: np.ndarray):
    """add constraints to SAT env for a model

    :return: feature sym, final classification layer
    """
    internal_ftr_syms = []
    ftr_sym = inp_sym
    for i in range(len(model) - 1):
        internal_ftr_syms.append(ftr_sym)
        if i == len(model) // 2:
            # start searching in the middle
            setup_var_preference(env, model[:i], internal_ftr_syms)
        ftr_sym = model[i](ftr_sym)

    return ftr_sym, model[-1]

def encode_attack(env, ftr_sym: np.ndarray, label: np.ndarray,
                  clsfy_layer: BinLinearEval, strict: bool):
    """encode constraints to SAT env for untargeted attack

    :param strict: whether to require output class have higher score than all
        other classes (vs higher score only then target class)
    :return: ``[props]`` where ``props`` is the lits to be disjuncted for attack
        the corresponding batch
    """

    assert (ftr_sym.ndim == 2 and label.ndim == 1 and
            ftr_sym.shape[0] == label.shape[0]), (
                f'result_shape={ftr_sym.shape} label_shape={label.shape}')

    props_attack = []
    if not strict:
        for i in range(label.shape[0]):
            found = False
            this_label = int(label[i])
            props = []
            for j in range(clsfy_layer.output_dim):
                if j == this_label:
                    found = True
                else:
                    v = clsfy_layer.forward_sym_lt(
                        ftr_sym[i:i+1], this_label, j)
                    v, = v.value.flat
                    props.append(v)
            assert found
            props_attack.append(props)
    else:
        for i in range(label.shape[0]):
            found = False
            this_label = int(label[i])
            props = []
            for j in range(clsfy_layer.output_dim):
                if j == this_label:
                    found = True
                    continue

                all_lt = [
                    clsfy_layer.forward_sym_lt(ftr_sym[i:i+1], k, j).value.flat
                    for k in range(clsfy_layer.output_dim) if k != j
                ]
                all_lt = [i for (i, ) in all_lt]    # remove batch
                props.append(encode_reified_and(env, all_lt))
            assert found
            props_attack.append(props)

    return props_attack


def check_model_cvt(testloader, device, model0, model1):
    """check whether conversion from ``model0`` to ``model1`` is correct"""
    fail_msg = None
    nr_test_correct0 = 0
    nr_test_correct1 = 0
    nr_test_tot = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs0 = model0(inputs)
        outputs1 = model1(inputs)
        nr_test_correct0 += get_nr_correct(outputs0, labels)
        nr_test_correct1 += get_nr_correct(outputs1, labels)
        nr_test_tot += inputs.size(0)

        if fail_msg:
            continue

        diff = ((outputs0 - outputs1).abs_().
                view(outputs0.size(0), -1).max(dim=1)[0])
        dmax = float(diff.max().item())
        if dmax >= 1e-4:
            bad_idx = int(torch.argmax(diff).item())
            inp = inputs[bad_idx:bad_idx+1]
            out0, = torch_as_npy(model0(inp))
            out1, = torch_as_npy(model1(inp))
            fail_msg = (f'model check failed: diff={dmax:.4}\n'
                        f'  {out0=}\n  {out1=}')

            if np.abs(out0 - out1).max() < dmax - 1e-4:
                # check if model result on a batch is different from result
                # on a single input
                out_whole = inputs
                out_sub = inp
                for idx, layer in enumerate(model1):
                    out_whole = layer(out_whole)
                    out_sub = layer(out_sub)
                    a = out_whole[bad_idx]
                    b = out_sub[0]
                    if (a - b).abs_().max() > 1e-4:
                        fail_msg += (
                            f'\n  model output unstable: layer={idx}\n'
                            f'   out_whole={torch_as_npy(a).flatten()}\n'
                            f'   out_sub  ={torch_as_npy(b).flatten()}\n'
                        )
                        break

    def percent(x):
        return x / nr_test_tot * 100
    print(f'test acc: orig={percent(nr_test_correct0):.2f}% '
          f'cvt={percent(nr_test_correct1):.2f}%')
    if fail_msg:
        print(f'*** WARNING *** {fail_msg}')
    # accuracy degeneration <= 1%
    assert (nr_test_correct0 - nr_test_correct1) < nr_test_tot * 0.01

def encode_reified_and(env, props, name='and'):
    ret = env.new_var(name)
    neg = []
    for p in props:
        neg.append(-p)
        env.add((-ret) | p)
    env.add_reduce_or(neg + [ret])
    return ret

def encode_ensemble_attack(env, props):
    assert props
    if len(props) == 1:
        env.add_reduce_or(props[0])
        return

    nr = len(props[0])
    atk_vars = []
    for p in props:
        assert len(p) == nr

    for i in range(nr):
        y = encode_reified_and(env, [p[i] for p in props], f'attack{i}')
        atk_vars.append(y)
    env.add_reduce_or(atk_vars)

class ModelVerifier:
    """a classification model verifier that acts according to the given args
    parsed by main. It is very stateful"""
    args = None
    models = None

    enable_score_check = True

    _results = None
    _next_result_update_time = 0
    _nr_robust = 0
    _max_solve_time = 0
    _cached_clause_info = None

    def __init__(self, args, models):
        self.args = args
        self.models = models
        for i in models:
            assert i.quant_step == models[0].quant_step
        self._results = {}

    def load_result(self, fin):
        self._results = json.load(fin)
        self._nr_robust = len([i for i in self._results.values()
                                 if i['result'] == 'UNSAT'])
        return len(self._results)

    def check_and_save(self, data_idx: int, inp: np.ndarray, label: np.ndarray):
        """run :meth:`check` and write the result to given result file"""
        check_result, (time_b, time), _ = self.check(inp, label)

        self._results[data_idx] = {
            'result': str(check_result),
            'build_time': time_b,
            'solve_time': time}

        self.update_result_file()
        nr_robust = self._nr_robust
        tot = len(self._results)
        self.log(f'#{tot}: result={check_result:5} time={time:.3f}s '
                 f'accum={nr_robust}/{tot}={nr_robust/max(tot, 1)*100:.2f}%')

    def log(self, msg):
        """write message to log; this method can be overwritten directly"""
        print(msg, flush=True)

    def check(self, inp: np.ndarray, label: np.ndarray):
        """verify model on a single input without writing to result file

        :param inp: ``(C, H, W)`` input image
        :param label: target class number

        :return: (check_result, (time_build, time_verify), inp_adv)
        """
        assert inp.ndim == 3 and label.size == 1
        env, check_result, (time_b, time), clsfy_scores, inp_adv = self._work(
            inp[np.newaxis], label.reshape((1, )))

        args = self.args

        def check_score(clsfy_scores, inp_adv):
            if not self.enable_score_check:
                return

            class_get_uniq = None
            for clsfy_score, model in zip(clsfy_scores, self.models):
                score_check = run_model_on_npy(model, inp_adv, add_batch=True)
                diff = np.abs(inp_adv - inp).max()
                class_get = np.argmax(clsfy_score)
                if class_get_uniq is None:
                    class_get_uniq = class_get
                self.log(f'numerical={clsfy_score} class={class_get} '
                         f'label={label} diff={diff:.2}/{args.eps}')
                err = np.abs(clsfy_score - score_check).max()
                assert (err < 1e-4 and diff - 1e-5 <= args.eps
                        and class_get != label
                        and class_get == class_get_uniq), (
                            f'{score_check=} {err=}')
            if args.show_adv:
                import cv2
                cv2.imshow('orig', npy_chw_f32_to_cv2_uint8(inp))
                cv2.imshow('adv', npy_chw_f32_to_cv2_uint8(inp_adv))
                if chr(cv2.waitKey(-1) & 0xFF) == 'q':
                    sys.exit(0)

            if args.write_adv:
                ensure_dir(args.write_adv)
                fname = Path(args.write_adv) / f'adv-{len(self._results)}.pkl'
                with fname.open('wb') as fout:
                    pickle.dump({'inp': inp, 'inp_adv': inp_adv,
                                 'label': int(label)},
                                fout,
                                pickle.HIGHEST_PROTOCOL)
                self.log(f'adversarial example written to {fname}')

        if check_result == SolveResult.UNSAT:
            self._nr_robust += 1
        elif check_result == SolveResult.SAT:
            inp_adv, = inp_adv # remove the batch dim
            check_score([i for (i,) in clsfy_scores],
                        inp_adv)
        else:
            assert check_result == SolveResult.TLE


        if args.write_slow_formula and time > self._max_solve_time:
            with open(args.write_slow_formula, 'w') as fout:
                env.write_formula(fout)
            self.log(f'slow formula updated: {self._max_solve_time:.3f}s '
                     f'to {time:.3f}s')
            self._max_solve_time = time

        return check_result, (time_b, time), inp_adv

    def update_result_file(self, force=False):
        args = self.args
        if not args.write_result:
            return
        if not force and time.time() < self._next_result_update_time:
            return

        gc.collect()
        with open(args.write_result, 'w') as fout:
            stopped = False
            while True:
                try:
                    json.dump(self._results, fout, indent=2)
                    break
                except KeyboardInterrupt:
                    # ignore interrupt during dump, try again
                    stopped = True
                    continue
            assert not stopped, 'canceled by SIGINT'
        self._next_result_update_time = time.time() + 10

    def _build_sat_env(self, env, inp_num, label):
        if self._cached_clause_info is not None:
            return self._build_sat_env_with_cache(env, inp_num, label)
        if not self.args.disable_model_cache:
            if env.make_clause_recorder() is None:
                self.args.disable_model_cache = True
            else:
                return self._build_sat_env_with_cache(env, inp_num, label)

        inp_sym = encode_input_linf(env, self.models[0].quant_step,
                                    inp_num, self.args.eps)
        return self._build_sat_env_combine_model(
            inp_sym, label,
            (encode_model(env, i, inp_sym) for i in self.models))

    def _build_sat_env_combine_model(self, inp_sym, label, miter):
        """combine model results and add attack target

        :param miter: iterator of model output feature, model classification
            layer
        :return ``inp_sym``, all ftrs, all clsfy layers
        """
        env = inp_sym.env
        all_ftr_syms = []
        all_clsfy_layers = []
        all_props_attack = []

        if self.args.no_attack:
            attack_level = 0
        else:
            attack_level = min(len(self.models), 2)

        for ftr_sym, clsfy in miter:
            all_ftr_syms.append(ftr_sym)
            all_clsfy_layers.append(clsfy)
            if attack_level:
                all_props_attack.append(encode_attack(
                    env, ftr_sym, label, clsfy, attack_level >= 2))

        if attack_level:
            encode_ensemble_attack(
                env,
                [i for i, in all_props_attack]  # remove batch dim
            )

        return inp_sym, all_ftr_syms, all_clsfy_layers

    def _build_sat_env_with_cache(self, env, inp_num, label):
        first_layer_type = (BinConv2dEval, BinLinearEval)
        if self._cached_clause_info is None:
            need_replay = False
            assert (self.args.var_preference in
                    VAR_PREFERENCE_FIRST_COMPATIBLE), (
                        'currently only first-compatible '
                        'var_preference is supported for clause cache'
                    )
            ftr_first = []
            ftr_last = []
            rec = env.make_clause_recorder()
            with rec.scoped_attach_to(env):
                for model_num, model in enumerate(self.models):
                    oshp = inp_num.shape
                    for start, l in enumerate(model):
                        oshp = l.infer_out_shape(oshp)
                        if isinstance(l, first_layer_type):
                            break
                    xname = l.full_layer_name
                    if len(self.models) > 1:
                        xname = f'm{model_num}_{xname}'
                    x = BoolTensor.from_empty(env, xname, oshp)
                    y = model[start+1:-1](x)
                    ftr_first.append(x)
                    ftr_last.append(y)
            self._cached_clause_info = (rec, ftr_first, ftr_last)
        else:
            need_replay = True

        recorder, all_ftr_first, all_ftr_last = self._cached_clause_info
        if need_replay:
            recorder.replay(env)
            for i in itertools.chain(all_ftr_first, all_ftr_last):
                i.env = env

        inp_sym = encode_input_linf(env, self.models[0].quant_step,
                                    inp_num, self.args.eps)

        for model, ftr_first in zip(self.models, all_ftr_first):
            with env.set_future_new_vars(ftr_first.value.flatten()):
                recomp = inp_sym
                for l in model:
                    recomp = l(recomp)
                    if isinstance(l, first_layer_type):
                        break

        setup_var_preference_first(env, inp_sym)
        return self._build_sat_env_combine_model(
            inp_sym, label,
            ((i, j[-1]) for i, j in zip(all_ftr_last, self.models)))

    def _work(self, inp_num, label):
        assert inp_num.min() >= 0 and inp_num.max() <= 1

        args = self.args

        env = make_sat_env(args)

        # build SAT env
        t0 = time.time()
        inp_sym, all_ftr_syms, all_clsfy_layers = self._build_sat_env(
            env, inp_num, label)
        if args.write_formula:
            with open(args.write_formula, 'w') as fout:
                env.write_formula(fout)
        build_time = time.time() - t0
        self.log(f'build: time={build_time:.3f}s ' +
                 'avg_free_vars={:.2f}'.format(
                     np.frompyfunc(len, 1, 1)(inp_sym.value).mean()))

        # solve
        t1 = time.time()
        check_result = env.solve()
        solve_time = time.time() - t1
        if (t := env.get_prev_solve_time()) is not None:
            solve_time = t

        # retrieve results
        if check_result == SolveResult.SAT:
            clsfy_scores = [
                run_model_on_npy(layer, ftr.as_npy(dtype=np.float32))
                for layer, ftr in zip(all_clsfy_layers, all_ftr_syms)
            ]
            inp_adv_num = inp_sym.as_npy()
            inp_adv_num = np.clip(inp_adv_num,
                                  inp_num - args.eps, inp_num + args.eps,
                                  out=inp_adv_num)
            inp_adv_num = np.clip(inp_adv_num, 0, 1, out=inp_adv_num)
        else:
            clsfy_scores = inp_adv_num = None

        return (env, check_result, (build_time, solve_time),
                clsfy_scores, inp_adv_num)

def init_argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--show-adv', action='store_true',
                        help='show discovered adversarial images')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8,
                        help='number of CPU workers for data augmentation')
    parser.add_argument(
        '--data', default=default_dataset_root(),
        help='dir for training data')
    parser.add_argument('--check-cvt', action='store_true',
                        help='check accuracy of converted model')
    parser.add_argument('--no-attack', action='store_true',
                        help='do not run attack and only check result')
    parser.add_argument('--sat-solver', choices=['z3', 'pysat', 'roundingsat',
                                                 'pysat_stat'],
                        default='pysat', help='SAT solver implementation')
    parser.add_argument('--pysat-name', default='minisatcs',
                        help='pysat solver name')
    parser.add_argument('--write-formula',
                        help='write internal formula to file; use @file.json '
                        'to write to the same directory of model file')
    parser.add_argument('--write-slow-formula',
                        help='write formula to file for the currently slowest '
                        'case')
    parser.add_argument('--write-result',
                        help='write result to a json file')
    parser.add_argument('--write-adv',
                        help='write adversarial results to a directory')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='set verbosity level')
    parser.add_argument('-t', '--timeout', type=float,
                        help='timeout in seconds')
    parser.add_argument('--skip', type=int,
                        help='skip given number of test examples')
    parser.add_argument('--num-cases', type=int,
                        help='set the number of cases to run')
    parser.add_argument('--var-preference', default='first-sp',
                        choices=['z', 'z-ent', 'mid-first', 'first',
                                 'first-sp', 'none'],
                        help='set var preference type; [z]: mid layer to first '
                        'layer spatial locality; [z-ent]: z guided by entropy; '
                        '[mid-first]: all mid layer and then first; '
                        '[first]: first layer; [first-sp]: first with spatial '
                        'locality; [none]: solver default'
                        )
    parser.add_argument('--continue', action='store_true',
                        help='set --skip value based on number of existing '
                        'entries in the result file')
    parser.add_argument('--random-sample', type=int,
                        help='sample a given number of test cases')
    parser.add_argument('--log-file', help='redirect output to given log file')
    parser.add_argument('--ensemble', default=[], action='append',
                        help='add another model to be ensembled')
    parser.add_argument('--disable-model-cache', action='store_true',
                        help='disable caching clauses of the models')
    parser.add_argument('--disable-fast-tensor-arith', action='store_true',
                        help='disable fast tensor arith impl, for checking '
                        'correctness')
    parser.add_argument('--verify-output-stability', action='store_true',
                        help='verify if model prediction is stable '
                        '(i.e. use model output as the label)')
    return parser

def main():
    parser = argparse.ArgumentParser(
        description='evaluate robustness of 0/1 binaried network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    init_argparser(parser)
    parser.add_argument('-e', '--eps', type=float, required=True,
                        help='adv input eps on 0-1 scale')
    parser.add_argument('model')
    args = parser.parse_args()

    with torch.no_grad():
        if args.log_file:
            with open(args.log_file, 'a') as fout:
                with redirect_stdout(fout):
                    with redirect_stderr(fout):
                        return main_impl(args)
        else:
            return main_impl(args)

def random_sample(testloader, num):
    rng = np.random.RandomState(92702102)
    tot = len(testloader.dataset)
    idx = rng.choice(len(testloader.dataset), size=num, replace=False)
    idx = set(map(int, idx))

    cur_idx = 0
    ret_x = None
    ret_y = None
    write_pos = 0
    def make_ret_like(v: torch.Tensor):
        shp = list(map(int, v.size()))
        shp.insert(0, num)
        return torch.empty(shp, dtype=v.dtype, device=v.device)

    idx_map = {}
    for x0, y0 in testloader:
        for x, y in zip(x0, y0):
            if cur_idx in idx:
                if ret_x is None:
                    ret_x = make_ret_like(x)
                    ret_y = make_ret_like(y)
                ret_x[write_pos] = x
                ret_y[write_pos] = y
                idx_map[write_pos] = cur_idx
                write_pos += 1
            cur_idx += 1
    assert cur_idx == tot and write_pos == num
    return idx_map, ((ret_x, ret_y), )

def load_model_and_cvt(args, device, path):
    orig_model = ModelHelper.create_with_load(path).to(device).eval()
    print(f'loaded {type(orig_model)} quant_step={orig_model.features[0].step}')
    model = orig_model.cvt_to_eval()
    for idx, layer in enumerate(model):
        layer.layer_name = str(idx)

    if model.is_mlp and args.var_preference in ['z', 'z-ent']:
        # var preference for MLP can not be z
        args.var_preference = 'first'
        print('WARNING: change var preference of MLP model to first')

    testloader = orig_model.make_dataset_loader(args, False)

    if args.check_cvt:
        print(f'original model {orig_model}')
        print(f'converted model {model}')
        with torch.no_grad():
            check_model_cvt(testloader, device, orig_model, model)
    return model, testloader

def get_ensemble_acc(device, testloader, models):
    nr_test_correct = 0
    nr_test_tot = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output_merged = None
        for model in models:
            out = model(inputs)
            out = torch.argmax(out, dim=1)
            if output_merged is None:
                output_merged = out
            else:
                eq = (out == output_merged).to(out.dtype)
                output_merged = output_merged.mul_(eq).add_((1 - eq).mul_(-1))
        nr_test_correct += int(
            (output_merged == labels).to(torch.int32).sum().item())
        nr_test_tot += inputs.size(0)
    return nr_test_correct / nr_test_tot

def main_impl(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'argv: {sys.argv}')
    print(f'parsed args: {args}')

    if args.write_result and args.write_result.startswith('@'):
        assert os.path.sep not in args.write_result
        args.write_result = str(Path(args.model).parent / args.write_result[1:])

    np.set_printoptions(precision=1, suppress=True)

    if args.ensemble and args.var_preference not in ['none', 'first']:
        args.var_preference = 'first'
        print('WARNING: change var preference of model ensemble to first')

    master_model, testloader = load_model_and_cvt(args, device, args.model)
    models = [master_model]
    for i in args.ensemble:
        models.append(load_model_and_cvt(args, device, i)[0])

    if args.check_cvt and len(models) > 1:
        with torch.no_grad():
            acc = get_ensemble_acc(device, testloader, models)
        print(f'ensemble test acc: {acc*100:.2f}%')
        if args.write_result:
            with open(args.write_result + '.ensemble_acc', 'w') as fout:
                print(acc, file=fout)

    verifier = ModelVerifier(args, models)

    if getattr(args, 'continue'):
        assert not args.skip and args.write_result
        if (p := Path(args.write_result)).exists():
            with p.open() as fin:
                args.skip = verifier.load_result(fin)
                print(f'continue from {args.skip}')

    if args.random_sample:
        idx_map, testloader = random_sample(testloader, args.random_sample)
        idx_getter = idx_map.__getitem__
    else:
        idx_getter = lambda x: x

    idx = 0
    for inputs, labels in testloader:
        if args.verify_output_stability:
            labels = torch.argmax(master_model(inputs.to(device)), dim=1)
        inputs = torch_as_npy(inputs)
        labels = torch_as_npy(labels)
        for i in range(inputs.shape[0]):
            if args.skip:
                args.skip -= 1
            else:
                if args.num_cases is not None:
                    if args.num_cases <= 0:
                        break
                    args.num_cases -= 1
                verifier.check_and_save(idx_getter(idx), inputs[i], labels[i])
            idx += 1


    verifier.update_result_file(force=True)
    if args.write_result:
        with open(args.write_result + '.finished', 'w') as fout:
            print(json.dumps(sys.argv), file=fout)
            print(args, file=fout)

if __name__ == '__main__':
    main()

