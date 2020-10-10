"""solve SAT using z3"""

from .satenv import SatEnv, BoolRef, SolveResult

import z3
import ctypes
import time

def as_z3_ast_array(props, ctx):
    from z3.z3 import _to_ast_array
    vals = [i.val for i in props]
    assert all(i.ctx is ctx for i in vals)
    return _to_ast_array(vals)

def as_z3_ast_array_coeffs(props, ctx):
    props, weights = zip(*props)
    args, sz = as_z3_ast_array(props, ctx)
    w = (ctypes.c_int * sz)()
    w[:] = weights
    return args, w, sz

def z3_reduce(op, props, ctx):
    args, sz = as_z3_ast_array(props, ctx)
    return Z3BoolRef(z3.BoolRef(op(ctx.ref(), sz, args), ctx))

class Z3BoolRef(BoolRef):
    __slots__ = ['val']

    def __init__(self, val):
        self.val = val

    def __neg__(self):
        v = self.val
        ctx = v.ctx
        return Z3BoolRef(z3.BoolRef(z3.Z3_mk_not(ctx.ref(), v.as_ast()), ctx))

    def __and__(self, rhs):
        assert type(rhs) is Z3BoolRef
        return z3_reduce(z3.Z3_mk_and, [self, rhs], self.val.ctx)

    def __or__(self, rhs):
        assert type(rhs) is Z3BoolRef
        return z3_reduce(z3.Z3_mk_or, [self, rhs], self.val.ctx)


class Z3SatEnv(SatEnv):
    _ctx = None
    _solver = None
    _model = None
    _timeout = 0

    def __init__(self):
        self._ctx = z3.Context()
        self._solver = z3.Solver(ctx=self._ctx)

    def add(self, v):
        assert type(v) is Z3BoolRef
        self._solver.add(v.val)

    def add_reduce_or(self, props):
        self._solver.add(z3_reduce(z3.Z3_mk_or, props, self._ctx).val)

    def solve(self):
        t0 = time.time()
        r = self._solver.check()
        if r == z3.sat:
            self._model = self._solver.model()
            return SolveResult.SAT
        if r == z3.unsat:
            return SolveResult.UNSAT
        if r == z3.unknown:
            if time.time() - t0 < self._timeout - 1:
                raise RuntimeError('early return of solver')
            return SolveResult.TLE
        raise RuntimeError(f'bad z3 check() result: {r}')

    def get(self, v):
        assert self._model is not None
        assert type(v) is Z3BoolRef
        v = self._model.eval(v.val, model_completion=True)
        assert z3.is_bool(v)
        t = z3.is_true(v)
        f = z3.is_false(v)
        assert t or f, f'bad {v}'
        return int(t)

    def _do_new_var(self, name, varid):
        return Z3BoolRef(z3.Bool(f'{name}_{varid}', self._ctx))

    def _make_fast_eq(self, x, y):
        return z3.BoolRef(z3.Z3_mk_eq(self._ctx.ref(), x.as_ast(), y.as_ast()),
                          self._ctx)

    def _make_pb(self, op, props, k, name):
        args, coeffs, sz = as_z3_ast_array_coeffs(props, self._ctx)
        v = z3.BoolRef(op(self._ctx.ref(), sz, args, coeffs, k), self._ctx)
        r = self.new_var(name).val
        self._solver.add(self._make_fast_eq(r, v))
        return Z3BoolRef(r)

    @classmethod
    def _get_bound(cls, props):
        """return min and max possible values of the props"""
        sum_min = 0
        sum_max = 0
        for _, i in props:
            if i > 0:
                sum_max += i
            else:
                sum_min += i
        return sum_min, sum_max

    def _make_true(self, name):
        v = self.new_var(name)
        self._solver.add(v.val)
        return v

    def _make_false(self, name):
        v = self.new_var(name)
        self._solver.add((-v).val)
        return v

    def make_at_least(self, props, k: int, name: str):
        sum_min, sum_max = self._get_bound(props)
        if sum_min >= k:
            return self._make_true(name)
        if sum_max < k:
            return self._make_false(name)
        return self._make_pb(z3.Z3_mk_pbge, props, k, name)

    def make_at_most(self, props, k: int, name: str) -> BoolRef:
        sum_min, sum_max = self._get_bound(props)
        if sum_max <= k:
            return self._make_true(name)
        if sum_min > k:
            return self._make_false(name)
        return self._make_pb(z3.Z3_mk_pble, props, k, name)

    def make_empty_expr(self):
        return Z3BoolRef(z3.BoolVal(True, self._ctx))

    def write_formula(self, fout):
        fout.write(self._solver.sexpr())

    def set_timeout(self, timeout):
        self._solver.set('timeout', int(timeout * 1e3))
        self._timeout = timeout
