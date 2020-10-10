"""solve SAT using pysat"""

from .satenv import SatEnv, BoolRef, SolveResult, ClauseRecorder
from .utils import setup_pyx_import
with setup_pyx_import():
    from .sat_pysat_card import CardEncoder

from pysat.solvers import Solver as PySatSolver

import numpy as np
import json
import itertools
import threading
import time

class CryptoMiniSatSolver:
    _model = None

    def __init__(self):
        from pycryptosat import Solver
        self.solver = Solver()
        self.add_clause = self.solver.add_clause

    def delete(self):
        pass

    def solve(self):
        sat, sol = self.solver.solve()
        if sat:
            model = []
            for idx, val in enumerate(sol):
                if val is True:
                    model.append(idx)
                elif val is False:
                    model.append(-idx)
            self._model = model
        return sat

    def get_model(self):
        assert self._model is not None
        return self._model


class MinisatCsImporter:
    Solver = None

    @classmethod
    def get(cls):
        if cls.Solver is None:
            with setup_pyx_import():
                from ._minisatcs import Solver, ClauseRecorder
            cls.Solver = Solver
            cls.ClauseRecorder = ClauseRecorder
        return cls


class MiniSatCsSolver:
    def __init__(self):
        self.solver = MinisatCsImporter.get().Solver()
        self.add_clause = self.solver.add_clause
        self.solve = self.solver.solve
        self.get_model = self.solver.get_model

    def delete(self):
        pass


class MinisatCsClauseRecorder(ClauseRecorder):
    def __init__(self):
        self.recorder = MinisatCsImporter.get().ClauseRecorder()

    @classmethod
    def _get_solver_impl(cls, env):
        assert type(env) is PySatEnv
        s = env._solver
        assert type(s) is MiniSatCsSolver
        return s.solver

    def scoped_attach_to(self, env):
        return self.recorder(self._get_solver_impl(env))

    def replay(self, env):
        assert env._varcnt == 0, 'can only be replayed to a new env'
        self.recorder.replay(self._get_solver_impl(env))
        env._varcnt = self.recorder.nr_var


class PySatSolverWithTle(PySatSolver):
    def solve(self, timeout):
        is_tle = False
        def on_timeout():
            nonlocal is_tle
            is_tle = True
            self.interrupt()

        self.clear_interrupt()
        if timeout is not None:
            timer = threading.Timer(timeout, on_timeout)
            timer.start()
            ret = super().solve_limited()
            timer.cancel()
            if is_tle:
                ret = None
            return ret
        return super().solve()


class PySatEnv(SatEnv):
    _solver = None
    _card_encoder = None
    _model = None

    _call_record = None
    _all_clauses = None
    _var_pref = None
    _var_name = None

    _add_call_record = None
    _add_clause_record = None
    _timeout = None

    def __init__(self, name, need_record):
        encoder_name = None
        if '-' in name:
            name, encoder_name = name.split('-')

        if name == 'cryptominisat':
            self._solver = CryptoMiniSatSolver()
        elif name == 'minisatcs':
            self._solver = MiniSatCsSolver()
            self.set_verbosity = self._solver.solver.set_verbosity
            self._do_set_var_name = self._solver.solver.set_var_name
            self.make_clause_recorder = lambda: MinisatCsClauseRecorder()
        else:
            self._solver = PySatSolverWithTle(name)

        if need_record:
            self._call_record = []
            self._all_clauses = []
            self._var_pref = {}
            self._var_name = {}
            self._add_call_record = self._call_record.append
            self._add_clause_record = self._all_clauses.append
            def set_var_name(var, name, *, impl=self._do_set_var_name):
                self._var_name[var] = name
                impl(var, name)
            self._do_set_var_name = set_var_name
        else:
            self._add_call_record = lambda _: None
            self._add_clause_record = lambda _: None

        self._card_encoder = CardEncoder(self, encoder_name)

    def __del__(self):
        if self._solver is not None:
            self._solver.delete()

    def _add_clause(self, c):
        self._add_clause_record(c)
        self._solver.add_clause(c)

    def add(self, clause):
        """add a clause to the underlying solver"""
        if type(clause) is PySatBoolRef:
            for i in clause.clauses:
                self.add(i)
            return
        assert all(isinstance(i, int) for i in clause)
        self._add_clause(clause)
        self._add_call_record(['clause', clause])
        return self

    def add_reduce_or(self, props):
        c = []
        for i in props:
            assert type(i) is PySatBoolRef and i.var is not None
            c.append(i.var)
        self.add(c)
        return self

    def _do_new_var(self, name, varid):
        self._add_call_record(['new_var', name, varid])
        self._do_set_var_name(varid, name)
        return PySatBoolRef(var=varid)

    def make_vpool(self):
        """make an object to act as vpool for pysat"""
        return self._VPool(self)

    def solve(self):
        t0 = time.time()
        succ = self._solver.solve(self._timeout)

        if (succ is None or
                (self._timeout and (time.time() - t0) > self._timeout)):
            return SolveResult.TLE

        if succ:
            model = {}
            for i in self._solver.get_model():
                if i > 0:
                    model[i] = True
                else:
                    model[-i] = False
            self._model = model
            return SolveResult.SAT
        assert succ is False
        return SolveResult.UNSAT

    def get(self, v):
        assert self._model is not None, (
            'solve() must succeed before calling get')
        assert type(v) is PySatBoolRef
        if v.var is None:
            return -1
        if v.var > 0:
            return int(self._model[v.var])
        return int(not self._model[-v.var])

    def _normalize_pb(self, props, bound):
        lits = []
        for i, j in props:
            assert type(i) is PySatBoolRef and i.var is not None
            i = i.var
            if j == -2:
                j = 2
                i = -i
                bound += 2
            elif j == -1:
                j = 1
                i = -i
                bound += 1
            elif j == 0:
                continue
            if j == 1:
                lits.append(i)
            else:
                assert j == 2, f'bad weight {j}'
                lits.extend([i, i])
        return lits, bound

    def make_at_least(self, props, bound: int, name: str):
        lits, lbd = self._normalize_pb(props, bound)
        ret = self._card_encoder.make_geq(lits, lbd, name)
        self._add_call_record(
            ['at_least_assign', ret, [[i.var, j] for i, j in props], bound])
        return PySatBoolRef(var=ret)

    def make_at_most(self, props, bound: int, name: str):
        lits, lbd = self._normalize_pb(props, bound)
        ret = self._card_encoder.make_leq(lits, lbd, name)
        self._add_call_record(
            ['at_most_assign', ret, [[i.var, j] for i, j in props], bound])
        return PySatBoolRef(var=ret)

    def make_empty_expr(self) -> BoolRef:
        return PySatBoolRef(expr=[])

    def write_formula(self, fout):
        name = getattr(fout, 'name', '')
        if name.endswith('.json'):
            rec = self._call_record
            rec.append(['cnf', self._all_clauses])
            try:
                json.dump(self._call_record, fout, indent=2)
            finally:
                rec.pop()
        elif name.endswith('.cnf'):
            fout.write(f'p cnf {self._varcnt} {len(self._all_clauses)}\n')
            for k, v in self._var_name.items():
                fout.write(f'c vname {k} {v}\n')
            for i in self._all_clauses:
                fout.write(' '.join(map(str, i)))
                if len(i) >= 2 and i[-2] == '#':
                    fout.write('\n')
                else:
                    fout.write(' 0\n')
            if self._var_pref:
                fout.write('c vpref ')
                fout.write(' '.join('{} {}'.format(*i)
                                    for i in self._var_pref.items()))
                fout.write(' 0\n')
        else:
            raise RuntimeError(
                f'only .json and .cnf formats are supported; got {name}')

    def set_var_preference(self, var, pref):
        assert type(var) is PySatBoolRef, f'bad var: {var}'
        var = var.var
        if self._var_pref is not None:
            self._var_pref[var] = pref
        if type(self._solver) is MiniSatCsSolver:
            self._solver.solver.set_var_preference(var, pref)

    def set_var_name(self, var, name):
        assert type(var) is PySatBoolRef, f'bad var: {var}'
        self._do_set_var_name(var.var, name)

    def set_timeout(self, timeout):
        self._timeout = timeout

    def _do_set_var_name(self, var, name):
        pass


class PySatStatEnv(PySatEnv):
    """only gather stats for geq/leq equations"""

    class FakeCardEncoder:
        def __init__(self, owner):
            self.owner = owner

        def make_geq(self, lits, bound, name):
            return self.owner._do_make_geq(lits, bound, name)

        def make_leq(self, lits, bound, name):
            return self.owner._do_make_leq(lits, bound, name)

    _eqn_sizes = None
    _eqn_bounds = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, need_record=False)
        self._eqn_sizes = []
        self._eqn_bounds = []
        self._card_encoder = self.FakeCardEncoder(self)

    def _do_make_geq(self, lits, bound, name):
        if 0 < bound <= len(lits):
            self._eqn_sizes.append(len(lits) * bound)
            self._eqn_bounds.append(bound)
        return 1

    def _do_make_leq(self, lits, bound, name):
        if 0 <= bound < len(lits):
            self._eqn_sizes.append(len(lits) * (bound + 1))
            self._eqn_bounds.append(bound + 1)
        return 1

    def solve(self):
        def stat(v):
            return (f'min={v.min()} max={v.max()} mean={v.mean():.2f} '
                    f'med={int(np.median(v))} sum={v.sum()}')

        raise RuntimeError(
            'can not call solve() on PySatStatEnv\n'
            f'size:  {stat(np.array(self._eqn_sizes))}\n'
            f'bound: {stat(np.array(self._eqn_bounds))}'
        )

    def set_timeout(self, timeout):
        raise RuntimeError('do not set timeout on PySatStatEnv')


class PySatBoolRef(BoolRef):
    """it can be either a single var (encoded as an integer) or an expression (a
    list of clauses)"""

    __slots__ = ['expr', 'var']

    def __init__(self, *, expr=None, var=None):
        assert int(expr is None) + int(var is None) == 1
        if expr is not None:
            assert isinstance(expr, list) and (
                not expr or isinstance(expr[0][0], int)), f'bad expr {expr}'
            self.expr = expr
            self.var = None
        else:
            assert isinstance(var, int), f'bad var {var}'
            self.expr = None
            self.var = var

    def __neg__(self):
        assert self.var is not None, (
            'must be transformed into a single var before negation')
        return PySatBoolRef(var=-self.var)

    @property
    def clauses(self):
        if self.var is not None:
            return [[self.var]]
        return self.expr

    def __and__(self, rhs):
        assert type(rhs) is PySatBoolRef
        return PySatBoolRef(expr=self.clauses + rhs.clauses)

    def __or__(self, rhs):
        assert type(rhs) is PySatBoolRef
        assert self.var is not None or rhs.var is not None, (
            'one operand in OR must be a single var')
        if self.var is not None:
            c = rhs.clauses
            v = self.var
        else:
            c = self.clauses
            v = rhs.var
        return PySatBoolRef(expr=[i + [v] for i in c])

