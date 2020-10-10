"""solve SAT using the PB solver roundingsat"""

from .satenv import SatEnv, SolveResult, SatChecker
from .sat_pysat import PySatBoolRef

import itertools
import tempfile
import subprocess
import sys
from pathlib import Path

roundingsat_exe = str(Path(__file__).resolve().parent / 'roundingsat_repo' /
                      'build' / 'roundingsat')

CHECK = False
"""whether to check the roundingsat solution"""

class RoundingSatEnv(SatEnv):
    """a warpper for roundingsat"""
    _timeout = None
    _timeout_safe = None

    _constr = None
    """list of strings in the OPB format to represent the constraints"""

    _model = None
    """map from var id to value"""

    _prev_solve_time = None

    _checker = None

    def __init__(self):
        if not Path(roundingsat_exe).exists():
            raise RuntimeError('roundingsat is not compiled; please run '
                               'roundingsat_build.sh: '
                               f'{roundingsat_exe} does not exist')
        self._constr = []
        if CHECK:
            self._checker = SatChecker()

    def add(self, clause):
        if type(clause) is PySatBoolRef:
            for i in clause.clauses:
                self.add(i)
            return
        assert all(isinstance(i, int) for i in clause)

        def fmt_lit(x):
            assert x != 0
            if x > 0:
                return f'1 x{x}'
            return f'1 ~x{-x}'

        if CHECK:
            self._checker.add_or(clause)

        self._constr.append(' '.join(itertools.chain(
            map(fmt_lit, clause),
            ['>= 1']
        )))
        return self

    def solve(self):
        self._prev_solve_time = None
        self._model = None
        with tempfile.NamedTemporaryFile() as ftmp:
            with open(ftmp.name, 'w') as fout:
                self.write_formula(fout)
            try:
                result = subprocess.run(
                    [roundingsat_exe, ftmp.name, '--print-sol=1'],
                    capture_output=True, timeout=self._timeout_safe,
                    encoding='utf-8')
            except subprocess.TimeoutExpired:
                return SolveResult.TLE
            except subprocess.CalledProcessError as exc:
                print(f'solver failed: ret={exc.returncode}')
                print(f'stdout: {exc.stdout}')
                print(f'stderr: {exc.stderr}')
                raise

        if result.returncode not in [0, 10, 20]:
            print(f'solver failed: ret={result.returncode}')
            print(f'stdout: {result.stdout}')
            print(f'stderr: {result.stderr}')
            raise RuntimeError('roundingsat failed')

        ret = None
        for line in result.stdout.split('\n'):
            if line.startswith('c total solve time'):
                p = line.split()
                assert p[-1] == 's'
                self._prev_solve_time = float(p[-2])
                if (self._timeout is not None and
                        self._prev_solve_time > self._timeout):
                    return SolveResult.TLE
            if line.startswith('s '):
                st_map = {
                    'UNKNOWN': SolveResult.TLE,
                    'SATISFIABLE': SolveResult.SAT,
                    'UNSATISFIABLE': SolveResult.UNSAT,
                }
                ret = st_map[line.split()[1]]
            if line.startswith('v '):
                self._model = {}
                for item in line[2:].split():
                    if item.startswith('-'):
                        assert item[1] == 'x'
                        self._model[int(item[2:])] = False
                    else:
                        assert item[0] == 'x'
                        self._model[int(item[1:])] = True
        if ret == SolveResult.SAT:
            assert self._model is not None
            if CHECK:
                self._checker.check(lambda x:
                                    int(self._model[x]) if x > 0 else
                                    int(not self._model[-x]))
        return ret

    def get(self, v):
        assert self._model is not None, (
            'solve() must succeed before calling get')
        assert type(v) is PySatBoolRef and v.var is not None
        if v.var > 0:
            return int(self._model[v.var])
        return int(not self._model[-v.var])

    def get_prev_solve_time(self):
        return self._prev_solve_time

    def add_reduce_or(self, props):
        c = []
        for i in props:
            assert type(i) is PySatBoolRef and i.var is not None
            c.append(i.var)
        self.add(c)
        return self

    def _do_new_var(self, name, varid):
        return PySatBoolRef(var=varid)

    def _do_make_at_least(self, props, bound: int, name: str):
        yvar = self.new_var(name)
        y = yvar.var
        wsum = 0

        # y = (x >= b), 0 <= x <= n :
        # 1. (x < b => y = 0) <=> x + b~y >= b
        # 2. (y = 0 => x < b) <=> (n-b+1)y - 1 >= x - b

        expr_pos = []
        expr_neg = []
        for lit, w in props:
            assert (type(lit) is PySatBoolRef and lit.var is not None
                    and type(w) is int)
            lit = lit.var
            if w < 0:
                lit = -lit
                bound -= w
                w = -w
            if lit > 0:
                lit = f'x{lit}'
            else:
                lit = f'~x{-lit}'
            expr_pos.append(f'{w} {lit}')
            expr_neg.append(f'{-w} {lit}')
            wsum += w

        xs_pos = ' '.join(expr_pos)
        xs_neg = ' '.join(expr_neg)

        self._constr.append(f'{xs_pos} {bound} ~x{y} >= {bound}')
        self._constr.append(f'{xs_neg} {wsum-bound+1} x{y} >= {1-bound}')
        return yvar

    def make_at_least(self, props, bound, name):
        y = self._do_make_at_least(props, bound, name)
        if CHECK:
            self._checker.add_geq([(l.var, w) for l, w in props], bound, y.var)
        return y

    def make_at_most(self, props, bound: int, name: str):
        y = self._do_make_at_least([(l, -w) for l, w in props], -bound, name)
        if CHECK:
            self._checker.add_leq([(l.var, w) for l, w in props], bound, y.var)
        return y

    def make_empty_expr(self):
        return PySatBoolRef(expr=[])

    def write_formula(self, fout):
        print(f'* #variable= {self._varcnt} #constraint= {len(self._constr)}',
              file=fout)
        for i in self._constr:
            fout.write(i)
            fout.write('\n')

    def set_timeout(self, timeout: float):
        self._timeout = timeout
        self._timeout_safe = timeout + 1
