# -*- coding: utf-8 -*-
# cython: language_level=3

"""utilities for encoding cardinality constraints"""

cdef class _SeqCounterEncoder:
    """sequential counters"""
    cdef object _add_clause
    cdef object _env

    def __init__(self, add_clause, env):
        self._add_clause = add_clause
        self._env = env

    def make_geq(self, list lits, int bound) -> int:
        cdef int varcnt
        cdef int b
        cdef int i

        varcnt = self._env._varcnt
        aux = [[None, lits[0]]]
        for i in range(1, len(lits)):
            cur = [None]
            for b in range(1, min(bound, i + 1) + 1):
                varcnt += 1
                cur.append(varcnt)
                if b == i + 1:
                    cls = [
                        [aux[i - 1][b - 1]],
                        [lits[i]]
                    ]
                elif b > 1:
                    cls = [
                        [aux[i - 1][b], aux[i - 1][b - 1]],
                        [aux[i - 1][b], lits[i]]
                    ]
                else:
                    cls = [
                        [aux[i - 1][b], lits[i]]
                    ]

                self._add_var_defeq(varcnt, cls)
            aux.append(cur)

        self._env._varcnt = varcnt
        ret = aux[len(lits) - 1][bound]
        return ret

    def make_leq(self, list lits, int bound) -> int:
        cdef int varcnt
        cdef int b
        cdef int i

        varcnt = self._env._varcnt
        aux = [[-lits[0]]]
        for i in range(1, len(lits)):
            cur = []
            for b in range(min(i, bound) + 1):
                varcnt += 1
                cur.append(varcnt)
                if b == i:
                    cls = [
                        [aux[i - 1][b - 1], -lits[i]]
                    ]
                elif b >= 1:
                    cls = [
                        [aux[i - 1][b - 1], aux[i - 1][b]],
                        [aux[i - 1][b - 1], -lits[i]]
                    ]
                else:
                    cls = [
                        [aux[i - 1][b]],
                        [-lits[i]]
                    ]

                self._add_var_defeq(varcnt, cls)
            aux.append(cur)

        self._env._varcnt = varcnt
        ret = aux[len(lits) - 1][bound]
        return ret

    cdef _add_var_defeq(self, int var, list clauses):
        """add a clause that assigns ``var`` with the disjuction of ``clauses``"""
        cdef int a0, a1, b0, b1
        cdef list i

        for i in clauses:
            i.append(-var)
            self._add_clause(i)
            i.pop()

        if len(clauses) == 1:
            (a0, a1), = clauses
            neg = [[-a0], [-a1]]
        elif len(clauses) == 2:
            if len(clauses[0]) == 1:
                (a0, ), (b0, ) = clauses
                neg = [[-a0, -b0]]
            else:
                (a0, a1), (b0, b1) = clauses
                a0, a1, b0, b1 = -a0, -a1, -b0, -b1
                neg = [[a0, b0], [a0, b1], [a1, b0], [a1, b1]]
        else:
            raise RuntimeError(
                f'only two clauses supported, got {len(clauses)}')

        for i in neg:
            i.append(var)
            self._add_clause(i)


cdef class _CardNetworkEncoder:
    """cardinality networks"""

    cdef object _add_clause
    cdef object _env

    def __init__(self, add_clause, env):
        self._add_clause = add_clause
        self._env = env

    cdef list _make_vars(self, int cnt):
        n0 = self._env._varcnt
        ret = list(range(n0 + 1, n0 + 1 + cnt))
        self._env._varcnt += cnt
        return ret

    cdef _add_reified_disjuction(self, int x, int a, int b):
        """add ``x = a || b``"""
        cdef object add
        add = self._add_clause
        add([-x, a, b])
        add([-a, x])
        add([-b, x])

    cdef _add_reified_conjuction(self, int x, int a, int b):
        """add ``x = a && b``"""
        cdef object add
        add = self._add_clause
        add([-a, -b, x])
        add([-x, a])
        add([-x, b])

    cdef _half_merge(self, list xs, list ys, list outs):
        cdef int n
        cdef int i
        n = len(xs)
        assert len(outs) == n * 2 and len(ys) == n
        if n == 1:
            a, = xs
            b, = ys
            c1, c2 = outs
            self._add_reified_disjuction(c1, a, b)
            self._add_reified_conjuction(c2, a, b)
            return

        assert n % 2 == 0
        cs = [None, None] + outs[1:-1]
        ds = [None, outs[0]] + self._make_vars(n - 1)
        es = [None] + self._make_vars(n - 1) + outs[-1:]
        self._half_merge(xs[::2], ys[::2], ds[1:])
        self._half_merge(xs[1::2], ys[1::2], es[1:])
        for i in range(1, n):
            self._add_reified_disjuction(cs[i*2], ds[i+1], es[i])
            self._add_reified_conjuction(cs[i*2+1], ds[i+1], es[i])

    cdef _half_sort(self, list xs, list cs):
        cdef int n
        cdef int i
        n = len(xs) // 2
        assert len(xs) == len(cs) == n * 2
        if n == 1:
            self._half_merge(xs[:1], xs[1:], cs)
            return

        d0 = self._make_vars(n)
        d1 = self._make_vars(n)
        self._half_sort(xs[:n], d0)
        self._half_sort(xs[n:], d1)
        self._half_merge(d0, d1, cs)

    cdef _simp_merge(self, list xs, list ys, list outs):
        cdef int n
        cdef int i
        n = len(xs)
        assert len(outs) == n + 1 and len(ys) == n
        if n == 1:
            self._half_merge(xs, ys, outs)
            return
        assert n % 2 == 0
        cs = [None, None] + outs[1:]
        ds = [None, outs[0]] + self._make_vars(n // 2)
        es = [None] + self._make_vars(n // 2 + 1)
        self._simp_merge(xs[::2], ys[::2], ds[1:])
        self._simp_merge(xs[1::2], ys[1::2], es[1:])
        for i in range(1, n // 2 + 1):
            self._add_reified_disjuction(cs[i*2], ds[i+1], es[i])
            self._add_reified_conjuction(cs[i*2+1], ds[i+1], es[i])

    cdef list _card_net_pow2(self, list xs, int k):
        assert k & (k - 1) == 0 and len(xs) >= k
        if len(xs) == 1:
            assert k == 1
            return xs

        cs = self._make_vars(k)
        if len(xs) == k:
            self._half_sort(xs, cs)
            return cs

        d0 = self._card_net_pow2(xs[:k], k)
        d1 = self._card_net_pow2(xs[k:], k)
        cs.extend(self._make_vars(1))
        self._simp_merge(d0, d1, cs)
        cs.pop()
        return cs

    cdef list _card_net(self, list lits, int bound):
        cdef int k
        cdef int i
        cdef int u
        k = 1
        while k <= bound:
            k <<= 1
        u = ((len(lits) - 1) // k + 1) * k
        pad = self._make_vars(u - len(lits))
        for i in pad:
            self._add_clause([-i])

        return self._card_net_pow2(lits + pad, k)

    def make_geq(self, list lits, int bound) -> int:
        assert bound > 0
        cs = self._card_net(lits, bound)
        return cs[bound - 1]

    def make_leq(self, list lits, int bound) -> int:
        cs = self._card_net(lits, bound)
        return -cs[bound]


ENCODER_MAP = {'seq': _SeqCounterEncoder, 'cn': _CardNetworkEncoder,
               None: _SeqCounterEncoder}

cdef class CardEncoder:
    """cardinality constraint encoder"""

    cdef object _add_clause
    cdef object _add_leq_assign
    cdef object _add_geq_assign
    cdef object _env
    cdef object _encoder

    def __init__(self, env, encoder_factory=None):
        self._env = env
        solver = env._solver.solver
        if env._all_clauses is None:
            self._add_clause = solver.add_clause
        else:
            # need clause recording
            def add_clause(c, *, _add=solver.add_clause,
                           _rec=env._add_clause_record):
                _add(c)
                _rec(c.copy())
            self._add_clause = add_clause

        if hasattr(solver, 'add_leq_assign'):
            if env._all_clauses is None:
                self._add_leq_assign = solver.add_leq_assign
                self._add_geq_assign = solver.add_geq_assign
            else:
                def add_leq_assign(lits, bound, dst, *,
                                   add=solver.add_leq_assign,
                                   rec=env._add_clause_record):
                    add(lits, bound, dst)
                    rec(lits + ['<=', bound, '#', dst])
                def add_geq_assign(lits, bound, dst, *,
                                   add=solver.add_geq_assign,
                                   rec=env._add_clause_record):
                    add(lits, bound, dst)
                    rec(lits + ['>=', bound, '#', dst])
                self._add_leq_assign = add_leq_assign
                self._add_geq_assign = add_geq_assign
        else:
            self._add_leq_assign = None
            self._add_geq_assign = None

        self._encoder = ENCODER_MAP[encoder_factory](
            self._add_clause, self._env)

    cdef _make_true(self, name):
        ret = self._env.new_var(name).var
        self._add_clause([ret])
        return ret

    cdef _make_false(self, name):
        ret = self._env.new_var(name).var
        self._add_clause([-ret])
        return ret

    def make_geq(self, list lits, int bound, name) -> int:
        cdef int i

        if self._add_geq_assign is not None:
            ret = self._env.new_var(name).var
            self._add_geq_assign(lits, bound, ret)
            return ret

        if bound <= 0:
            return self._make_true(name)
        if bound > len(lits):
            return self._make_false(name)

        if bound > len(lits) // 2:
            return self.make_leq([-i for i in lits], len(lits) - bound, name)

        return self._encoder.make_geq(lits, bound)

    def make_leq(self, list lits, int bound, name) -> int:
        cdef int i

        if self._add_leq_assign is not None:
            ret = self._env.new_var(name).var
            self._add_leq_assign(lits, bound, ret)
            return ret

        if bound < 0:
            return self._make_false(name)
        if bound >= len(lits):
            return self._make_true(name)

        if bound > len(lits) // 2:
            return self.make_geq([-i for i in lits], len(lits) - bound, name)

        return self._encoder.make_leq(lits, bound)
