# -*- coding: utf-8 -*-
# cython: language_level=3

from libcpp.vector cimport vector as cpp_vector
from libcpp cimport bool

import threading

cdef extern from "minisatcs_wrapper.h":
    cdef cppclass WrappedMinisatSolver:
        void new_clause_prepare() except+
        void new_clause_add_lit(int lit) except+
        void new_clause_commit() except+
        void new_clause_commit_leq(int bound, int dst) except+
        void new_clause_commit_geq(int bound, int dst) except+
        void set_var_preference(int x, int p) except+
        void set_var_name(int x, const char*) except+
        int solve_with_signal(bool setup, double timeout) nogil except+
        bool previous_timeout() except+
        cpp_vector[int] get_model() except+
        void set_recorder(MinisatClauseRecorder*) except+

        int verbosity
        int phase_saving
        bool rnd_pol

    cdef cppclass MinisatClauseRecorder:
        int nr_var()
        void replay(WrappedMinisatSolver&) nogil except+


cdef class ClauseRecorder:
    """usage:

        recorder = ClauseRecorder()
        with recorder(solver):
            ...
        recorder.replay(new_solver)

    """
    cdef MinisatClauseRecorder _recorder

    def replay(self, solver):
        cdef Solver cs = solver
        self._recorder.replay(cs._solver)

    def __call__(self, solver):
        return ScopedClauseRecorder(self, solver)

    @property
    def nr_var(self):
        return self._recorder.nr_var()


cdef class Solver:
    cdef WrappedMinisatSolver _solver
    cdef int _nr_clause

    def __init__(self, verbosity=0):
        self._nr_clause = 0
        self._solver.verbosity = verbosity
        self._solver.phase_saving = 0
        self._solver.rnd_pol = True

    cdef _prepare_clause(self, list clause):
        cdef int i
        self._solver.new_clause_prepare()
        for i in clause:
            self._solver.new_clause_add_lit(i)
        self._nr_clause += 1

    def add_clause(self, clause):
        self._prepare_clause(clause)
        self._solver.new_clause_commit()

    def add_leq_assign(self, clause, bound, dst):
        self._prepare_clause(clause)
        self._solver.new_clause_commit_leq(bound, dst)

    def add_geq_assign(self, clause, bound, dst):
        self._prepare_clause(clause)
        self._solver.new_clause_commit_geq(bound, dst)

    def set_var_preference(self, var, pref):
        """set the branching preference of a var to break ties for activity;
        default preference is zero, and lower value means higher preference

        :param var: a literal (can be negative)
        """
        self._solver.set_var_preference(var, pref)

    def set_var_name(self, var, str name):
        """set the name of a var, for debug purposes

        :param var: the literal to represent a var (can be in negation form)
        """
        self._solver.set_var_name(var, name.encode('utf-8'))

    def set_verbosity(self, level):
        self._solver.verbosity = level

    def solve(self, timeout):
        is_main = threading.current_thread() == threading.main_thread()
        if timeout is None:
            timeout = -1
        ret = self._solver.solve_with_signal(is_main, timeout)
        return [False, True, None][ret]

    def get_model(self):
        ret = self._solver.get_model()
        assert len(ret), 'model unavailable'
        return ret

    @property
    def nr_clause(self):
        return self._nr_clause


cdef class ScopedClauseRecorder:
    cdef bool _entered
    cdef MinisatClauseRecorder* _recorder
    cdef WrappedMinisatSolver* _solver

    def __init__(self, ClauseRecorder recorder, Solver solver):
        self._entered = False
        self._recorder = &recorder._recorder
        self._solver = &solver._solver

    def __enter__(self):
        assert not self._entered
        self._entered = True
        self._solver.set_recorder(self._recorder)

    def __exit__(self, *args, **kwargs):
        assert self._entered
        self._entered = False
        self._solver.set_recorder(NULL)
