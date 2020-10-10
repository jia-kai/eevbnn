# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
from numpy cimport PyArray_DATA
cimport numpy as np

cdef extern from "_atomic.h":
    np.int32_t atomic_fetch_add_i32(np.int32_t*, np.int32_t)
    np.int32_t atomic_load_i32(const np.int32_t*)

cdef class AtomicInt32:
    cdef object _buf_refkeep
    cdef np.int32_t* _addr

    def __init__(self, np.ndarray[np.int32_t, ndim=1] buf):
        self._buf_refkeep = buf
        self._addr = <np.int32_t*> PyArray_DATA(buf)

    def fetch_add(self, np.int32_t delta):
        return int(atomic_fetch_add_i32(self._addr, delta))

    def load(self):
        return int(atomic_load_i32(self._addr))
