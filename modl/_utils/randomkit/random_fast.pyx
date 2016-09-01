# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
#  Copyright 2005 Robert Kern (robert.kern@gmail.com)
cimport cython

from libc cimport stdlib

import numpy as np
cimport numpy as np

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]

    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state)
    unsigned long rk_ulong(rk_state *state)
    unsigned long rk_interval(unsigned long max, rk_state *state) nogil
    double rk_double(rk_state *state)
    void rk_fill(void *buffer, size_t size, rk_state *state)
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state)
    double rk_gauss(rk_state *state)

cdef class RandomState:

    def __init__(self, seed=None):
        self.internal_state = <rk_state*>stdlib.malloc(sizeof(rk_state))
        self.initial_seed = seed
        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            stdlib.free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None):
        cdef rk_error errcode
        if seed is None:
            errcode = rk_randomseed(self.internal_state)
        elif type(seed) is int:
            rk_seed(seed, self.internal_state)
        elif isinstance(seed, np.integer):
            iseed = int(seed)
            rk_seed(iseed, self.internal_state)
        else:
            raise ValueError("Wrong seed")

    cdef long randint(self, unsigned long high) nogil:
        return <long>rk_interval(high, self.internal_state)

    def shuffle(self, object x):
        cdef int i, j
        cdef int copy

        i = len(x) - 1
        try:
            j = len(x[0])
        except:
            j = 0

        if (j == 0):
            # adaptation of random.shuffle()
            while i > 0:
                j = rk_interval(i, self.internal_state)
                x[i], x[j] = x[j], x[i]
                i = i - 1
        else:
            # make copies
            copy = hasattr(x[0], 'copy')
            if copy:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j].copy(), x[i].copy()
                    i = i - 1
            else:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j][:], x[i][:]
                    i = i - 1

    def __reduce__(self):
        return (RandomState, (self.initial_seed, ))


cdef class OurRandomState(RandomState):
    cdef void shuffle(self, long[:] x) nogil:
        cdef int i, j
        cdef int copy

        cdef long x_temp

        i = x.shape[0] - 1
        while i > 0:
            j = rk_interval(i, self.internal_state)
            x_temp = x[i]
            x[i] = x[j]
            x[j] = x_temp
            i = i - 1
        return

@cython.final
cdef class Sampler(object):
    def __init__(self, long n_features, long len_subset, long subset_sampling,
                 unsigned long random_seed):
        self.n_features = n_features
        self.len_subset = len_subset
        self.subset_sampling = subset_sampling
        self.random_state = OurRandomState(seed=random_seed)

        self.feature_range = np.arange(n_features, dtype='long')
        self.temp_subset = np.zeros(len_subset, dtype='long')
        self.lim_sup = 0
        self.lim_inf = 0

        self.random_state.shuffle(self.feature_range)

    cpdef long[:] yield_subset(self) nogil:
        cdef long remainder
        if self.subset_sampling == 1:
            self.random_state.shuffle(self.feature_range)
            self.lim_inf = 0
            self.lim_sup = self.len_subset
        else:
            if self.n_features != self.len_subset:
                self.lim_inf = self.lim_sup
                remainder = self.n_features - self.lim_inf
                if remainder == 0:
                    self.random_state.shuffle(self.feature_range)
                    self.lim_inf = 0
                elif remainder < self.len_subset:
                    self.temp_subset[:remainder] = self.feature_range[:remainder]
                    self.feature_range[:remainder] = self.feature_range[self.lim_inf:]
                    self.feature_range[self.lim_inf:] = self.temp_subset[:remainder]
                    self.random_state.shuffle(self.feature_range[remainder:])
                    self.lim_inf = 0
                self.lim_sup = self.lim_inf + self.len_subset
            else:
                self.lim_inf = 0
                self.lim_sup = self.n_features
        return self.feature_range[self.lim_inf:self.lim_sup]