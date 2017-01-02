# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
#  Copyright 2005 Robert Kern (robert.kern@gmail.com)
cimport cython

from libc cimport stdlib
from cython cimport view

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
    double rk_gauss(rk_state *state) nogil

cdef extern from "distributions.h":
    long rk_binomial(rk_state *state, long n, double p) nogil
    long long rk_geometric(rk_state *state, double p) nogil

cdef unsigned int MAX_INT = np.iinfo(np.uint32).max

cdef class RandomState:
    def __cinit__(self, seed=None):
        self.internal_state = <rk_state*>stdlib.malloc(sizeof(rk_state))
        self.initial_seed = seed
        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            stdlib.free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None):
        cdef rk_error errcode
        cdef unsigned long long_seed
        if seed is None:
            errcode = rk_randomseed(self.internal_state)
        else:
            long_seed = <unsigned long> seed
            rk_seed(seed, self.internal_state)

    cdef int randint(self, unsigned int high=MAX_INT) nogil:
        return <int>rk_interval(high, self.internal_state)

    cdef int binomial(self, int n, double p) nogil:
        return <int>rk_binomial(self.internal_state, n, p)

    cdef int geometric(self, double p) nogil:
        return <int>rk_geometric(self.internal_state, p)

    cdef double randn(self) nogil:
        return rk_gauss(self.internal_state)

    cdef void shuffle(self, int[:] x) nogil:
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

    cdef int[:] permutation(self, int size):
        cdef int i
        cdef int[:] res = view.array((size, ), sizeof(int), format='i')
        for i in range(size):
            res[i] = i
        self.shuffle(res)
        return res

@cython.final
cdef class Sampler(object):
    def __cinit__(self, int range, bint rand_size,
                  bint replacement,
                 unsigned long random_seed):
        """

        Parameters
        ----------
        range
        reduction
        sampling: int in {1, 2, 3}.
            1: Bernouilli sampling
            2: Fixed-size sampling without replacement
            3: Fixed-size sampling
        random_seed

        Returns
        -------

        """
        self.range = range
        self.rand_size = rand_size
        self.replacement = replacement
        self.random_state = RandomState(seed=random_seed)

        self.box = self.random_state.permutation(self.range)
        self.temp = view.array((self.range, ),
                                        sizeof(int),
                                        format='i')
        self.lim_sup = 0
        self.lim_inf = 0

        self.random_state.shuffle(self.box)

    cpdef int[:] yield_subset(self, double reduction) nogil:
        cdef int remainder
        cdef int len_subset
        if self.rand_size:
            len_subset = self.random_state.binomial(self.range,
                                                         1. / reduction)
        else:
            len_subset = int(self.range / reduction)
        if self.replacement:
            self.random_state.shuffle(self.box)
            self.lim_inf = 0
            self.lim_sup = len_subset
        else: # Without replacement
            if self.range != len_subset:
                self.lim_inf = self.lim_sup
                remainder = self.range - self.lim_inf
                if remainder == 0:
                    self.random_state.shuffle(self.box)
                    self.lim_inf = 0
                elif remainder < len_subset:
                    self.temp[:remainder] = self.box[:remainder]
                    self.box[:remainder] = self.box[self.lim_inf:]
                    self.box[self.lim_inf:] = self.temp[:remainder]
                    self.random_state.shuffle(self.box[remainder:])
                    self.lim_inf = 0
                self.lim_sup = self.lim_inf + len_subset
            else:
                self.lim_inf = 0
                self.lim_sup = self.range
        return self.box[self.lim_inf:self.lim_sup]