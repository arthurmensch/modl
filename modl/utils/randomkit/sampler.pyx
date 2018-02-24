# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport view
import numpy as np

cdef class Sampler(object):
    def __init__(self, range, rand_size,
                  replacement,
                  random_seed):
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
        self.range = <long> range
        self.rand_size = <bint> rand_size
        self.replacement = <bint> replacement
        self.random_state = RandomState(seed=<unsigned long> random_seed)

        self.box = self.random_state.permutation(self.range)
        self.temp = view.array((self.range, ), sizeof(long), format='l')
        self.lim_sup = 0
        self.lim_inf = 0

        self.random_state.shuffle(self.box)

    cpdef long[:] yield_subset(self, double reduction):
        cdef long remainder
        cdef long len_subset
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
        return np.array(self.box[self.lim_inf:self.lim_sup])