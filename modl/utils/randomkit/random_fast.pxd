
cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

cdef class RandomState:

    cdef rk_state *internal_state
    cdef public object initial_seed
    cpdef long randint(self, unsigned long high)
    cpdef binomial(self, int n, double p)
    cpdef long[:] permutation(self, long size)