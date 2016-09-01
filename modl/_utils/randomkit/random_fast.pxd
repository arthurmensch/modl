cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

cdef class RandomState:
    cdef rk_state *internal_state
    cdef object initial_seed
    cdef long randint(self, unsigned long high) nogil

cdef class OurRandomState(RandomState):
    cdef void shuffle(self, long[:] x) nogil

cdef class Sampler(object):
    cdef long n_features
    cdef long len_subset
    cdef long subset_sampling

    cdef long[:] feature_range
    cdef long[:] temp_subset
    cdef long lim_sup
    cdef long lim_inf

    cdef OurRandomState random_state

    cpdef long[:] yield_subset(self) nogil