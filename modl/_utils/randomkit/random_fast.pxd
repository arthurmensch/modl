cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

cdef class RandomState:
    cdef rk_state *internal_state
    cdef unsigned long initial_seed
    cdef int randint(self, unsigned int high=*) nogil
    cdef void shuffle(self, int[:] x) nogil
    cdef int[:] permutation(self, int size)
    cdef int binomial(self, int n, double p) nogil
    cdef int geometric(self, double p) nogil
    cdef double randn(self) nogil

cdef class Sampler(object):
    cdef int range
    cdef bint rand_size
    cdef bint replacement

    cdef int[:] box
    cdef int[:] temp
    cdef int lim_sup
    cdef int lim_inf

    cdef RandomState random_state

    cpdef int[:] yield_subset(self, double reduction) nogil