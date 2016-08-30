
cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    unsigned long rk_interval(unsigned long max, rk_state *state)

cdef class RandomState:
    cdef rk_state *internal_state
    cdef object initial_seed
    cdef long randint(self, unsigned long high) nogil

cdef class RandomStateMemoryView(RandomState):
    cdef void shuffle(self, long[:] x) nogil