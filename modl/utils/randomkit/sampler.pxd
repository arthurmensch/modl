from .random_fast cimport RandomState

cdef class Sampler(object):
    cdef long range
    cdef bint rand_size
    cdef bint replacement

    cdef long[:] box
    cdef long[:] temp
    cdef long lim_sup
    cdef long lim_inf

    cdef RandomState random_state

    cpdef long[:] yield_subset(self, double reduction)