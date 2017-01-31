from .random_fast cimport RandomState

cdef class Sampler(object):
    cdef public long range
    cdef public bint rand_size
    cdef public bint replacement

    cdef public long[:] box
    cdef public long[:] temp
    cdef public long lim_sup
    cdef public long lim_inf

    cdef public RandomState random_state

    cpdef long[:] yield_subset(self, double reduction)