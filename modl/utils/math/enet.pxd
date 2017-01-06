# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# noinspection PyUnresolvedReferences
cimport numpy as np
from cython cimport floating

cpdef floating enet_norm(floating[:] v, floating l1_ratio) nogil

cpdef void enet_projection(floating[:] v, floating[:] out, floating radius,
                             floating l1_ratio) nogil

cpdef void enet_scale(floating[:] X,
                              floating l1_ratio, floating radius=*) nogil
