# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# noinspection PyUnresolvedReferences
cimport numpy as np
from cython cimport floating
ctypedef np.uint32_t UINT32_t

cpdef floating enet_norm_fast(floating[:] v, floating l1_ratio) nogil

cpdef void enet_projection_fast(floating[:] v, floating[:] b, floating radius,
                             floating l1_ratio) nogil

cpdef void enet_scale_matrix_fast(floating[:, :] X,
                floating l1_ratio, floating radius=*) nogil

cpdef void enet_scale_fast(floating[:] X,
                              floating l1_ratio, floating radius=*) nogil
