# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# noinspection PyUnresolvedReferences
cimport numpy as np

ctypedef np.uint32_t UINT32_t

cpdef double enet_norm_fast(double[:] v, double l1_ratio) nogil

cpdef void enet_projection_fast(double[:] v, double[:] b, double radius,
                             double l1_ratio) nogil

cpdef void enet_scale_matrix_fast(double[::1, :] X,
                double l1_ratio, double radius=*) nogil

cpdef void enet_scale_fast(double[:] X,
                              double l1_ratio, double radius=*) nogil
