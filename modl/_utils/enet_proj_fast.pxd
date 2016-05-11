# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np

ctypedef np.uint32_t UINT32_t

cpdef double enet_norm(double[:] v, double l1_ratio) nogil

cpdef void enet_projection_inplace(double[:] v, double[:] b, double radius,
                             double l1_ratio) nogil