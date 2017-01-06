# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport view

import numpy as np

cpdef void _predict(double[:] X_data,
             int[:] X_indices,
             int[:] X_indptr,
             double[:, :] P,
             double[:, :] Q):
    """Adapted from spira"""
    cdef int n_rows = P.shape[0]
    cdef int n_components = P.shape[1]

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i, k
    cdef double dot

    for u in range(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*> &X_data[0] + X_indptr[u]
        indices = <int*> &X_indices[0] + X_indptr[u]

        for ii in range(n_nz):
            i = indices[ii]

            dot = 0
            for k in range(n_components):
                dot += P[u, k] * Q[k, i]

            data[ii] = dot