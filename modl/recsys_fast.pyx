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

def _subset_weights(int[:] subset, long n_iter, long[:] feature_n_iter,
                        long batch_size, double learning_rate, double offset):
    cdef int len_subset = subset.shape[0]
    cdef int feature_count
    cdef int i, jj, j
    w = view.array((len_subset,), sizeof(double), format='d')
    w[:] = 1
    for jj in range(len_subset):
        j = subset[jj]
        feature_count = feature_n_iter[j]
        w[jj] = 1
        for i in range(1, 1 + batch_size):
            w[jj] *= (1 - (n_iter + i) / (feature_count + i) * pow(
                (1 + offset) / (offset + n_iter + i), learning_rate))
        w[jj] = 1 - w[jj]
    return np.asarray(w)