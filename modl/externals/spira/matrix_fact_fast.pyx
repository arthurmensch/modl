# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np


from libc.math cimport fabs


cdef double _cd(np.ndarray[double, ndim=1] X_data,
                np.ndarray[int, ndim=1] X_indices,
                np.ndarray[int, ndim=1] X_indptr,
                double* P,
                double* Q,
                np.ndarray[double, ndim=1] residuals,
                np.ndarray[double, ndim=1] g,  # first derivatives
                np.ndarray[double, ndim=1] h,  # second derivatives
                np.ndarray[double, ndim=1] delta,
                int n_rows,
                int n_cols,
                int n_components,
                double alpha,
                int update_Q):

    cdef int u, kk, k, ii, i, n, m
    cdef double reg, p_uk, q_ki, old
    cdef double* ptr
    cdef double violation = 0
    cdef int n_nz
    cdef double* data
    cdef int* indices


    m = n_cols if update_Q else n_rows

    for k in xrange(n_components):

        # Reset first and second derivatives
        for i in xrange(m):
            g[i] = 0
            h[i] = 0

        # Compute first and second derivatives
        for u in xrange(n_rows):
            p_uk = (P + u * n_components + k)[0]

            n_nz = X_indptr[u+1] - X_indptr[u]
            data = <double*>X_data.data + X_indptr[u]
            indices = <int*>X_indices.data + X_indptr[u]

            for ii in xrange(n_nz):
                i = indices[ii];
                n = X_indptr[u] + ii

                q_ki = (Q + i * n_components + k)[0]

                if update_Q:
                    reg = alpha * q_ki;
                    g[i] += residuals[n] * p_uk + reg
                    h[i] += p_uk * p_uk + alpha
                else:
                    reg = alpha * p_uk
                    g[u] += residuals[n] * q_ki + reg
                    h[u] += q_ki * q_ki + alpha


        # Update coefficients
        for i in xrange(m):
            if h[i] == 0:
                continue

            if update_Q:
                ptr = Q + i * n_components + k
            else:
                ptr = P + i * n_components + k

            old = ptr[0]
            ptr[0] -= g[i] / h[i]
            delta[i] = ptr[0] - old
            violation += fabs(g[i])

        # Update residuals
        for u in xrange(n_rows):

            n_nz = X_indptr[u+1] - X_indptr[u]
            data = <double*>X_data.data + X_indptr[u]
            indices = <int*>X_indices.data + X_indptr[u]

            p_uk = (P + u * n_components + k)[0]

            for ii in xrange(n_nz):
                i = indices[ii]
                n = X_indptr[u] + ii

                if update_Q:
                    residuals[n] += delta[i] * p_uk
                else:
                    q_ki = (Q + i * n_components + k)[0]
                    residuals[n] += delta[u] * q_ki

    # End loop over components

    return violation


def _cd_fit(self,
            np.ndarray[double, ndim=1] X_data,
            np.ndarray[int, ndim=1] X_indices,
            np.ndarray[int, ndim=1] X_indptr,
            np.ndarray[double, ndim=2, mode='c'] P,
            np.ndarray[double, ndim=2, mode='fortran'] Q,
            np.ndarray[double, ndim=1] residuals,
            np.ndarray[double, ndim=1] g,
            np.ndarray[double, ndim=1] h,
            np.ndarray[double, ndim=1] delta,
            int n_components,
            double alpha,
            int max_iter,
            double tol,
            callback,
            int verbose):

    cdef int n, it
    cdef double violation_init = 0
    cdef double violation
    cdef int has_callback = callback is not None

    cdef int n_rows = P.shape[0]
    cdef int n_cols = Q.shape[1]

    cdef double* P_ptr = <double*>P.data
    cdef double* Q_ptr = <double*>Q.data


    # Initialize ratings
    for n in xrange(X_data.shape[0]):
        residuals[n] = -X_data[n]

    # Estimate P and Q
    for it in xrange(max_iter):
        violation = 0;

        violation += _cd(X_data, X_indices, X_indptr, P_ptr, Q_ptr, residuals,
                         g, h, delta, n_rows, n_cols, n_components, alpha, 0)

        violation += _cd(X_data, X_indices, X_indptr, P_ptr, Q_ptr,
                         residuals, g, h, delta, n_rows, n_cols,
                         n_components, alpha, 1)

        if has_callback:
            callback(self)

        if it == 0:
            violation_init = violation;

        if verbose:
            print it + 1, violation / violation_init

        if violation / violation_init < tol:
            if verbose:
                print "Converged"
            break


def _predict(np.ndarray[double, ndim=1] X_data,
             np.ndarray[int, ndim=1] X_indices,
             np.ndarray[int, ndim=1] X_indptr,
             np.ndarray[double, ndim=2, mode='c'] P,
             np.ndarray[double, ndim=2, mode='fortran'] Q):

    cdef int n_rows = P.shape[0]
    cdef int n_components = P.shape[1]

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i, k
    cdef double dot

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            i = indices[ii]

            dot = 0
            for k in xrange(n_components):
                dot += P[u, k] * Q[k, i]

            data[ii] = dot
