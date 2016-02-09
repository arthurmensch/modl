# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
from libc.math cimport sqrt, log, exp, ceil, pow, fabs

from scipy.linalg.cython_blas cimport dger, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dposv

import numpy as np

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int zero = 0
cdef int one = 1
cdef double zerod = 0
cdef double oned = 1
cdef double moned = -1

ctypedef np.uint32_t UINT32_t

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef void _shuffle(long[:] arr, UINT32_t* random_state):
    cdef int len_arr = arr.shape[0]
    cdef int i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_int(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]


cpdef int _update_code_full_fast(double[:] X_data, int[:] X_indices,
                  int[:] X_indptr, long n_rows, long n_cols,
                  long[:] row_range,
                  double alpha, double[::1, :] P, double[::1, :] Q,
                  double[:] Q_mult,
                  double[::1, :] Q_idx,
                  double[::1, :] C,
                  bint exp_mult):
    cdef int len_row_range = row_range.shape[0]
    cdef int n_components = P.shape[0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* G_ptr = &C[0, 0]
    cdef double* X_data_ptr = &X_data[0]
    cdef int info = 0
    cdef int ii, jj, i
    cdef int nnz
    cdef double reg
    cdef this_Q_mult
    for ii in range(len_row_range):
        i = row_range[ii]
        nnz = X_indptr[i + 1] - X_indptr[i]
        # print('Filling Q')

        for k in range(n_components):
            this_Q_mult = exp(Q_mult[k]) if exp_mult else Q_mult[k]
            for jj in range(nnz):
                Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * this_Q_mult
        # print('Computing Gram')

        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &nnz,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &zerod,
              G_ptr, &n_components
              )
        # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        reg = 2 * alpha * nnz / n_cols
        for p in range(n_components):
            C[p, p] += reg

        # print('Computing Q**T x')
        # Qx = Q_idx.dot(x)
        dgemv(&NTRANS,
              &n_components, &nnz,
              &oned,
              Q_idx_ptr, &n_components,
              X_data_ptr + X_indptr[i], &one,
              &zerod,
              P_ptr + i * n_components, &one
              )

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        # print('Solving linear system')
        dposv(&UP, &n_components, &one, G_ptr, &n_components,
              P_ptr + i * n_components, &n_components,
              &info)
        if info != 0:
            return -1
    return 0

cdef get_w(double[:] w, long[:] idx, long[:] counter, long batch_size,
           long learning_rate):
    cdef int idx_len = idx.shape[0]
    cdef int count = counter[0]
    cdef int i, jj, j
    w[0] = 1
    for i in range(count + 1, count + 1 + batch_size):
        w[0] *= (1 - pow(i, - learning_rate))
    w[0] = 1 - w[0]

    for jj in range(idx_len):
        j = idx[jj]
        count = counter[j + 1]
        w[jj + 1] = 1
        for i in range(count + 1, count + 1 + batch_size):
            w[jj + 1] *= (1 - pow(i, - learning_rate))
        w[jj + 1] = 1 - w[jj + 1]


cdef int _update_code_fast(double[:] X_data, int[:] X_indices,
                  int[:] X_indptr, long n_rows, long n_cols,
                  double alpha, double learning_rate,
                  double offset,
                  double[::1, :] A, double[::1, :] B,
                  double[::1, :] G, double[::1, :] T,
                  long[:] counter,
                  double[::1, :] P, double[::1, :] Q,
                  double[:] Q_mult,
                  long[:] row_batch,
                  double[::1, :] C,
                  double[::1, :] Q_idx,
                  double[::1, :] P_batch,
                  double[:] sub_Qx,
                  char[:] idx_mask,
                  long[:] idx_concat,
                  bint impute,
                  bint exp_mult):
    cdef int len_batch = row_batch.shape[0],
    cdef int n_components = P.shape[0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* P_batch_ptr = &P_batch[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* C_ptr = &C[0, 0]
    cdef double* X_data_ptr = &X_data[0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double reg, v
    cdef int last = 0
    cdef double w_B, w_A
    cdef double Q_exp_mult

    # get_w(double[:] w, idx, long[:] counter, long batch_size,
    #            long learning_rate)

    for jj in range(n_cols):
        idx_mask[jj] = 0


    for ii in range(len_batch):
        i = row_batch[ii]
        nnz = X_indptr[i + 1] - X_indptr[i]
        # print('Filling Q')
        for k in range(n_components):
            Q_exp_mult = exp(Q_mult[k]) if exp_mult else Q_mult[k]
            for jj in range(nnz):
                # Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * exp(Q_mult[k])
                Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * Q_exp_mult

        # print('Computing Gram')

        if impute:
            reg = 2 * alpha
            v = 1 # nnz / n_cols
            for p in range(n_components):
                sub_Qx[p] = 0
                for jj in range(nnz):
                    j = X_indices[X_indptr[i] + jj]
                    T[p, 0] -= T[p, j + 1]
                    T[p, j + 1] = Q_idx[p, jj] * X_data[X_indptr[i] + jj]
                    sub_Qx[p] += T[p, j + 1]
                T[p, 0] += sub_Qx[p]
                P_batch[p, ii] = (1 - v) * sub_Qx[p] + v * T[p, 0]
            for p in range(n_components):
                for n in range(n_components):
                    C[p, n] = G[p, n]
        else:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &nnz,
                  &oned,
                  Q_idx_ptr, &n_components,
                  Q_idx_ptr, &n_components,
                  &zerod,
                  C_ptr, &n_components
                  )
            reg = 2 * alpha * nnz / n_cols

            # print('Computing Q**T x')
            # Qx = Q_idx.dot(x)
            dgemv(&NTRANS,
                  &n_components, &nnz,
                  &oned,
                  Q_idx_ptr, &n_components,
                  X_data_ptr + X_indptr[i], &one,
                  &zerod,
                  P_batch_ptr + ii * n_components, &one
                  )

        # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        for p in range(n_components):
            C[p, p] += reg

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        # print('Solving linear system')
        dposv(&UP, &n_components, &one, C_ptr, &n_components,
              P_batch_ptr + ii * n_components, &n_components,
              &info)
        if info != 0:
            raise ValueError

        # A *= 1 - w_A * len_batch
        # A += P[row_batch].T.dot(P[row_batch]) * w_A
        counter[0] += 1
        w_A = pow((1. + offset) /(offset + counter[0]), learning_rate)
        for k in range(n_components):
            for m in range(n_components):
                A[k, m] *= 1 - w_A
        dger(&n_components, &n_components,
             &w_A,
             P_batch_ptr + ii * n_components, &one,
             P_batch_ptr + ii * n_components, &one,
             A_ptr, &n_components)

        # w_B = np.power(counter[1][idx], - learning_rate)[np.newaxis, :]
        # B[:, idx] *= 1 - w_B
        # B[:, idx] += np.outer(P[j], x) * w_B
        # Use a loop to avoid copying a contiguous version of B
        for jj in range(nnz):
            j = X_indices[X_indptr[i] + jj]
            idx_mask[j] = 1
            counter[j + 1] += 1
            w_B = pow((1. + offset) /(offset + counter[j + 1]), learning_rate)
            for k in range(n_components):
                B[k, j] = (1 - w_B) * B[k, j] + \
                                 w_B * P_batch[k, ii]\
                                 * X_data[X_indptr[i] + jj]
        # dger(&n_components, &nnz,
        #      &w_B,
        #      P_batch_ptr + ii * n_components, &one,
        #      X_data + X_indptr[i], &one,
        #      &mu_B,
        #      &B_batch)

        P[:, i] = P_batch[:, ii]

    for ii in range(n_cols):
        if idx_mask[ii]:
            idx_concat[last] = ii
            last += 1

    return last


cdef void _update_dict_fast(double[::1, :] A, double[::1, :] B,
                            double[::1, :] G,
                            double[::1, :] Q,
                            double[:] Q_mult,
                            double[:] Q_norm,
                            double[::1, :] R,
                            double[::1, :] Q_idx,
                            double[::1, :] old_sub_G,
                            long[:] idx,
                            bint fit_intercept, long[:] components_range,
                            bint impute,
                            bint partial,
                            bint exp_mult):

    cdef int n_components = Q.shape[0]
    cdef int idx_len = idx.shape[0]
    cdef unsigned int components_range_len = components_range.shape[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* old_sub_G_ptr = &old_sub_G[0, 0]
    cdef double this_Q_mult, old_norm = 0
    cdef unsigned int k, kk, j, jj

    # print("Q mult: % .4f" % Q_mult[1])
    # print("Q norm: % .4f" % Q_norm[1])

    for kk in range(n_components):
        if exp_mult:
            this_Q_mult = exp(Q_mult[kk])
        else:
            this_Q_mult = Q_mult[kk]
        for jj in range(idx_len):
            j = idx[jj]
            R[kk, jj] = B[kk, j]
            Q_idx[kk, jj] = Q[kk, j] * this_Q_mult

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &idx_len,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &zerod,
              old_sub_G_ptr, &n_components
              )

    # R = B - AQ
    dgemm(&NTRANS, &NTRANS,
          &n_components, &idx_len, &n_components,
          &moned,
          A_ptr, &n_components,
          Q_idx_ptr, &n_components,
          &oned,
          R_ptr, &n_components)

    for kk in range(components_range_len):
        k = components_range[kk]
        if partial:
            old_norm = 0
            Q_norm[k] = 0
            for jj in range(idx_len):
                old_norm += Q_idx[k, jj] ** 2
            if old_norm == 0:
                continue
        else:
            for jj in range(idx_len):
                Q_norm[k] -= Q_idx[k, jj] ** 2
        dger(&n_components, &idx_len, &oned,
             A_ptr + k * n_components,
             &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)
        for jj in range(idx_len):
            Q_idx[k, jj] = R[k, jj] / A[k, k]
            Q_norm[k] += Q_idx[k, jj] ** 2
        if partial:
            Q_norm[k] /= old_norm
        if Q_norm[k] > 1:
            # Live update of Q_idx
            for jj in range(idx_len):
                Q_idx[k, jj] /= sqrt(Q_norm[k])
            if not partial:
                if exp_mult:
                    Q_mult[k] -= .5 * log(Q_norm[k])
                else:
                    Q_mult[k] /= sqrt(Q_norm[k])
                Q_norm[k] = 1
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &idx_len, &moned,
             A_ptr + k  * n_components,
             &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)

    for kk in range(n_components):
        this_Q_mult = exp(Q_mult[kk]) if exp_mult else Q_mult[kk]
        for jj in range(idx_len):
            j = idx[jj]
            Q[kk, j] = Q_idx[kk, jj] / this_Q_mult

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &idx_len,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )
        for k in range(n_components):
            for j in range(n_components):
                G[j, k] -= old_sub_G[j, k]


def _online_dl_fast(double[:] X_data, int[:] X_indices,
                    int[:] X_indptr, long n_rows, long n_cols,
                    long[:] row_range,
                    long max_idx_size,
                    double alpha, double learning_rate,
                    double offset,
                    double[::1, :] A, double[::1, :] B,
                    long[:] counter,
                    double[::1, :] G, double[::1, :] T,
                    double[::1, :] P, double[::1, :] Q,
                    double[:] Q_mult,
                    long n_epochs, long batch_size,
                    UINT32_t random_seed,
                    long verbose,
                    bint fit_intercept,
                    bint partial,
                    bint impute,
                    bint exp_mult,
                    callback):

    cdef int len_row_range = row_range.shape[0]
    cdef int n_batches = int(ceil(len_row_range / batch_size))
    cdef int n_cols_int = n_cols
    cdef int n_components = P.shape[0]
    cdef UINT32_t seed = random_seed
    cdef double[::1, :] Q_idx = np.zeros((n_components, max_idx_size),
                                         order='F')
    cdef double[::1, :] R = np.zeros((n_components, max_idx_size),
                                     order='F')
    cdef double[::1, :] P_batch = np.zeros((n_components, batch_size),
                                           order='F')
    cdef double[::1, :] C = np.zeros((n_components, n_components), order='F')
    cdef double[:] sub_Qx = np.zeros(n_components)
    cdef double[::1, :] old_sub_G = np.zeros((n_components, n_components),
                                             order='F')
    cdef char[:] idx_mask = np.zeros(n_cols, dtype='i1')
    cdef long[:] idx_concat = np.zeros(max_idx_size, dtype='int')
    cdef long[:] components_range
    cdef int i, start, stop, last, last_call = 0
    cdef long[:] row_batch
    cdef double[:] Q_norm = np.ones(n_components)
    cdef double norm = 0

    cdef double* Q_ptr = &Q[0, 0]
    cdef double* G_ptr = &G[0, 0]

    cdef double new_rmse, old_rmse

    cdef double min

    for k in range(n_components):
        for j in range(n_cols):
            norm += Q[k, j] ** 2
        norm = sqrt(norm)
        for j in range(n_cols):
           Q[k, j] /= norm
        if exp_mult:
            Q_mult[k] = 0
        else:
            Q_mult[k] = 1

    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)
    old_rmse = 5
    for epoch in range(n_epochs):
        n_batches = int(ceil(len_row_range / batch_size))
        _shuffle(row_range, &seed)
        for i in range(n_batches):
            start = i * batch_size
            stop = start + batch_size
            if stop > len_row_range:
                stop = len_row_range
            row_batch = row_range[start:stop]
            last = _update_code_fast(X_data, X_indices,
                                     X_indptr, n_rows, n_cols,
                                     alpha, learning_rate,
                                     offset,
                                     A, B, G, T,
                                     counter,
                                     P, Q,
                                     Q_mult,
                                     row_batch,
                                     C,
                                     Q_idx,
                                     P_batch,
                                     sub_Qx,
                                     idx_mask,
                                     idx_concat,
                                     impute,
                                     exp_mult)
            _shuffle(components_range, &seed)
            _update_dict_fast(
                    A,
                    B,
                    G,
                    Q,
                    Q_mult,
                    Q_norm,
                    R,
                    Q_idx,
                    old_sub_G,
                    idx_concat[:last],
                    fit_intercept,
                    components_range,
                    impute,
                    partial,
                    exp_mult)
            # Numerical stability

            if not partial:
                min = 0 if exp_mult else 1
                for k in range(n_components):
                    if Q_mult[k] < min:
                        min = Q_mult[k]
                lim = -50 if exp_mult else 1e-10
                if min <= 1e-3:
                    for k in range(n_components):
                        this_Q_mult = exp(Q_mult[k]) if exp_mult else Q_mult[k]
                        for j in range(n_cols):
                                Q[k, j] *= this_Q_mult
                        Q_mult[k] = 0 if exp_mult else 1

            if verbose and counter[0] // int(ceil(
                        len_row_range / verbose)) == last_call + 1:
                print("Iteration %i" % (counter[0]))
                last_call += 1
                callback()
        # if new_rmse > 0:
        #     if fabs(new_rmse - old_rmse) / old_rmse < 0.01:
        #         print('Reducing batch size')
        #         batch_size //= 2
        #         old_rmse = new_rmse