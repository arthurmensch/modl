# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
from libc.math cimport sqrt, log, exp, ceil, pow, fabs

from modl.enet_proj_fast cimport enet_projection_inplace, enet_norm
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

# cdef class FastDictMFStats:
#     cdef double[:] loss
#     cdef double loss_indep
#     cdef long subset_stop
#     cdef long subset_start
#     cdef long[:] subset_array
#     cdef double[::1, :] T
#     cdef double[::1, :] G
#     cdef long[:] counter
#     cdef double[::1, :] B
#     cdef double[::1, :] A
#     cdef long n_iter
#     cdef long n_verbose_call
#
#     def __init__(self, A, B, counter, G, T, loss,
#                  loss_indep, subset_array, subset_start,
#                  subset_stop,
#                  n_iter,
#                  n_verbose_call):
#         self.loss = loss
#         self.loss_indep = loss_indep
#         self.subset_stop = subset_stop
#         self.subset_start = subset_start
#         self.subset_array = subset_array
#         self.T = T
#         self.G = G
#         self.counter = counter
#         self.B = B
#         self.A = A
#         self.n_iter = n_iter
#         self.n_verbose_call = n_verbose_call
#
#
#
# cdef FastDictMFStats _stat_from_slow_stat(stat_object):
#     cdef FastDictMFStats stat = FastDictMFStats(
#     stat_object.A
#     ,stat_object.B
#     ,stat_object.counter
#     ,stat_object.T
#     ,stat_object.G
#     ,stat_object.loss
#     ,stat_object.loss_indep
#     ,stat_object.subset_array
#     ,stat_object.subset_start
#     ,stat_object.subset_stop
#     ,stat_object.n_iter
#     ,stat_object.n_verbose_call
#     )
#     return stat

# cpdef int _update_code_sparse_full_fast(double[:] X_data, int[:] X_indices,
#                   int[:] X_indptr, long n_rows, long n_cols,
#                   long[:] row_range,
#                   double alpha, double[::1, :] P, double[::1, :] Q,
#                   double[:] Q_mult,
#                   double[::1, :] Q_idx,
#                   double[::1, :] G_temp,
#                   ):
#     cdef int len_row_range = row_range.shape[0]
#     cdef int n_components = P.shape[0]
#     cdef double* Q_idx_ptr = &Q_idx[0, 0]
#     cdef double* Q_ptr = &Q[0, 0]
#     cdef double* P_ptr = &P[0, 0]
#     cdef double* G_temp_ptr = &G_temp[0, 0]
#     cdef double* X_data_ptr = &X_data[0]
#     cdef int info = 0
#     cdef int ii, jj, i
#     cdef int nnz
#     cdef double reg
#     cdef double this_Q_mult
#     for ii in range(len_row_range):
#         i = row_range[ii]
#         nnz = X_indptr[i + 1] - X_indptr[i]
#         # print('Filling Q')
#         for k in range(n_components):
#             this_Q_mult = exp(Q_mult[k])
#             for jj in range(nnz):
#                 Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * this_Q_mult
#         # print('Computing Gram')
#         dgemm(&NTRANS, &TRANS,
#               &n_components, &n_components, &nnz,
#               &oned,
#               Q_idx_ptr, &n_components,
#               Q_idx_ptr, &n_components,
#               &zerod,
#               G_temp_ptr, &n_components
#               )
#         # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
#         reg = 2 * alpha * nnz / n_cols
#         for p in range(n_components):
#             G_temp[p, p] += reg
#         # print('Computing Q**T x')
#         # Qx = Q_idx.dot(x)
#         dgemv(&NTRANS,
#               &n_components, &nnz,
#               &oned,
#               Q_idx_ptr, &n_components,
#               X_data_ptr + X_indptr[i], &one,
#               &zerod,
#               P_ptr + i * n_components, &one
#               )
#         # P[j] = linalg.solve(C, Qx, sym_pos=True,
#         #                     overwrite_a=True, check_finite=False)
#         # print('Solving linear system')
#         dposv(&UP, &n_components, &one, G_temp_ptr, &n_components,
#               P_ptr + i * n_components, &n_components,
#               &info)
#         if info != 0:
#             return -1
#     return 0

# cdef get_w(double[:] w, long[:] idx, long[:] counter, long batch_size,
#            long learning_rate):
#     cdef int idx_len = idx.shape[0]
#     cdef int count = counter[0]
#     cdef int i, jj, j
#     w[0] = 1
#     for i in range(count + 1, count + 1 + batch_size):
#         w[0] *= (1 - pow(i, - learning_rate))
#     w[0] = 1 - w[0]
#
#     for jj in range(idx_len):
#         j = idx[jj]
#         count = counter[j + 1]
#         w[jj + 1] = 1
#         for i in range(count + 1, count + 1 + batch_size):
#             w[jj + 1] *= (1 - pow(i, - learning_rate))
#         w[jj + 1] = 1 - w[jj + 1]

cdef _update_code(X, subset, alpha, learning_rate,
                  offset,
                  Q, A, B, counter,
                  impute,
                  Q_subset,
                  P_batch,
                  G_temp,
                  debug):
    cdef int len_batch = X.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef int n_components = Q.shape[0]
    cdef double* Q_subset_ptr = &Q_subset[0, 0]
    cdef double* P_batch_ptr = &P_batch[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* G_temp_ptr = &G_temp[0, 0]
    cdef double* X_ptr = &X[0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double reg, v
    cdef int last = 0
    cdef double w_B, w_A

    for jj in range(len_subset):
        Q_subset[k, jj] = Q[k, subset[jj]]
    if impute:
        reg = alpha
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
                G_temp[p, n] = G[p, n]
    else:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &zerod,
              G_temp_ptr, &n_components
              )

        # print('Computing Q**T x')
        # Qx = Q_idx.dot(x)
        dgemm(&NTRANS, &TRANS,
              &n_components, &len_subset, &len_batch
              &oned,
              Q_subset_ptr, &n_components,
              X_ptr, &one,
              &zerod,
              P_batch_ptr + ii * n_components, &one
              )

    # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
    for p in range(n_components):
        G_temp[p, p] += alpha

    # print('Solving linear system')
    # P[j] = linalg.solve(C, Qx, sym_pos=True,
    #                     overwrite_a=True, check_finite=False)
    dposv(&UP, &n_components, &one, G_temp_ptr, &n_components,
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
        j = subset[jj]
        idx_mask[j] = 1
        counter[j + 1] += 1
        w_B = pow((1. + offset) /(offset + counter[j + 1]), learning_rate)
        for k in range(n_components):
            B[k, j] = (1 - w_B) * B[k, j] + \
                             w_B * P_batch[k, ii]\
                             * X[X_indptr[i] + jj]
    P[i, :] = P_batch[:, ii]


cdef int _update_code_sparse_fast(double[:] X_data, int[:] X_indices,
                  int[:] X_indptr, long n_rows, long n_cols,
                  double[:, ::1] P, double[::1, :] Q,
                  double[:] Q_mult,
                  double alpha, double learning_rate,
                  double offset,
                  double[::1, :] A, double[::1, :] B,
                  double[::1, :] G, double[::1, :] T,
                  long[:] counter,
                  long[:] row_batch,
                  double[::1, :] G_temp,
                  double[::1, :] Q_idx,
                  double[::1, :] P_batch,
                  double[:] sub_Qx,
                  char[:] idx_mask,
                  long[:] idx_concat,
                  bint impute,
                  ):
    cdef int len_batch = row_batch.shape[0],
    cdef int n_components = Q.shape[0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* P_batch_ptr = &P_batch[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* G_temp_ptr = &G_temp[0, 0]
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
            Q_exp_mult = exp(Q_mult[k])
            for jj in range(nnz):
                # Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * exp(Q_mult[k])
                Q_idx[k, jj] = Q[k, X_indices[X_indptr[i] + jj]] * Q_exp_mult

        # print('Computing Gram')

        if impute:
            reg = alpha
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
                    G_temp[p, n] = G[p, n]
        else:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &nnz,
                  &oned,
                  Q_idx_ptr, &n_components,
                  Q_idx_ptr, &n_components,
                  &zerod,
                  G_temp_ptr, &n_components
                  )
            reg = alpha * nnz / n_cols

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
            G_temp[p, p] += reg

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        # print('Solving linear system')
        dposv(&UP, &n_components, &one, G_temp_ptr, &n_components,
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
        P[i, :] = P_batch[:, ii]

    for ii in range(n_cols):
        if idx_mask[ii]:
            idx_concat[last] = ii
            last += 1

    return last



# Tested
cpdef _update_dict(double[::1, :] Q,
                  int[:] subset,
                  bint freeze_first_col,
                  double l1_ratio,
                  double[::1, :] A, double[::1, :] B,
                  double[::1, :] G,
                  bint impute,
                  double[::1, :] R,
                  double[::1, :] Q_subset,
                  double[::1, :] old_sub_G,
                  double[:] norm,
                  double[:] buffer,
                  long[:] components_range):
    cdef int n_components = Q.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef unsigned int components_range_len = components_range.shape[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* Q_subset_ptr = &Q_subset[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* old_sub_G_ptr = &old_sub_G[0, 0]
    cdef double old_norm = 0
    cdef unsigned int k, kk, j, jj

    for k in range(n_components):
        for jj in range(len_subset):
            j = subset[jj]
            R[k, jj] = B[k, j]
            Q_subset[k, jj] = Q[k, j]

    for kk in range(components_range_len):
        k = components_range[kk]
        norm[k] = enet_norm(Q_subset[k,:len_subset], l1_ratio)

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &zerod,
              old_sub_G_ptr, &n_components
              )

    # R = B - AQ
    dgemm(&NTRANS, &NTRANS,
          &n_components, &len_subset, &n_components,
          &moned,
          A_ptr, &n_components,
          Q_subset_ptr, &n_components,
          &oned,
          R_ptr, &n_components)

    for kk in range(components_range_len):
        k = components_range[kk]
        dger(&n_components, &len_subset, &oned,
             A_ptr + k * n_components,
             &one, Q_subset_ptr + k, &n_components, R_ptr, &n_components)
        for jj in range(len_subset):
            Q_subset[k, jj] = R[k, jj] / A[k, k]
        enet_projection_inplace(Q_subset[k, :len_subset], buffer[:len_subset], norm[k], l1_ratio)
        for jj in range(len_subset):
            Q_subset[k, jj] = buffer[jj]
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &len_subset, &moned,
             A_ptr + k  * n_components,
             &one, Q_subset_ptr + k, &n_components, R_ptr, &n_components)

    for kk in range(n_components):
        for jj in range(len_subset):
            j = subset[jj]
            Q[kk, j] = Q_subset[kk, jj]

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )
        for k in range(n_components):
            for j in range(n_components):
                G[j, k] -= old_sub_G[j, k]








# cdef void _update_dict_fast(double[::1, :] Q,
#                             double[:] Q_mult,
#                             double[:] Q_norm,
#                             double[::1, :] A, double[::1, :] B,
#                             double[::1, :] G,
#                             double[::1, :] R,
#                             double[::1, :] Q_idx,
#                             double[::1, :] old_sub_G,
#                             long[:] idx,
#                             bint fit_intercept, long[:] components_range,
#                             bint impute,
#                             bint partial):
#
#     cdef int n_components = Q.shape[0]
#     cdef int idx_len = idx.shape[0]
#     cdef unsigned int components_range_len = components_range.shape[0]
#     cdef double* Q_ptr = &Q[0, 0]
#     cdef double* Q_idx_ptr = &Q_idx[0, 0]
#     cdef double* A_ptr = &A[0, 0]
#     cdef double* R_ptr = &R[0, 0]
#     cdef double* G_ptr = &G[0, 0]
#     cdef double* old_sub_G_ptr = &old_sub_G[0, 0]
#     cdef double this_Q_mult, old_norm = 0
#     cdef unsigned int k, kk, j, jj
#
#     for kk in range(n_components):
#         this_Q_mult = exp(Q_mult[kk])
#         for jj in range(idx_len):
#             j = idx[jj]
#             R[kk, jj] = B[kk, j]
#             Q_idx[kk, jj] = Q[kk, j] * this_Q_mult
#
#     if impute:
#         dgemm(&NTRANS, &TRANS,
#               &n_components, &n_components, &idx_len,
#               &oned,
#               Q_idx_ptr, &n_components,
#               Q_idx_ptr, &n_components,
#               &zerod,
#               old_sub_G_ptr, &n_components
#               )
#
#     # R = B - AQ
#     dgemm(&NTRANS, &NTRANS,
#           &n_components, &idx_len, &n_components,
#           &moned,
#           A_ptr, &n_components,
#           Q_idx_ptr, &n_components,
#           &oned,
#           R_ptr, &n_components)
#
#     for kk in range(components_range_len):
#         k = components_range[kk]
#         if partial:
#             old_norm = 0
#             Q_norm[k] = 0
#             for jj in range(idx_len):
#                 old_norm += Q_idx[k, jj] ** 2
#             if old_norm == 0:
#                 continue
#         else:
#             for jj in range(idx_len):
#                 Q_norm[k] -= Q_idx[k, jj] ** 2
#         dger(&n_components, &idx_len, &oned,
#              A_ptr + k * n_components,
#              &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)
#         for jj in range(idx_len):
#             Q_idx[k, jj] = R[k, jj] / A[k, k]
#             Q_norm[k] += Q_idx[k, jj] ** 2
#         if partial:
#             Q_norm[k] /= old_norm
#         if Q_norm[k] > 1:
#             # Live update of Q_idx
#             for jj in range(idx_len):
#                 Q_idx[k, jj] /= sqrt(Q_norm[k])
#             if not partial:
#                 Q_mult[k] -= .5 * log(Q_norm[k])
#                 Q_norm[k] = 1
#         # R -= A[:, k] Q[:, k].T
#         dger(&n_components, &idx_len, &moned,
#              A_ptr + k  * n_components,
#              &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)
#
#     for kk in range(n_components):
#         this_Q_mult = exp(Q_mult[kk])
#         for jj in range(idx_len):
#             j = idx[jj]
#             Q[kk, j] = Q_idx[kk, jj] / this_Q_mult
#
#     if impute:
#         dgemm(&NTRANS, &TRANS,
#               &n_components, &n_components, &idx_len,
#               &oned,
#               Q_idx_ptr, &n_components,
#               Q_idx_ptr, &n_components,
#               &oned,
#               G_ptr, &n_components
#               )
#         for k in range(n_components):
#             for j in range(n_components):
#                 G[j, k] -= old_sub_G[j, k]


# def _online_dl_main_loop_sparse_fast(X, double[::1, :] Q, double[:, ::1] P,
#                                      stat_slow,
#                                      double alpha, double learning_rate,
#                                      double offset,
#                                      bint freeze_first_col,
#                                      long batch_size,
#                                      long max_n_iter,
#                                      bint impute,
#                                      bint partial,
#                                      long verbose,
#                                      UINT32_t random_seed,
#                                      callback):
#
#     cdef double[:] X_data = X.data
#     cdef int[:] X_indices = X.indices
#     cdef int[:] X_indptr = X.indptr
#     cdef int n_rows = X.shape[0]
#     cdef int n_cols = X.shape[1]
#
#     cdef long[:] row_range = X.getnnz(axis=1).nonzero()[0]
#     cdef int len_row_range = row_range.shape[0]
#
#     cdef FastDictMFStats stat = _stat_from_slow_stat(stat_slow)
#
#     cdef int max_idx_size = min(n_cols, X.getnnz(axis=1).max() * batch_size)
#
#     cdef int n_batches = int(ceil(len_row_range / batch_size))
#     cdef int n_components = Q.shape[0]
#
#     cdef UINT32_t seed = random_seed
#
#     cdef double[::1, :] Q_idx = np.zeros((n_components, max_idx_size),
#                                          order='F')
#     cdef double[::1, :] R = np.zeros((n_components, max_idx_size),
#                                      order='F')
#     cdef double[::1, :] P_batch = np.zeros((n_components, batch_size),
#                                            order='F')
#     cdef double[::1, :] G_temp = np.zeros((n_components, n_components),
#                                           order='F')
#     cdef double[:] sub_Qx = np.zeros(n_components)
#     cdef double[::1, :] old_sub_G = np.zeros((n_components, n_components),
#                                              order='F')
#     cdef char[:] idx_mask = np.zeros(n_cols, dtype='i1')
#     cdef long[:] idx_concat = np.zeros(max_idx_size, dtype='int')
#     cdef long[:] components_range
#     cdef int i, start, stop, last, last_call = 0
#     cdef long[:] row_batch
#     cdef double[:] Q_norm = np.ones(n_components)
#     cdef double[:] Q_mult = np.zeros(n_components)
#     cdef double norm = 0
#
#     cdef double new_rmse, old_rmse
#
#     cdef double this_Q_mult
#     cdef bint update
#
#     for k in range(n_components):
#         for j in range(n_cols):
#             norm += Q[k, j] ** 2
#         norm = sqrt(norm)
#         for j in range(n_cols):
#            Q[k, j] /= norm
#            Q_mult[k] = 0
#
#     if not freeze_first_col:
#         components_range = np.arange(n_components)
#     else:
#         components_range = np.arange(1, n_components)
#
#     n_batches = int(ceil(len_row_range / batch_size))
#     _shuffle(row_range, &seed)
#     for i in range(n_batches):
#         start = i * batch_size
#         stop = start + batch_size
#         if stop > len_row_range:
#             stop = len_row_range
#         row_batch = row_range[start:stop]
#         if stat.n_iter + len(row_batch) - 1 >= max_n_iter:
#             return
#         last = _update_code_sparse_fast(X_data, X_indices,
#                                  X_indptr, n_rows, n_cols,
#                                  P, Q,
#                                  Q_mult,
#                                  alpha, learning_rate,
#                                  offset,
#                                  stat.A, stat.B, stat.G, stat.T,
#                                  stat.counter,
#                                  row_batch,
#                                  G_temp,
#                                  Q_idx,
#                                  P_batch,
#                                  sub_Qx,
#                                  idx_mask,
#                                  idx_concat,
#                                  impute)
#         _shuffle(components_range, &seed)
#         _update_dict_fast(
#                 Q,
#                 Q_mult,
#                 Q_norm,
#                 stat.A,
#                 stat.B,
#                 stat.G,
#                 R,
#                 Q_idx,
#                 old_sub_G,
#                 idx_concat[:last],
#                 freeze_first_col,
#                 components_range,
#                 impute,
#                 partial)
#         stat.n_iter += row_batch.shape[0]
#         # Numerical stability
#         if not partial:
#             update = False
#             lim = -50
#             for k in range(n_components):
#                 if stat.Q_mult[k] < lim:
#                     update = True
#             if update:
#                 for k in range(n_components):
#                     this_Q_mult = exp(stat.Q_mult[k])
#                     for j in range(n_cols):
#                             Q[k, j] *= this_Q_mult
#                     Q_mult[k] = 0
#         if verbose and stat.n_iter // ceil(
#                 int(n_rows / verbose)) == stat.n_verbose_call + 1:
#             print("Iteration %i" % stat.n_iter)
#             stat.n_verbose_call += 1
#             if callback is not None:
#                 callback()
#         stat_slow.n_iter = stat.n_iter
#         stat_slow.n_verbose_call = stat.n_verbose_call