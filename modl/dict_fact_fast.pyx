# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport pow
cimport numpy as np

from scipy.linalg.cython_lapack cimport dposv
from scipy.linalg.cython_blas cimport dgemm, dger
from ._utils.enet_proj_fast cimport enet_projection_inplace, enet_norm

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int zero = 0
cdef int one = 1
cdef double zerod = 0
cdef double oned = 1
cdef double moned = -1

ctypedef np.uint32_t UINT32_t

cpdef void _get_weights(double[:] w, long[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset):
    cdef int len_subset = subset.shape[0]
    cdef int full_count = counter[0]
    cdef int count
    cdef int i, jj, j
    w[0] = 1
    for i in range(full_count + 1, full_count + 1 + batch_size):
        w[0] *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w[0] = 1 - w[0]
    for jj in range(len_subset):
        j = subset[jj]
        count = counter[j + 1]
        w[jj + 1] = 1
        for i in range(1, 1 + batch_size):
            w[jj + 1] *= (1 - (full_count + i) / (count + i) * pow(
                (1 + offset) / (offset + full_count + i), learning_rate))
        w[jj + 1] = 1 - w[jj + 1]

cpdef double _get_simple_weights(long[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset):
    cdef int len_subset = subset.shape[0]
    cdef int full_count = counter[0]
    cdef int count
    cdef int i, jj, j
    cdef double w = 1
    for i in range(full_count + 1, full_count + 1 + batch_size):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w


# cpdef long _update_code_sparse_batch(double[:] X_data,
#                                      int[:] X_indices,
#                                      int[:] X_indptr,
#                                      int n_rows,
#                                      int n_cols,
#                                      long[:] row_batch,
#                                      long[:] sample_subset,
#                                      double alpha,
#                                      double learning_rate,
#                                      double offset,
#                                      double[::1, :] Q,
#                                      double[:, ::1] P,
#                                      double[::1, :] A,
#                                      double[::1, :] B,
#                                      long[:] counter,
#                                      double[::1, :] E,
#                                      double[:] reg,
#                                      double[:] weights,
#                                      double[::1, :] G,
#                                      double[::1, :] beta,
#                                      double[:] impute_mult,  # [E_norm, multiplier]
#                                      bint impute,
#                                      bint exact_E,
#                                      bint persist_P,
#                                      double[::1, :] Q_subset,
#                                      double[::1, :] P_temp,
#                                      double[::1, :] G_temp,
#                                      double[::1, :] X_temp,
#                                      char[:] subset_mask,
#                                      int[:] dict_subset,
#                                      int[:] dict_subset_lim,
#                                      ) except *:
#     """
#     Parameters
#     ----------
#     X_data: masked data matrix (csr)
#     X_indices: masked data matrix (csr)
#     X_indptr: masked data matrix (csr)
#     n_rows: masked data matrix (csr)
#     n_cols: masked data matrix (csr)
#     alpha: regularization parameter
#     learning_rate: decrease rate in the learning sequence (in [.5, 1])
#     offset: offset in the learning sequence
#     Q: Dictionary
#     A: Algorithm variable
#     B: Algorithm variable
#     counter: Algorithm variable
#     G: Algorithm variable
#     beta: Algorithm variable
#     impute: Online update of Gram matrix
#     Q_subset : Temporary array. Holds the subdictionary
#     P_temp: Temporary array. Holds the codes for the mini batch
#     G_temp: Temporary array. Holds the Gram matrix.
#     subset_mask: Holds the binary mask for visited features
#     dict_subset: for union of seen features
#     dict_subset_lim: for union of seen features (holds the number of seen
#      features)
#     weights_temp: Temporary array. Holds the update weights
#     n_iter: Updated by the function, for early stopping
#     max_n_iter: Iteration budget
#     update_P: keeps an updated version of the code
#     """
#     cdef int len_batch = row_batch.shape[0]
#     cdef int n_components = Q.shape[0]
#     cdef int ii, i, j, jj, k, idx_j
#     cdef int l = 0
#     cdef int[:] subset
#     subset_mask[:] = 0
#     for ii in range(len_batch):
#         i = row_batch[ii]
#         subset = X_indices[X_indptr[i]:X_indptr[i + 1]]
#         len_subset = subset.shape[0]
#         for jj in range(len_subset):
#             idx_j = X_indptr[i] + jj
#             X_temp[0, jj] = X_data[idx_j]
#             j = subset[jj]
#             if not subset_mask[j]:
#                 subset_mask[j] = 1
#                 dict_subset[l] = j
#                 l += 1
#
#         _update_code(X_temp[:1],
#                      subset,
#                      sample_subset[i:i + 1],
#                      alpha,
#                      learning_rate,
#                      offset,
#                      Q, P,
#                      A, B,
#                      counter,
#                      E,
#                      reg,
#                      weights,
#                      G,
#                      beta,
#                      impute_mult,
#                      impute,
#                      1.,
#                      exact_E,
#                      persist_P,
#                      Q_subset,
#                      P_temp,
#                      G_temp,
#                      subset_mask,
#                      )
#     dict_subset_lim[0] = l

cpdef void _update_code(double[::1, :] X,
                        long[:] subset,
                        long[:] sample_subset,
                        double alpha,
                        double learning_rate,
                        double offset,
                        long var_red,
                        long projection,
                        double reduction,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[::1, :] beta_,
                        double[:] multiplier_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] D_subset,
                        double[::1, :] this_X,
                        double[::1, :] this_code,
                        double[::1, :] this_G,
                        double[:] w_arr) except *:
    """
    Compute code for a mini-batch and update algorithm statistics accordingly

    Parameters
    ----------
    X: masked data matrix
    this_subset: indices (loci) of masked data
    alpha: regularization parameter
    learning_rate: decrease rate in the learning sequence (in [.5, 1])
    offset: offset in the learning sequence
    D_: Dictionary
    A_: algorithm variable
    B_: algorithm variable
    counter_: algorithm variable
    G: algorithm variable
    T: algorithm variable
    impute: Online update of Gram matrix
    D_subset : Temporary array. Holds the subdictionary
    this_code: Temporary array. Holds the codes for the mini batch
    this_G: emporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    weights: Temporary array. Holds the update weights

    """
    cdef int batch_size = X.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef double* X_ptr = &X[0, 0]
    cdef double* D_subset_ptr = &D_subset[0, 0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* B_ptr = &B_[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double* this_code_ptr = &this_code[0, 0]
    cdef double* this_G_ptr = &this_G[0, 0]
    cdef double* this_X_ptr = &this_X[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double v
    cdef int last = 0
    cdef double one_m_w, w, wdbatch, w_norm

    for jj in range(len_subset):
        j = subset[jj]
        for ii in range(batch_size):
            this_X[ii, jj] = X[ii, j]
        for k in range(n_components):
            D_subset[k, jj] = D_[k, j]

    counter_[0] += batch_size

    if var_red == 3: # weight_based
        for jj in range(len_subset):
            j = subset[jj]
            counter_[j + 1] += batch_size
        _get_weights(w_arr, subset, counter_, batch_size,
             learning_rate, offset)

        # P_temp = np.dot(D_subset, this_X.T)
        dgemm(&NTRANS, &TRANS,
              &n_components, &batch_size, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              this_X_ptr, &batch_size,
              &zerod,
              this_code_ptr, &n_components
              )

        # this_G = D_subset.dot(D_subset.T)
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &zerod,
              this_G_ptr, &n_components
              )
        for p in range(n_components):
            this_G[p, p] += alpha / reduction

        dposv(&UP, &n_components, &batch_size, this_G_ptr, &n_components,
              this_code_ptr, &n_components,
              &info)
        if info != 0:
            raise ValueError

        wdbatch = w_arr[0] / batch_size
        one_m_w = 1 - w_arr[0]
        # A_ *= 1 - w_A
        # A_ += this_code.dot(this_code.T) * w_A / batch_size
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &batch_size,
              &wdbatch,
              this_code_ptr, &n_components,
              this_code_ptr, &n_components,
              &one_m_w,
              A_ptr, &n_components
              )

        # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
        # Reuse D_subset as B_subset
        for jj in range(len_subset):
            j = subset[jj]
            for k in range(n_components):
                D_subset[k, jj] = B_[k, j] * (1 - w_arr[jj + 1])
            for ii in range(batch_size):
                this_X[ii, jj] *= w_arr[jj + 1] / batch_size
        dgemm(&NTRANS, &NTRANS,
              &n_components, &len_subset, &batch_size,
              &oned,
              this_code_ptr, &n_components,
              this_X_ptr, &batch_size,
              &oned,
              D_subset_ptr, &n_components)
        for jj in range(len_subset):
            j = subset[jj]
            for k in range(n_components):
                B_[k, j] = D_subset[k, jj]
    else:
        for ii in range(batch_size):
            for jj in range(len_subset):
                this_X[ii, jj] *= reduction

        # P_temp = np.dot(D_subset, this_X.T)
        dgemm(&NTRANS, &TRANS,
              &n_components, &batch_size, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              this_X_ptr, &batch_size,
              &zerod,
              this_code_ptr, &n_components
              )

        w = _get_simple_weights(subset, counter_, batch_size,
                                learning_rate, offset)

        if w != 1:
            multiplier_[0] *= 1 - w

        w_norm = w / multiplier_[0]

        for p in range(n_components):
            for q in range(n_components):
                this_G[p, q] = G_[p, q]
            this_G[p, p] += alpha

        if var_red != 1:
            for ii in range(batch_size):
                i = sample_subset[ii]
                row_counter_[i] += 1
                w = pow(row_counter_[i], -learning_rate)
                for p in range(n_components):
                    beta_[i, p] *= 1 - w
                    beta_[i, p] += this_code[p, ii] * w
                    this_code[p, ii] = beta_[i, p]
        dposv(&UP, &n_components, &batch_size, this_G_ptr, &n_components,
              this_code_ptr, &n_components,
              &info)
        if info != 0:
            raise ValueError

        wdbatch = w_norm / batch_size
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &batch_size,
              &wdbatch,
              this_code_ptr, &n_components,
              this_code_ptr, &n_components,
              &oned,
              A_ptr, &n_components
              )

        if var_red == 4:
            # self.B_ += this_code.dot(X) * w_norm / batch_size
            dgemm(&NTRANS, &NTRANS,
              &n_components, &n_cols, &batch_size,
              &wdbatch,
              this_code_ptr, &n_components,
              X_ptr, &batch_size,
              &oned,
              B_ptr, &n_components
              )
        else:
            # Reuse D_subset as B_subset
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    D_subset[k, jj] = B_[k, j]
            dgemm(&NTRANS, &NTRANS,
                  &n_components, &len_subset, &batch_size,
                  &wdbatch,
                  this_code_ptr, &n_components,
                  this_X_ptr, &batch_size,
                  &oned,
                  D_subset_ptr, &n_components)
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    B_[k, j] = D_subset[k, jj]

    for ii in range(batch_size):
        i = sample_subset[ii]
        for k in range(n_components):
            code_[i, k] = this_code[k, ii]




cpdef void _update_dict(double[::1, :] D_,
                  long[:] dict_subset,
                  bint fit_intercept,
                  double l1_ratio,
                  long projection,
                  long var_red,
                  long[:] D_range,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  double[::1, :] R,
                  double[::1, :] D_subset,
                  double[:] norm,
                  double[:] buffer):
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef int len_subset = dict_subset.shape[0]
    cdef unsigned int components_range_len = D_range.shape[0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* D_subset_ptr = &D_subset[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double old_norm = 0
    cdef unsigned int k, kk, j, jj

    for k in range(n_components):
        for jj in range(len_subset):
            j = dict_subset[jj]
            D_subset[k, jj] = D_[k, j]
            R[k, jj] = B_[k, j]

    for kk in range(components_range_len):
        k = D_range[kk]
        if projection == 1:
            norm[k] = enet_norm(D_[k], l1_ratio)
        else:
            norm[k] = enet_norm(D_subset[k, :len_subset], l1_ratio)
    if projection == 2 and var_red != 3:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &moned,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )

    # R = B - AQ
    dgemm(&NTRANS, &NTRANS,
          &n_components, &len_subset, &n_components,
          &moned,
          A_ptr, &n_components,
          D_subset_ptr, &n_components,
          &oned,
          R_ptr, &n_components)

    for kk in range(components_range_len):
        k = D_range[kk]
        dger(&n_components, &len_subset, &oned,
             A_ptr + k * n_components,
             &one, D_subset_ptr + k, &n_components, R_ptr, &n_components)

        for jj in range(len_subset):
            D_subset[k, jj] = R[k, jj] / A_[k, k]

        if projection == 1:
            for jj in range(len_subset):
                j = dict_subset[jj]
                D_[k, j] = D_subset[k, jj]
            enet_projection_inplace(D_[k], buffer,
                                    norm[k], l1_ratio)
            for jj in range(n_cols):
                D_[k, jj] = buffer[jj]
            for jj in range(len_subset):
                j = dict_subset[jj]
                D_subset[k, jj] = D_[k, j]
        else:
            enet_projection_inplace(D_subset[k, :len_subset],
                                    buffer[:len_subset],
                                    norm[k], l1_ratio)
            for jj in range(len_subset):
                D_subset[k, jj] = buffer[jj]
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &len_subset, &moned,
             A_ptr + k * n_components,
             &one, D_subset_ptr + k, &n_components, R_ptr, &n_components)

    if projection == 2:
        for jj in range(len_subset):
            j = dict_subset[jj]
            for kk in range(n_components):
                D_[kk, j] = D_subset[kk, jj]
        if var_red != 3:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &len_subset,
                  &oned,
                  D_subset_ptr, &n_components,
                  D_subset_ptr, &n_components,
                  &oned,
                  G_ptr, &n_components
                  )
    elif var_red != 3:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &n_cols,
              &oned,
              D_ptr, &n_components,
              D_ptr, &n_components,
              &zerod,
              G_ptr, &n_components
              )


cpdef void _predict(double[:] X_data,
             int[:] X_indices,
             int[:] X_indptr,
             double[:, ::1] P,
             double[::1, :] Q):
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