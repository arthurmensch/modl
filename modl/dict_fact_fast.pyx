# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from scipy.linalg.cython_lapack cimport dposv
from scipy.linalg.cython_blas cimport dgemm, dger

from ._utils.enet_proj_fast cimport enet_projection_inplace, enet_norm

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


cdef _get_weights(double[:] w, int[:] subset, long[:] counter, long batch_size,
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


cpdef long _update_code_sparse_batch(double[:] X_data,
                                     int[:] X_indices,
                                     int[:] X_indptr,
                                     int n_rows,
                                     int n_cols,
                                     long[:] row_batch,
                                     double alpha,
                                     double learning_rate,
                                     double offset,
                                     double[::1, :] Q,
                                     double[:, ::1] P,
                                     double[::1, :] A,
                                     double[::1, :] B,
                                     long[:] counter,
                                     double[::1, :] G,
                                     double[::1, :] T,
                                     bint impute,
                                     double[::1, :] Q_subset,
                                     double[::1, :] P_temp,
                                     double[::1, :] G_temp,
                                     double[::1, :] this_X,
                                     char[:] subset_mask,
                                     int[:] dict_subset,
                                     int[:] dict_subset_lim,
                                     double[:] weights,
                                     long n_iter,
                                     long max_n_iter,
                                     bint update_P,
                                     ):
    """
    Parameters
    ----------
    X_data: masked data matrix (csr)
    X_indices: masked data matrix (csr)
    X_indptr: masked data matrix (csr)
    n_rows: masked data matrix (csr)
    n_cols: masked data matrix (csr)
    alpha: regularization parameter
    learning_rate: decrease rate in the learning sequence (in [.5, 1])
    offset: offset in the learning sequence
    Q: Dictionary
    A: Algorithm variable
    B: Algorithm variable
    counter: Algorithm variable
    G: Algorithm variable
    T: Algorithm variable
    impute: Online update of Gram matrix
    Q_subset : Temporary array. Holds the subdictionary
    P_temp: Temporary array. Holds the codes for the mini batch
    G_temp: Temporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    dict_subset: for union of seen features
    dict_subset_lim: for union of seen features (holds the number of seen
     features)
    weights: Temporary array. Holds the update weights
    n_iter: Updated by the function, for early stopping
    max_n_iter: Iteration budget
    update_P: keeps an updated version of the code
    """
    cdef int len_batch = row_batch.shape[0]
    cdef int n_components = Q.shape[0]
    cdef int ii, i, j, jj, k, idx_j
    cdef int l = 0
    cdef double reg
    cdef int[:] subset
    subset_mask[:] = 0
    for ii in range(len_batch):
        i = row_batch[ii]
        if n_iter >= max_n_iter > 0:
            break
        subset = X_indices[X_indptr[i]:X_indptr[i + 1]]
        len_subset = subset.shape[0]
        reg = alpha * subset.shape[0] / n_cols
        for jj in range(len_subset):
            idx_j = X_indptr[i] + jj
            this_X[0, jj] = X_data[idx_j]
            j = subset[jj]
            if not subset_mask[j]:
                subset_mask[j] = 1
                dict_subset[l] = j
                l += 1

        _update_code(this_X, subset, reg, learning_rate,
                     offset, Q, A, B,
                     counter,
                     G,
                     T,
                     impute,
                     Q_subset,
                     P_temp,
                     G_temp,
                     subset_mask,
                     weights,
                     )
        if update_P:
            for k in range(n_components):
                P[i, k] = P_temp[k, 0]
        n_iter += 1
    dict_subset_lim[0] = l
    return n_iter

cpdef _update_code(double[::1, :] X, int[:] subset,
                   double alpha,
                   double learning_rate,
                   double offset,
                   double[::1, :] Q,
                   double[::1, :] A,
                   double[::1, :] B,
                   long[:] counter,
                   double[::1, :] G,
                   double[::1, :] T,
                   bint impute,
                   double[::1, :] Q_subset,
                   double[::1, :] P_temp,
                   double[::1, :] G_temp,
                   char[:] subset_mask,
                   double[:] weights):
    """
    Compute code for a mini-batch and update algorithm statistics accordingly

    Parameters
    ----------
    X: masked data matrix
    subset: indices (loci) of masked data
    alpha: regularization parameter
    learning_rate: decrease rate in the learning sequence (in [.5, 1])
    offset: offset in the learning sequence
    Q: Dictionary
    A: algorithm variable
    B: algorithm variable
    counter: algorithm variable
    G: algorithm variable
    T: algorithm variable
    impute: Online update of Gram matrix
    Q_subset : Temporary array. Holds the subdictionary
    P_temp: Temporary array. Holds the codes for the mini batch
    G_temp: emporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    weights: Temporary array. Holds the update weights

    """
    cdef int batch_size = X.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef int n_components = Q.shape[0]
    cdef double* Q_subset_ptr = &Q_subset[0, 0]
    cdef double* P_temp_ptr = &P_temp[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* G_temp_ptr = &G_temp[0, 0]
    cdef double* X_ptr = &X[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double reg, v
    cdef int last = 0
    cdef double one_m_w_A, w_A

    for k in range(n_components):
        for jj in range(len_subset):
            Q_subset[k, jj] = Q[k, subset[jj]]

    _get_weights(weights, subset, counter, batch_size,
               learning_rate, offset)

    counter[0] += batch_size

    for jj in range(len_subset):
        j = subset[jj]
        counter[j + 1] += batch_size

    # G = Q.T.dot(Q)
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &len_subset,
          &oned,
          Q_subset_ptr, &n_components,
          Q_subset_ptr, &n_components,
          &zerod,
          G_temp_ptr, &n_components
          )
    # Qx = Q_subset.dot(x)
    dgemm(&NTRANS, &TRANS,
          &n_components, &batch_size, &len_subset,
          &oned,
          Q_subset_ptr, &n_components,
          X_ptr, &batch_size,
          &zerod,
          P_temp_ptr, &n_components
          )

    # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
    for p in range(n_components):
        G_temp[p, p] += alpha
    # P[j] = linalg.solve(G_temp, Qx, sym_pos=True,
    #                     overwrite_a=True, check_finite=False)
    dposv(&UP, &n_components, &batch_size, G_temp_ptr, &n_components,
          P_temp_ptr, &n_components,
          &info)
    if info != 0:
        raise ValueError

    # A *= 1 - w_A * len_batch
    # A += P[row_batch].T.dot(P[row_batch]) * w_A
    one_m_w_A = 1 - weights[0]
    w_A = weights[0] / batch_size
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &batch_size,
          &w_A,
          P_temp_ptr, &n_components,
          P_temp_ptr, &n_components,
          &one_m_w_A,
          A_ptr, &n_components
          )
    # B[:, idx] *= 1 - w_B
    for jj in range(len_subset):
        j = subset[jj]
        subset_mask[j] = 1
        # Reuse Q_subset as B_subset
        for k in range(n_components):
            Q_subset[k, jj] = B[k, j] * (1 - weights[jj + 1])
        for ii in range(batch_size):
            X[ii, jj] *= weights[jj + 1] / batch_size
    dgemm(&NTRANS, &NTRANS,
          &n_components, &len_subset, &batch_size,
          &oned,
          P_temp_ptr, &n_components,
          X_ptr, &batch_size,
          &oned,
          Q_subset_ptr, &n_components)
    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            B[k, j] = Q_subset[k, jj]


cpdef _update_dict(double[::1, :] Q,
                  int[:] subset,
                  bint freeze_first_col,
                  double l1_ratio,
                  bint full_projection,
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
    cdef int n_features = Q.shape[1]
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
        if full_projection:
            norm[k] = enet_norm(Q[k], l1_ratio)
        else:
            norm[k] = enet_norm(Q_subset[k, :len_subset], l1_ratio)
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
        if full_projection:
            for jj in range(len_subset):
                j = subset[jj]
                Q[k, j] = Q_subset[k, jj]
            enet_projection_inplace(Q[k], buffer,
                                    norm[k], l1_ratio)
            for jj in range(n_features):
                Q[k, jj] = buffer[jj]
            for jj in range(len_subset):
                j = subset[jj]
                Q_subset[k, jj] = Q[k, j]
        else:
            enet_projection_inplace(Q_subset[k, :len_subset],
                                    buffer[:len_subset],
                                    norm[k], l1_ratio)
            for jj in range(len_subset):
                Q_subset[k, jj] = buffer[jj]
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &len_subset, &moned,
             A_ptr + k  * n_components,
             &one, Q_subset_ptr + k, &n_components, R_ptr, &n_components)

    if not full_projection:
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


def _predict(double[:] X_data,
             int[:] X_indices,
             int[:] X_indptr,
             double[:, ::1] P,
             double[::1, :] Q):
    """Adapted from spira"""
    # FIXME we could use BLAS here
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