# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport pow

from scipy.linalg.cython_lapack cimport dposv
from scipy.linalg.cython_blas cimport dgemm, dger
from ._utils.enet_proj_fast cimport enet_projection_inplace, enet_norm

cimport numpy as np

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int zero = 0
cdef int one = 1
cdef double zerod = 0
cdef double oned = 1
cdef double moned = -1

ctypedef np.uint32_t UINT32_t

cpdef long _update_code_sparse_batch(double[:] X_data,
                                     int[:] X_indices,
                                     int[:] X_indptr,
                                     int n_rows,
                                     int n_cols,
                                     long[:] row_batch,
                                     long[:] sample_subset,
                                     double alpha,
                                     double learning_rate,
                                     double offset,
                                     double[::1, :] Q,
                                     double[:, ::1] P,
                                     double[::1, :] A,
                                     double[::1, :] B,
                                     long[:] counter,
                                     double[::1, :] E,
                                     double[:] reg,
                                     double[:] weights,
                                     double[::1, :] G,
                                     double[::1, :] beta,
                                     double[:] impute_mult,  # [E_norm, multiplier]
                                     bint impute,
                                     bint exact_E,
                                     bint persist_P,
                                     double[::1, :] Q_subset,
                                     double[::1, :] P_temp,
                                     double[::1, :] G_temp,
                                     double[::1, :] X_temp,
                                     char[:] subset_mask,
                                     int[:] dict_subset,
                                     int[:] dict_subset_lim,
                                     ) except *:
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
    beta: Algorithm variable
    impute: Online update of Gram matrix
    Q_subset : Temporary array. Holds the subdictionary
    P_temp: Temporary array. Holds the codes for the mini batch
    G_temp: Temporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    dict_subset: for union of seen features
    dict_subset_lim: for union of seen features (holds the number of seen
     features)
    weights_temp: Temporary array. Holds the update weights
    n_iter: Updated by the function, for early stopping
    max_n_iter: Iteration budget
    update_P: keeps an updated version of the code
    """
    cdef int len_batch = row_batch.shape[0]
    cdef int n_components = Q.shape[0]
    cdef int ii, i, j, jj, k, idx_j
    cdef int l = 0
    cdef int[:] subset
    subset_mask[:] = 0
    for ii in range(len_batch):
        i = row_batch[ii]
        subset = X_indices[X_indptr[i]:X_indptr[i + 1]]
        len_subset = subset.shape[0]
        for jj in range(len_subset):
            idx_j = X_indptr[i] + jj
            X_temp[0, jj] = X_data[idx_j]
            j = subset[jj]
            if not subset_mask[j]:
                subset_mask[j] = 1
                dict_subset[l] = j
                l += 1

        _update_code(X_temp[:1],
                     subset,
                     sample_subset[i:i + 1],
                     alpha,
                     learning_rate,
                     offset,
                     Q, P,
                     A, B,
                     counter,
                     E,
                     reg,
                     weights,
                     G,
                     beta,
                     impute_mult,
                     impute,
                     1.,
                     exact_E,
                     persist_P,
                     Q_subset,
                     P_temp,
                     G_temp,
                     subset_mask,
                     )
    dict_subset_lim[0] = l

cpdef void _update_code(double[::1, :] X_temp,
                        int[:] this_subset,
                        long[:] sample_subset,
                        double alpha,
                        double learning_rate,
                        double offset,
                        double[::1, :] Q,
                        double[:, ::1] P,
                        double[::1, :] A,
                        double[::1, :] B,
                        long[:] counter,
                        double[::1, :] E,
                        double[:] reg,
                        double[:] weights,
                        double[::1, :] G,
                        double[::1, :] beta,
                        double[:] impute_mult,  # [multiplier, E_norm, F]
                        bint impute,
                        double reduction,
                        bint exact_E,
                        bint persist_P,
                        double[::1, :] Q_subset,
                        double[::1, :] P_temp,
                        double[::1, :] G_temp,
                        char[:] subset_mask) except *:
    """
    Compute code for a mini-batch and update algorithm statistics accordingly

    Parameters
    ----------
    X: masked data matrix
    this_subset: indices (loci) of masked data
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
    cdef int batch_size = X_temp.shape[0]
    cdef int len_subset = this_subset.shape[0]
    cdef int n_components = Q.shape[0]
    cdef int n_cols = Q.shape[1]
    cdef double* Q_subset_ptr = &Q_subset[0, 0]
    cdef double* P_temp_ptr = &P_temp[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* G_temp_ptr = &G_temp[0, 0]
    cdef double* X_temp_ptr = &X_temp[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double v, this_alpha
    cdef int last = 0
    cdef double one_m_w, w, wd, w_norm, sum_reg_strength
    cdef double reg_strength, inv_reg_strength, sum_X, this_sample_reg

    for jj in range(len_subset):
        j = this_subset[jj]
        for k in range(n_components):
            Q_subset[k, jj] = Q[k, j]

    counter[0] += batch_size

    for jj in range(len_subset):
        j = this_subset[jj]
        counter[j + 1] += batch_size

    if impute:
        this_alpha = alpha
        for ii in range(batch_size):
            for jj in range(len_subset):
                j = this_subset[jj]
                if reduction == 1.:
                    X_temp[ii, jj] /= (1. *  counter[j + 1]) / counter[0]
                else:
                    X_temp[ii, jj] *= reduction
    else:
        this_alpha = alpha * (1. * len_subset) / n_cols
    # P_temp = Q_subset.dot(X_temp)
    dgemm(&NTRANS, &TRANS,
          &n_components, &batch_size, &len_subset,
          &oned,
          Q_subset_ptr, &n_components,
          X_temp_ptr, &batch_size,
          &zerod,
          P_temp_ptr, &n_components
          )
    w = pow((1. + offset) / (offset + counter[0]), learning_rate)

    if impute:
        sum_reg_strength = 0
        if w != 1:
            impute_mult[0] *= 1 - w
        w_norm = w / impute_mult[0]

        sum_reg_strength = 0
        for ii in range(batch_size):
            i = sample_subset[ii]
            reg_strength = 0
            inv_reg_strength = 1
            for jj in range(n_components):
                if P[i, jj]:
                    reg_strength = 1
                    sum_reg_strength += 1
                    break
            inv_reg_strength = reg_strength
            # if reg_strength != 0:
            #     inv_reg_strength = 1. / reg_strength
            # else:
            #     inv_reg_strength = 0
            # sum_reg_strength += reg_strength
            sum_X = 0
            for jj in range(len_subset):
                sum_X += X_temp[ii, jj] ** 2
            reg[i] += w_norm * (
                2. * this_alpha + sum_X * inv_reg_strength)
            weights[i] += w_norm

            for jj in range(n_components):
                beta[i, jj] += w_norm * (
                    P_temp[jj, ii] + P[i, jj] * sum_X * inv_reg_strength)
                P_temp[jj, ii] = beta[i, jj]

            this_sample_reg = reg[i] / weights[i]
            # G = Q.T.dot(Q)
            for j in range(n_components):
                for k in range(n_components):
                    G_temp[j, k] = G[j, k]
                G_temp[j, j] += this_sample_reg

            dposv(&UP, &n_components, &one, G_temp_ptr, &n_components,
                  P_temp_ptr + ii * n_components, &n_components,
                  &info)
            if info != 0:
                raise ValueError
            for jj in range(n_components):
                P_temp[jj, ii] /= weights[i]
                P[i, jj] = P_temp[jj, ii]
        impute_mult[1] += w_norm / batch_size * sum_reg_strength
        if exact_E:
            for ii in range(n_components):
                for jj in range(n_cols):
                    E[ii, jj] += w_norm / batch_size * Q[ii, jj] * sum_reg_strength

    else:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &zerod,
              G_temp_ptr, &n_components
              )
        # C.flat[::n_components + 1] += this_alpha
        for p in range(n_components):
            G_temp[p, p] += this_alpha
        # P[j] = linalg.solve(G_temp, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        dposv(&UP, &n_components, &batch_size, G_temp_ptr, &n_components,
              P_temp_ptr, &n_components,
              &info)
        if persist_P:
            for ii in range(batch_size):
                i = sample_subset[ii]
                for k in range(n_components):
                    P[i, k] = P_temp[k, ii]
        if info != 0:
            raise ValueError

    # A += P[row_batch].T.dot(P[row_batch]) * w
    if impute:
        wd = w_norm / batch_size
    else:
        wd = w / batch_size
        one_m_w = 1 - w
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &batch_size,
          &wd,
          P_temp_ptr, &n_components,
          P_temp_ptr, &n_components,
          &oned if impute else &one_m_w,
          A_ptr, &n_components
          )
    # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
    for jj in range(len_subset):
        j = this_subset[jj]
        subset_mask[j] = 1
        # Reuse Q_subset as B_subset
        if impute:
            w = w_norm
        else:
            w = pow((1. + offset) / (offset + counter[j + 1]), learning_rate)
        for k in range(n_components):
            Q_subset[k, jj] = B[k, j]
            if not impute:
                 Q_subset[k, jj] *= (1 - w)
        for ii in range(batch_size):
            X_temp[ii, jj] *= w / batch_size
    dgemm(&NTRANS, &NTRANS,
          &n_components, &len_subset, &batch_size,
          &oned,
          P_temp_ptr, &n_components,
          X_temp_ptr, &batch_size,
          &oned,
          Q_subset_ptr, &n_components)
    for jj in range(len_subset):
        j = this_subset[jj]
        for k in range(n_components):
            B[k, j] = Q_subset[k, jj]



cpdef void _update_dict(double[::1, :] Q,
                  int[:] subset,
                  bint freeze_first_col,
                  double l1_ratio,
                  bint full_projection,
                  double[::1, :] A,
                  double[::1, :] B,
                  double[::1, :] E,
                  double[::1, :] G,
                  double[:] impute_mult,
                  bint impute,
                  bint exact_E,
                  double[::1, :] R,
                  double[::1, :] Q_subset,
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
    cdef double old_norm = 0
    cdef unsigned int k, kk, j, jj

    for k in range(n_components):
        if impute:
            A[k, k] += impute_mult[1]
        for jj in range(len_subset):
            j = subset[jj]
            if impute:
                if exact_E:
                    R[k, jj] = B[k, j] + E[k, j]
                else:
                    R[k, jj] = B[k, j] + impute_mult[1] * Q[k, j]
            else:
                R[k, jj] = B[k, j]
            Q_subset[k, jj] = Q[k, j]

    for kk in range(components_range_len):
        k = components_range[kk]
        if full_projection:
            norm[k] = enet_norm(Q[k], l1_ratio)
        else:
            norm[k] = enet_norm(Q_subset[k, :len_subset], l1_ratio)

    if impute and not full_projection:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &moned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
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
                j = subset[jj]
                Q[k, j] = Q_subset[k, jj]
        if full_projection:
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
    if impute:
        for k in range(n_components):
            A[k, k] -= impute_mult[1]

    if full_projection:
        if impute:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &n_features,
                  &oned,
                  Q_ptr, &n_components,
                  Q_ptr, &n_components,
                  &zerod,
                  G_ptr, &n_components
                  )
    else:
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