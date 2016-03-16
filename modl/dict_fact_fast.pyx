# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

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
                                     long[:] sample_idx,
                                     double alpha,
                                     double learning_rate,
                                     double offset,
                                     double[::1, :] Q,
                                     double[:, ::1] P,
                                     double[::1, :] A,
                                     double[::1, :] B,
                                     long[:] counter,
                                     double[::1, 1] E,
                                     double[:] F,
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
                                     double[::1, :] this_X,
                                     double[:] reg_strength,
                                     double[:] inv_reg_strength,
                                     char[:] subset_mask,
                                     int[:] dict_subset,
                                     int[:] dict_subset_lim,
                                     double[:] weights,
                                     long n_iter,
                                     long max_n_iter,
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
        subset = X_indices[X_indptr[i]:X_indptr[i + 1]]
        len_subset = subset.shape[0]
        if impute:
            reg = alpha
        else:
            reg = alpha * len_subset / n_cols
        for jj in range(len_subset):
            idx_j = X_indptr[i] + jj
            this_X[0, jj] = X_data[idx_j]
            j = subset[jj]
            if not subset_mask[j]:
                subset_mask[j] = 1
                dict_subset[l] = j
                l += 1

        _update_code(this_X[:1], subset, sample_idx[i:i + 1],
                     reg, learning_rate,
                     offset, Q, P,
                     A, B,
                     counter,
                     E,
                     F,
                     reg,
                     weights,
                     G,
                     beta,
                     impute_mult,
                     impute,
                     exact_E,
                     persist_P,
                     Q_subset,
                     P_temp,
                     G_temp,
                     subset_mask,
                     weights,
                     )
        n_iter += 1
    dict_subset_lim[0] = l
    return n_iter

cpdef void _update_code(double[::1, :] this_X, int[:] this_subset,
                   long[:] sample_subset,
                   double alpha,
                   double learning_rate,
                   double offset,
                   double[::1, :] Q,
                   double[:, ::1] P,
                   double[::1, :] A,
                   double[::1, :] B,
                   long[:] counter,
                   double[::1, 1] E,
                   double[:] F,
                   double[:] reg,
                   double[:] weights,
                   double[::1, :] G,
                   double[::1, :] beta,
                   double[:] impute_mult,  # [E_norm, multiplier]
                   bint impute,
                   bint exact_E,
                   bint persist_P,
                   double[::1, :] Q_subset,
                   double[::1, :] this_P,
                   double[::1, :] this_G,
                   double[:] reg_strength,
                   double[:] inv_reg_strength,
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
    this_P: Temporary array. Holds the codes for the mini batch
    this_G: emporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    weights: Temporary array. Holds the update weights

    """
    cdef int batch_size = this_X.shape[0]
    cdef int len_subset = this_subset.shape[0]
    cdef int n_components = Q.shape[0]
    cdef int n_cols = Q.shape[1]
    cdef double* Q_subset_ptr = &Q_subset[0, 0]
    cdef double* this_P_ptr = &this_P[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* this_G_ptr = &this_G[0, 0]
    cdef double* this_X_ptr = &this_X[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double reg, v, this_alpha
    cdef int last = 0
    cdef double one_m_w_A, w_A, sum_reg_strength

    for k in range(n_components):
        for jj in range(len_subset):
            Q_subset[k, jj] = Q[k, this_subset[jj]]

    counter[0] += batch_size

    for jj in range(len_subset):
        j = this_subset[jj]
        counter[j + 1] += batch_size

    if impute:
        this_alpha = alpha
        for ii in range(batch_size):
            for jj in range(len_subset):
                j = this_subset[jj]
                this_X[ii, jj] /= counter[j + 1] / counter[0]
    else:
        this_alpha = alpha * len_subset/ n_cols

    # Qx = Q_subset.dot(x)
    dgemm(&NTRANS, &TRANS,
          &n_components, &batch_size, &len_subset,
          &oned,
          Q_subset_ptr, &n_components,
          this_X_ptr, &batch_size,
          &zerod,
          this_P_ptr, &n_components
          )
    w = pow((1. + offset) / (offset + counter[0]), learning_rate)
    one_m_w = 1 - w

    if impute:
        # G = Q.T.dot(Q)
        for j in range(n_components):
            for k in range(n_components):
                this_G[j, k] = G[j, k]
        if w != 1:
            impute_mult[1] *= 1 - w
        w_norm = w / impute_mult[1]
        for ii in range(batch_size):
            i = sample_subset[ii]
            reg_strength[ii] = 0
            for jj in range(n_components):
                reg_strength[ii] += P[i, jj] ** 2
            sum_reg_strength += reg_strength[ii]
            if reg_strength[ii] != 0:
                inv_reg_strength[ii] = 1 / reg_strength[ii]
            else:
                inv_reg_strength[ii] = 0
        # inv_reg_strength = np.where(reg_strength, 1. / reg_strength, 0)


        sum_reg_strength = 0
        for ii in range(batch_size):
            i = sample_subset[ii]
            reg_strength = 0
            for jj in range(n_components):
                reg_strength += P[i, jj] ** 2
            if reg_strength != 0:
                inv_reg_strength = 1 / reg_strength
            else:
                inv_reg_strength = 0
            sum_reg_strength += reg_strength
            sum_X = 0
            for jj in range(n_cols):
                sum_X += this_X[ii, jj] ** 2
            reg[i] += w_norm * (
                this_alpha + .5 * sum_X * inv_reg_strength)
            weights[i] += w_norm

            for jj in range(n_components):
                beta[i, jj] += w_norm * (
                    this_P.T + P[i, jj] * sum_X * inv_reg_strength)
                this_P[jj, ii] = beta[i, jj]

            i = sample_subset[ii]
            this_sample_reg = reg[i] / weights[i]
            for jj in range(n_components):
                this_G[jj, jj] += this_sample_reg
            dposv(&UP, &n_components, &batch_size, this_G_ptr, &n_components,
                  this_P_ptr + ii * n_components, &n_components,
                  &info)
            for jj in range(n_components):
                this_G[jj, jj] -= this_sample_reg
                this_P[jj, ii] /= weights[i] * impute_mult[1]
            P[i, :] = this_P[:, ii]

        if exact_E:
            for ii in range(n_components):
                for jj in range(n_cols):
                    E[ii, jj] += w_norm / batch_size * Q[ii, jj] * sum_reg_strength
        else:
            impute_mult[0] += w_norm / batch_size * sum_reg_strength
        for ii in range(n_components):
            F[ii] += w_norm / batch_size * sum_reg_strength



    else:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &zerod,
              this_G_ptr, &n_components
              )
        # C.flat[::n_components + 1] += this_alpha
        for p in range(n_components):
            this_G[p, p] += this_alpha
        # P[j] = linalg.solve(G_temp, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        dposv(&UP, &n_components, &batch_size, this_G_ptr, &n_components,
              this_P_ptr, &n_components,
              &info)
        if persist_P:
            for ii in range(batch_size):
                i = sample_subset[ii]
                for k in range(n_components):
                    P[i, k] = this_P[k, ii]
        if info != 0:
            raise ValueError

    # A += P[row_batch].T.dot(P[row_batch]) * w
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &batch_size,
          &w,
          this_P_ptr, &n_components,
          this_P_ptr, &n_components,
          &oned,
          A_ptr, &n_components
          )
    # B += this_X.T.dot(P[row_batch]) * w_B
    if impute:
        dgemm(&NTRANS, &NTRANS,
                      &n_components, &len_subset, &batch_size,
                      &oned,
                      this_P_ptr, &n_components,
                      this_X_ptr, &batch_size,
                      &oned,
                      B_ptr, &n_components)
    else:
        for jj in range(len_subset):
            j = this_subset[jj]
            subset_mask[j] = 1
            # Reuse Q_subset as B_subset
            w_B = pow((1. + offset) / (offset + counter[j + 1]), learning_rate)
            for k in range(n_components):
                Q_subset[k, jj] = B[k, j]
            for ii in range(batch_size):
                this_X[ii, jj] *= w_B / batch_size
        dgemm(&NTRANS, &NTRANS,
              &n_components, &len_subset, &batch_size,
              &oned,
              this_P_ptr, &n_components,
              this_X_ptr, &batch_size,
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
                  double[:] F,
                  double[::1, :] G,
                  double E_mult,
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

    if not full_projection:
        for kk in range(n_components):
            for jj in range(len_subset):
                j = subset[jj]
                Q[kk, j] = Q_subset[kk, jj]

    if impute and not full_projection:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              Q_subset_ptr, &n_components,
              Q_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )

    if impute and full_projection:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &n_features,
              &oned,
              Q_ptr, &n_components,
              Q_ptr, &n_components,
              &zerod,
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