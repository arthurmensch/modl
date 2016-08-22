# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
import numpy as np
from libc.math cimport pow
from libc.math cimport ceil
from scipy.linalg.cython_blas cimport dgemm, dger
from scipy.linalg.cython_lapack cimport dposv
from sklearn.linear_model import cd_fast

from ._utils.enet_proj_fast cimport enet_projection_inplace, enet_norm

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int zero = 0
cdef int one = 1
cdef double zerod = 0
cdef double oned = 1
cdef double moned = -1

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

cdef void _shuffle_int(int[:] arr, UINT32_t* random_state):
    cdef int len_arr = arr.shape[0]
    cdef int i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_int(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]

cpdef void _get_weights(double[:] w, int[:] subset, long[:] counter, long batch_size,
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
        w[jj + 1] = w[0] * float(counter[0]) / counter[j + 1]

cpdef void _update_code(double[::1, :] this_X,
                        int[:] subset,
                        long[:] sample_subset,
                        double alpha,
                        double pen_l1_ratio,
                        double learning_rate,
                        double offset,
                        bint present_boost,
                        long projection,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[::1, :] beta_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] _D_subset,
                        double[::1, :] _code_temp,
                        double[::1, :] _G_temp,
                        double[:] _w_temp,
                        np.ndarray[double, ndim=2, mode='c'] _beta_temp,
                        object rng) except *:
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
    _D_subset : Temporary array. Holds the subdictionary
    _code_temp: Temporary array. Holds the codes for the mini batch
    _G_temp: emporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    weights: Temporary array. Holds the update weights

    """
    cdef int len_batch = sample_subset.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef double* D_subset_ptr = &_D_subset[0, 0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* B_ptr = &B_[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double* code_temp_ptr = &_code_temp[0, 0]
    cdef double* G_temp_ptr = &_G_temp[0, 0]
    cdef double* this_X_ptr = &this_X[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double v
    cdef int last = 0
    cdef double one_m_w, w, wdbatch, w_norm
    cdef double reduction = float(n_cols) / len_subset

    cdef double sample_learning_rate = 2.5 - 2 * learning_rate

    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            _D_subset[k, jj] = D_[k, j]

    counter_[0] += len_batch

    for jj in range(len_subset):
        j = subset[jj]
        counter_[j + 1] += len_batch
        for ii in range(len_batch):
            this_X[ii, jj] *= reduction
    _get_weights(_w_temp, subset, counter_, len_batch,
                 learning_rate, offset)

    # P_temp = np.dot(D_subset, this_X.T)
    dgemm(&NTRANS, &TRANS,
          &n_components, &len_batch, &len_subset,
          &oned,
          D_subset_ptr, &n_components,
          this_X_ptr, &len_batch,
          &zerod,
          code_temp_ptr, &n_components
          )

    for ii in range(len_batch):
        i = sample_subset[ii]
        row_counter_[i] += 1
        w = pow(row_counter_[i], -sample_learning_rate)
        for p in range(n_components):
            beta_[i, p] *= 1 - w
            beta_[i, p] += _code_temp[p, ii] * w
            if present_boost:
                _code_temp[p, ii] /= reduction
                _code_temp[p, ii] += beta_[i, p] * (1 - 1. / reduction)
            else:
                _code_temp[p, ii] = beta_[i, p]
    if pen_l1_ratio == 0:
        for p in range(n_components):
            for q in range(n_components):
                _G_temp[p, q] = G_[p, q]
            _G_temp[p, p] += alpha
        dposv(&UP, &n_components, &len_batch, G_temp_ptr, &n_components,
              code_temp_ptr, &n_components,
              &info)
        if info != 0:
            raise ValueError
    else:
        for ii in range(len_batch):
            i = sample_subset[ii]
            for p in range(n_components):
                _beta_temp[ii, p] = _code_temp[p, ii]
                _code_temp[p, ii] = code_[i, p]
            cd_fast.enet_coordinate_descent_gram(
                _code_temp[:, ii], alpha * pen_l1_ratio,
                                   alpha * (1 - pen_l1_ratio),
                np.asarray(G_.T), _beta_temp[ii],
                np.asarray(this_X[ii], order='C'), 100,
                1e-3, rng, True, False)

    wdbatch = _w_temp[0] / len_batch
    one_m_w = 1 - _w_temp[0]
    # A_ *= 1 - w_A
    # A_ += this_code.dot(this_code.T) * w_A / batch_size
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &len_batch,
          &wdbatch,
          code_temp_ptr, &n_components,
          code_temp_ptr, &n_components,
          &one_m_w,
          A_ptr, &n_components
          )

    # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
    # Reuse D_subset as B_subset
    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            _D_subset[k, jj] = B_[k, j] * (1 - _w_temp[jj + 1])
        for ii in range(len_batch):
            this_X[ii, jj] *= _w_temp[jj + 1] / len_batch / reduction
    dgemm(&NTRANS, &NTRANS,
          &n_components, &len_subset, &len_batch,
          &oned,
          code_temp_ptr, &n_components,
          this_X_ptr, &len_batch,
          &oned,
          D_subset_ptr, &n_components)
    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            B_[k, j] = _D_subset[k, jj]

    for ii in range(len_batch):
        i = sample_subset[ii]
        for k in range(n_components):
            code_[i, k] = _code_temp[k, ii]

cpdef void _update_dict(double[::1, :] D_,
                  int[:] dict_subset,
                  bint fit_intercept,
                  double l1_ratio,
                  long projection,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  long[:] _D_range,
                  double[::1, :] _R,
                  double[::1, :] _D_subset,
                  double[:] _norm_temp,
                  double[:] _proj_temp):
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef int len_subset = dict_subset.shape[0]
    cdef unsigned int components_range_len = _D_range.shape[0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* D_subset_ptr = &_D_subset[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* R_ptr = &_R[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double old_norm = 0
    cdef unsigned int k, kk, j, jj

    for k in range(n_components):
        for jj in range(len_subset):
            j = dict_subset[jj]
            _D_subset[k, jj] = D_[k, j]
            _R[k, jj] = B_[k, j]

    for kk in range(components_range_len):
        k = _D_range[kk]
        if projection == 1:
            _norm_temp[k] = enet_norm(D_[k], l1_ratio)
        else:
            _norm_temp[k] = enet_norm(_D_subset[k, :len_subset], l1_ratio)

    if projection == 2:
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
        k = _D_range[kk]
        dger(&n_components, &len_subset, &oned,
             A_ptr + k * n_components,
             &one, D_subset_ptr + k, &n_components, R_ptr, &n_components)

        for jj in range(len_subset):
            if A_[k, k] != 0:
                _D_subset[k, jj] = _R[k, jj] / A_[k, k]

        if projection == 1:
            for jj in range(len_subset):
                j = dict_subset[jj]
                D_[k, j] = _D_subset[k, jj]
            enet_projection_inplace(D_[k], _proj_temp,
                                    _norm_temp[k], l1_ratio)
            for jj in range(n_cols):
                D_[k, jj] = _proj_temp[jj]
            for jj in range(len_subset):
                j = dict_subset[jj]
                _D_subset[k, jj] = D_[k, j]
        else:
            enet_projection_inplace(_D_subset[k, :len_subset],
                                    _proj_temp[:len_subset],
                                    _norm_temp[k], l1_ratio)
            for jj in range(len_subset):
                _D_subset[k, jj] = _proj_temp[jj]
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &len_subset, &moned,
             A_ptr + k * n_components,
             &one, D_subset_ptr + k, &n_components, R_ptr, &n_components)

    if projection == 2:
        for jj in range(len_subset):
            j = dict_subset[jj]
            for kk in range(n_components):
                D_[kk, j] = _D_subset[kk, jj]
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )
    else:
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

def dict_learning_dense(double[:, ::1] X,
                    long[:] row_range,
                    long[:] sample_subset,
                    long batch_size,
                    double alpha,
                    double learning_rate,
                    double offset,
                    bint fit_intercept,
                    double l1_ratio,
                    double pen_l1_ratio,
                    bint present_boost,
                    long projection,
                    bint replacement,
                    double[::1, :] D_,
                    double[:, ::1] code_,
                    double[::1, :] A_,
                    double[::1, :] B_,
                    double[::1, :] G_,
                    double[::1, :] beta_,
                    long[:] counter_,
                    long[:] row_counter_,
                    double[::1, :] _D_subset,
                    double[::1, :] _code_temp,
                    double[::1, :] _G_temp,
                    double[::1, :] _this_X,
                    double[:] _w_temp,
                    long _len_subset,
                    int[:] _subset_range,
                    int[:] _temp_subset,
                    int[:] _subset_lim,
                    long[:] _this_sample_subset,
                    double[::1, :] _R,
                    long[:] _D_range,
                    double[:] _norm_temp,
                    double[:] _proj_temp,
                    UINT32_t random_seed,
                    long verbose,
                    long[:] n_iter_,
                    _callback):

    cdef int len_row_range = row_range.shape[0]
    cdef long[:] row_batch
    cdef int n_batches = int(ceil(len_row_range / batch_size))
    cdef int start = 0
    cdef int stop = 0
    cdef int len_batch = 0

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int n_components = D_.shape[0]

    cdef long old_n_iter = n_iter_[0]
    cdef long new_verbose_iter_ = 0

    cdef int i, ii, jj, j

    cdef int[:] subset

    rng = np.random.RandomState(random_seed)

    _beta_temp = np.zeros((batch_size, n_components), order='C')

    for i in range(n_batches):
        if verbose and n_iter_[0] - old_n_iter >= new_verbose_iter_:
            print("Iteration %i" % n_iter_[0])
            new_verbose_iter_ += n_rows // verbose
            _callback()
        start = i * batch_size
        stop = start + batch_size
        if stop > len_row_range:
            stop = len_row_range
        len_batch = stop - start
        row_batch = row_range[start:stop]

        _update_subset(replacement,
                       _len_subset,
                       _subset_range,
                       _subset_lim,
                       _temp_subset,
                       random_seed)
        subset = _subset_range[_subset_lim[0]:_subset_lim[1]]

        for ii in range(len_batch):
            for jj in range(_len_subset):
                j = subset[jj]
                _this_X[ii, jj] = X[row_batch[ii], j]
            _this_sample_subset[ii] = sample_subset[row_batch[ii]]
        _update_code(_this_X,
                     subset,
                     _this_sample_subset[:len_batch],
                     alpha,
                     pen_l1_ratio,
                     learning_rate,
                     offset,
                     present_boost,
                     projection,
                     D_,
                     code_,
                     A_,
                     B_,
                     G_,
                     beta_,
                     counter_,
                     row_counter_,
                     _D_subset,
                     _code_temp,
                     _G_temp,
                     _w_temp,
                     _beta_temp,
                     rng)
        _shuffle(_D_range, &random_seed)
        _update_dict(D_,
                     subset,
                     fit_intercept,
                     l1_ratio,
                     projection,
                     A_,
                     B_,
                     G_,
                     _D_range,
                     _R,
                     _D_subset,
                     _norm_temp,
                     _proj_temp)
        n_iter_[0] += len_batch


cpdef void _update_subset(bint replacement,
                   long _len_subset,
                   int[:] _subset_range,
                   int[:] _subset_lim,
                   int[:] _temp_subset,
                   UINT32_t random_seed):
    cdef long n_cols = _subset_range.shape[0]
    cdef long remainder
    if replacement:
        _shuffle_int(_subset_range, &random_seed)
        _subset_lim[0] = 0
        _subset_lim[1] = _len_subset
    else:
        if _len_subset != n_cols:
            _subset_lim[0] = _subset_lim[1]
            remainder = n_cols - _subset_lim[0]
            if remainder == 0:
                _shuffle_int(_subset_range, &random_seed)
                _subset_lim[0] = 0
            elif remainder < _len_subset:
                _temp_subset[:remainder] = _subset_range[0:remainder]
                _subset_range[0:remainder] = _subset_range[_subset_lim[0]:]
                _subset_range[_subset_lim[0]:] = _temp_subset[:remainder]
                _shuffle_int(_subset_range[remainder:], &random_seed)
                _subset_lim[0] = 0
            _subset_lim[1] = _subset_lim[0] + _len_subset
        else:
            _subset_lim[0] = 0
            _subset_lim[1] = n_cols