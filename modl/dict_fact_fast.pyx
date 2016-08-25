# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
import numpy as np
from libc.math cimport pow, ceil, floor
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


cdef inline UINT32_t rand_long(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random longeger in [0; end)."""
    return our_rand_r(random_state) % end


cdef void _shuffle(long[:] arr, UINT32_t* random_state) nogil:
    cdef long len_arr = arr.shape[0]
    cdef long i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_long(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]

cdef void _shuffle_long(long[:] arr, UINT32_t* random_state) nogil:
    cdef long len_arr = arr.shape[0]
    cdef long i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_long(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]

cpdef void _get_weights(double[:] w, long[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset) nogil:
    cdef long len_subset = subset.shape[0]
    cdef double reduction = (counter.shape[0] - 1) / float(len_subset)
    cdef long full_count = counter[0]
    cdef long count
    cdef long i, jj, j
    w[0] = 1
    for i in range(full_count + 1, full_count + 1 + batch_size):
        w[0] *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w[0] = 1 - w[0]
    for jj in range(len_subset):
        j = subset[jj]
        if counter[j + 1] == 0:
            w[j + 1] = 1
        else:
            w[jj + 1] = min(1, w[0] * float(counter[0]) / counter[j + 1])

cpdef double _get_simple_weights(long count, long batch_size,
           double learning_rate, double offset) nogil:
    cdef long i
    cdef double w = 1
    for i in range(count + 1 - batch_size, count + 1):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w

cpdef void _update_code(double[::1, :] full_X,
                        long[:] subset,
                        long[:] this_sample_subset,
                        double alpha,
                        double pen_l1_ratio,
                        double learning_rate,
                        double sample_learning_rate,
                        double offset,
                        long solver,
                        long weights,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[:, :] Dx_average_,
                        double[:, :, ::1] G_average_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] D_subset,
                        double[::1, :] Dx,
                        double[::1, :] G_temp,
                        double[::1, :] this_X,
                        object rng) except *:
    """
    Compute code for a mini-batch and update algorithm statistics accordingly

    Parameters
    ----------
    X: masked data matrix
    this_subset: indices (loci) of masked data
    alpha: regularization parameter
    learning_rate: decrease rate in the learning sequence (in [.5, 1])
    offset: offset in the learning se   quence
    D_: Dictionary
    A_: algorithm variable
    B_: algorithm variable
    counter_: algorithm variable
    G: algorithm variable
    T: algorithm variable
    impute: Online update of Gram matrix
    D_subset : Temporary array. Holds the subdictionary
    Dx: Temporary array. Holds the codes for the mini batch
    G_temp: emporary array. Holds the Gram matrix.
    subset_mask: Holds the binary mask for visited features
    weights: Temporary array. Holds the update weights

    """
    cdef int len_batch = this_sample_subset.shape[0]
    cdef int len_subset = subset.shape[0]
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef double* D_subset_ptr = &D_subset[0, 0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* B_ptr = &B_[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double* Dx_ptr = &Dx[0, 0]
    cdef double* G_temp_ptr = &G_temp[0, 0]
    cdef double* this_X_ptr = &this_X[0, 0]
    cdef double* full_X_ptr = &full_X[0, 0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m, p
    cdef int nnz
    cdef double v
    cdef int last = 0
    cdef double one_m_w_A, w_sample, w_A_batch, w_norm
    cdef double reduction = float(n_cols) / len_subset

    for ii in range(len_batch):
        for jj in range(len_subset):
            j = subset[jj]
            this_X[ii, jj] = full_X[ii, j]

    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            D_subset[k, jj] = D_[k, j]

    for jj in range(len_subset):
        for ii in range(len_batch):
            this_X[ii, jj] *= reduction

    # Dx = np.dot(D_subset, this_X.T)
    dgemm(&NTRANS, &TRANS,
          &n_components, &len_batch, &len_subset,
          &oned,
          D_subset_ptr, &n_components,
          this_X_ptr, &len_batch,
          &zerod,
          Dx_ptr, &n_components
          )

    if solver == 1:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &reduction,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &zerod,
              G_temp_ptr, &n_components
              )
    else:
        for ii in range(len_batch):
            i = this_sample_subset[ii]
            w_sample = pow(row_counter_[i], -sample_learning_rate)
            for p in range(n_components):
                Dx_average_[i, p] *= 1 - w_sample
                Dx_average_[i, p] += Dx[p, ii] * w_sample
                Dx[p, ii] = Dx_average_[i, p]
        if solver == 3:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &len_subset,
                  &reduction,
                  D_subset_ptr, &n_components,
                  D_subset_ptr, &n_components,
                  &zerod,
                  G_temp_ptr, &n_components
                  )
            for ii in range(len_batch):
                i = this_sample_subset[ii]
                for p in range(n_components):
                    for q in range(n_components):
                        G_average_[i, p, q] *= 1 - w_sample
                        G_average_[i, p, q] += G_temp[p, q] * w_sample
        else:
            if pen_l1_ratio == 0:
                for p in range(n_components):
                    for q in range(n_components):
                        G_temp[p, q] = G_[p, q]
            else:
                G_temp = G_

    if pen_l1_ratio == 0:
        if solver == 1 or solver == 2:
            for p in range(n_components):
                G_temp[p, p] += alpha
            dposv(&UP, &n_components, &len_batch, G_temp_ptr, &n_components,
                  Dx_ptr, &n_components,
                  &info)
            if info != 0:
                raise ValueError
        else:
            for ii in range(len_batch):
                i = this_sample_subset[ii]
                for p in range(n_components):
                    for q in range(n_components):
                        G_temp[p, q] = G_average_[i, p, q]
                    G_temp[p, p] += alpha
            dposv(&UP, &n_components, &len_batch, G_temp_ptr, &n_components,
                  Dx_ptr + ii * n_components, &one,
                  &info)
            if info != 0:
                raise ValueError
            for ii in range(len_batch):
                for k in range(n_components):
                    code_[i, k] = Dx[k, ii]
    else:
        for ii in range(len_batch, nogil=True):
            i = this_sample_subset[ii]
            if solver == 3:
                G_temp = G_average_[i].T
            cd_fast.enet_coordinate_descent_gram(
                code_[i], alpha * pen_l1_ratio,
                                alpha * (1 - pen_l1_ratio),
                np.asarray(G_temp.T), np.asarray(Dx[:, ii], order='C'),
                np.asarray(this_X[ii], order='C'), 100,
                1e-3, rng, True, False)
            for p in range(n_components):
                Dx[p, ii] = code_[i, p]
    for jj in range(len_subset):
        for ii in range(len_batch):
            this_X[ii, jj] /= reduction

    w_A = _get_simple_weights(counter_[0], len_batch, learning_rate,
                              offset)

    # Dx = this_code
    w_A_batch = w_A / len_batch
    one_m_w_A = 1 - w_A
    # A_ *= 1 - w_A
    # A_ += this_code.dot(this_code.T) * w_A / batch_size
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &len_batch,
          &w_A_batch,
          Dx_ptr, &n_components,
          Dx_ptr, &n_components,
          &one_m_w_A,
          A_ptr, &n_components
          )
    if weights == 1:
        dgemm(&NTRANS, &NTRANS,
              &n_components, &n_cols, &len_batch,
              &w_A_batch,
              Dx_ptr, &n_components,
              full_X_ptr, &len_batch,
              &one_m_w_A,
              B_ptr, &n_components)
    else:
        # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
        # Reuse D_subset as B_subset
        for jj in range(len_subset):
            j = subset[jj]
            w_B = min(1, w_A * counter_[0] + counter_[j + 1])
            for k in range(n_components):
                D_subset[k, jj] = B_[k, j] * (1 - w_B)
            for ii in range(len_batch):
                this_X[ii, jj] *= w_B / len_batch
        dgemm(&NTRANS, &NTRANS,
              &n_components, &len_subset, &len_batch,
              &oned,
              Dx_ptr, &n_components,
              this_X_ptr, &len_batch,
              &oned,
              D_subset_ptr, &n_components)
        for jj in range(len_subset):
            j = subset[jj]
            for k in range(n_components):
                B_[k, j] = D_subset[k, jj]

cpdef void _update_dict(double[::1, :] D_,
                  long[:] subset,
                  double l1_ratio,
                  long solver,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  long[:] D_range,
                  double[::1, :] R,
                  double[::1, :] D_subset,
                  double[:] norm_temp,
                  double[:] proj_temp):
    cdef int len_subset = subset.shape[0]
    cdef int n_components = D_.shape[0]
    cdef int n_cols = D_.shape[1]
    cdef unsigned long components_range_len = D_range.shape[0]
    cdef double* D_ptr = &D_[0, 0]
    cdef double* D_subset_ptr = &D_subset[0, 0]
    cdef double* A_ptr = &A_[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double* G_ptr = &G_[0, 0]
    cdef double old_norm = 0
    cdef unsigned long k, kk, j, jj
    for k in range(n_components):
        for jj in range(len_subset):
            j = subset[jj]
            D_subset[k, jj] = D_[k, j]
            R[k, jj] = B_[k, j]

    for kk in range(components_range_len):
        k = D_range[kk]
        norm_temp[k] = enet_norm(D_subset[k, :len_subset], l1_ratio)
    if solver == 2:
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
            if A_[k, k] > 1e-12:
                D_subset[k, jj] = R[k, jj] / A_[k, k]
                # print(D_subset[k, jj])

        enet_projection_inplace(D_subset[k, :len_subset],
                                proj_temp[:len_subset],
                                norm_temp[k], l1_ratio)
        for jj in range(len_subset):
            D_subset[k, jj] = proj_temp[jj]
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &len_subset, &moned,
             A_ptr + k * n_components,
             &one, D_subset_ptr + k, &n_components, R_ptr, &n_components)

    for jj in range(len_subset):
        j = subset[jj]
        for kk in range(n_components):
            D_[kk, j] = D_subset[kk, jj]
    if solver == 2:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )

cpdef void _predict(double[:] X_data,
             long[:] X_indices,
             long[:] X_indptr,
             double[:, ::1] P,
             double[::1, :] Q):
    """Adapted from spira"""
    cdef long n_rows = P.shape[0]
    cdef long n_components = P.shape[1]

    cdef long n_nz
    cdef double* data
    cdef long* indices

    cdef long u, ii, i, k
    cdef double dot

    for u in range(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*> &X_data[0] + X_indptr[u]
        indices = <long*> &X_indices[0] + X_indptr[u]

        for ii in range(n_nz):
            i = indices[ii]

            dot = 0
            for k in range(n_components):
                dot += P[u, k] * Q[k, i]

            data[ii] = dot

def dict_learning(double[:, ::1] X,
                    long[:] row_range,
                    long[:] sample_subset,
                    long batch_size,
                    double alpha,
                    double learning_rate,
                    double sample_learning_rate,
                    double offset,
                    double l1_ratio,
                    double pen_l1_ratio,
                    double reduction,
                    long solver,
                    long weights,
                    long subset_sampling,
                    long dict_subset_sampling,
                    double[::1, :] D_,
                    double[:, ::1] code_,
                    double[::1, :] A_,
                    double[::1, :] B_,
                    double[::1, :] G_,
                    double[:, :] Dx_average_,
                    double[:, :, ::1] G_average_,
                    long[:] n_iter_,
                    long[:] counter_,
                    long[:] row_counter_,
                    double[::1, :] D_subset,
                    double[::1, :] Dx,
                    double[::1, :] G_temp,
                    double[::1, :] this_X,
                    double[::1, :] full_X,
                    long[:] subset_range,
                    long[:] subset_temp,
                    long[:] subset_lim,
                    long[:] this_sample_subset,
                    double[::1, :] R,
                    long[:] D_range,
                    double[:] norm_temp,
                    double[:] proj_temp,
                    long verbose,
                    UINT32_t random_seed,
                    _callback):

    cdef int len_row_range = row_range.shape[0]
    cdef int n_batches = int(ceil(len_row_range / batch_size))
    cdef int start = 0
    cdef int stop = 0
    cdef int len_batch = 0

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int n_components = D_.shape[0]

    cdef int old_n_iter = n_iter_[0]
    cdef int new_verbose_iter_ = 0

    cdef int len_subset = int(floor(n_cols / reduction))

    cdef int i, ii, jj, j

    cdef long[:] row_batch
    cdef long[:] subset

    rng = np.random.RandomState(random_seed)

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
        _update_subset(subset_sampling == 2,
                       len_subset,
                       subset_range,
                       subset_lim,
                       subset_temp,
                       random_seed)
        subset = subset_range[subset_lim[0]:subset_lim[1]]

        counter_[0] += len_batch
        for jj in range(subset.shape[0]):
            j = subset[jj]
            counter_[j + 1] += len_batch

        for ii in range(len_batch):
            i = sample_subset[row_batch[ii]]
            this_sample_subset[ii] = i
            row_counter_[i] += 1
            for jj in range(n_cols):
                full_X[ii, jj] = X[row_batch[ii], jj]
        _update_code(full_X,
                     subset,
                     this_sample_subset[:len_batch],
                     alpha,
                     pen_l1_ratio,
                     learning_rate,
                     sample_learning_rate,
                     offset,
                     solver,
                     weights,
                     D_,
                     code_,
                     A_,
                     B_,
                     G_,
                     Dx_average_,
                     G_average_,
                     counter_,
                     row_counter_,
                     D_subset,
                     Dx,
                     G_temp,
                     this_X,
                     rng)
        _shuffle(D_range, &random_seed)
        _update_dict(D_,
                     subset,
                     l1_ratio,
                     solver,
                     A_,
                     B_,
                     G_,
                     D_range,
                     R,
                     D_subset,
                     norm_temp,
                     proj_temp)
        n_iter_[0] += len_batch


cpdef void _update_subset(bint replacement,
                   long _len_subset,
                   long[:] _subset_range,
                   long[:] _subset_lim,
                   long[:] _temp_subset,
                   UINT32_t random_seed) nogil:
    cdef long n_cols = _subset_range.shape[0]
    cdef long remainder
    if replacement:
        _shuffle_long(_subset_range, &random_seed)
        _subset_lim[0] = 0
        _subset_lim[1] = _len_subset
    else:
        if _len_subset != n_cols:
            _subset_lim[0] = _subset_lim[1]
            remainder = n_cols - _subset_lim[0]
            if remainder == 0:
                _shuffle_long(_subset_range, &random_seed)
                _subset_lim[0] = 0
            elif remainder < _len_subset:
                _temp_subset[:remainder] = _subset_range[0:remainder]
                _subset_range[0:remainder] = _subset_range[_subset_lim[0]:]
                _subset_range[_subset_lim[0]:] = _temp_subset[:remainder]
                _shuffle_long(_subset_range[remainder:], &random_seed)
                _subset_lim[0] = 0
            _subset_lim[1] = _subset_lim[0] + _len_subset
        else:
            _subset_lim[0] = 0
            _subset_lim[1] = n_cols