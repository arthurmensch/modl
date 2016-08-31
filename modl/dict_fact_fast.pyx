# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython.parallel import parallel, prange
from libc.math cimport pow, ceil, floor, fmin, fmax, fabs
from posix.time cimport gettimeofday, timeval, timezone, suseconds_t
from scipy.linalg.cython_blas cimport dgemm, dger, daxpy, ddot, dasum, dgemv
from scipy.linalg.cython_lapack cimport dposv

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
    """Generate a random longeger in [0; end)."""
    return our_rand_r(random_state) % end


cdef void _shuffle(long[:] arr, UINT32_t* random_state) nogil:
    cdef long len_arr = arr.shape[0]
    cdef long i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_int(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]

cdef void _shuffle_long(long[:] arr, UINT32_t* random_state) nogil:
    cdef long len_arr = arr.shape[0]
    cdef long i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_int(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]


cdef double abs_max(int n, double* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef double max(int n, double* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m

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
            w[jj + 1] = fmin(1, w[0] * float(counter[0]) / counter[j + 1])

cpdef double _get_simple_weights(long count, long batch_size,
           double learning_rate, double offset) nogil:
    cdef long i
    cdef double w = 1
    for i in range(count + 1 - batch_size, count + 1):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w

cdef int _update_code(double[::1, :] full_X,
                        long[:] subset,
                        long[:] this_sample_subset,
                        double alpha,
                        double pen_l1_ratio,
                        double tol,
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
                        double[::1, :, :] G_average_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] D_subset,
                        double[::1, :] Dx,
                        double[::1, :] G_temp,
                        double[::1, :] this_X,
                        double[:, ::1] H,
                        double[:, ::1] XtA,
                        UINT32_t* random_seed,
                        int num_threads) nogil:
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
    cdef int ii, jj, i, j, k, m, p, q
    cdef int nnz
    cdef double v
    cdef int last = 0
    cdef double one_m_w, w_sample, w_batch, w_norm, w_A, w_B
    cdef double reduction = float(n_cols) / len_subset
    cdef int subset_thread_size = int(ceil(float(n_cols) / num_threads))
    cdef int this_subset_thread_size
    cdef double this_X_norm
    cdef timeval tv0, tv1
    cdef timezone tz
    cdef suseconds_t aggregation_time, coding_time, prepare_time

    gettimeofday(&tv0, &tz)
    for ii in range(len_batch):
        for jj in range(len_subset):
            j = subset[jj]
            this_X[ii, jj] = full_X[ii, j]

    for jj in range(len_subset):
        j = subset[jj]
        for k in range(n_components):
            D_subset[k, jj] = D_[k, j]

    this_X_norm = ddot(&len_subset, this_X_ptr, &one, this_X_ptr, &one) * reduction
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
                        G_average_[p, q, i] *= 1 - w_sample
                        G_average_[p, q, i] += G_temp[p, q] * w_sample
        else:
            if pen_l1_ratio == 0:
                for p in range(n_components):
                    for q in range(n_components):
                        G_temp[p, q] = G_[p, q]
            else:
                G_temp = G_
    gettimeofday(&tv1, &tz)
    prepare_time = tv1.tv_usec - tv0.tv_usec
    gettimeofday(&tv0, &tz)
    if pen_l1_ratio == 0:
        if solver == 3:
            for ii in range(len_batch):
                i = this_sample_subset[ii]
                for p in range(n_components):
                    for q in range(n_components):
                        G_temp[p, q] = G_average_[p, q, i]
                    G_temp[p, p] += alpha
                dposv(&UP, &n_components, &len_batch, G_temp_ptr, &n_components,
                      Dx_ptr + ii * n_components, &one,
                      &info)
                for p in range(n_components):
                    G_temp[p, p] -= alpha
            if info != 0:
                return -1
        else:
            for p in range(n_components):
                G_temp[p, p] += alpha
            dposv(&UP, &n_components, &len_batch, G_temp_ptr, &n_components,
                  Dx_ptr, &n_components,
                  &info)
            if info != 0:
                return -1
        for ii in range(len_batch):
            for k in range(n_components):
                code_[i, k] = Dx[k, ii]

    else:
        with parallel(num_threads=num_threads):
            for ii in prange(len_batch, schedule='static'):
                i = this_sample_subset[ii]
                enet_coordinate_descent_gram_(
                    code_[i], alpha * pen_l1_ratio,
                              alpha * (1 - pen_l1_ratio),
                    G_average_[:, :, i] if solver == 3 else G_temp,
                    Dx[:, ii],
                    this_X_norm,
                    H[ii],
                    XtA[ii],
                    1000,
                    tol, random_seed, 0, 0)
                for p in range(n_components):
                    Dx[p, ii] = code_[i, p]
    for jj in range(len_subset):
        for ii in range(len_batch):
            this_X[ii, jj] /= reduction
    gettimeofday(&tv1, &tz)
    coding_time = tv1.tv_usec - tv0.tv_usec

    gettimeofday(&tv0, &tz)
    w_A = _get_simple_weights(counter_[0], len_batch, learning_rate,
                              offset)

    # Dx = this_code
    w_batch = w_A / len_batch
    one_m_w = 1 - w_A
    # A_ *= 1 - w_A
    # A_ += this_code.dot(this_code.T) * w_A / batch_size
    dgemm(&NTRANS, &TRANS,
          &n_components, &n_components, &len_batch,
          &w_batch,
          Dx_ptr, &n_components,
          Dx_ptr, &n_components,
          &one_m_w,
          A_ptr, &n_components
          )
    if weights == 1:
        with parallel(num_threads=num_threads):
            for ii in prange(0, n_cols, subset_thread_size):
                if n_cols - ii < subset_thread_size:
                    this_subset_thread_size = n_cols - ii
                else:
                    this_subset_thread_size = subset_thread_size
                # printf("%i, %i\n", ii, this_subset_thread_size)
                dgemm(&NTRANS, &NTRANS,
                      &n_components, &this_subset_thread_size, &len_batch,
                      &w_batch,
                      Dx_ptr, &n_components,
                      full_X_ptr + ii * len_batch, &len_batch,
                      &one_m_w,
                      B_ptr + ii * n_components, &n_components)
    else:
        # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
        # Reuse D_subset as B_subset
        for jj in range(len_subset):
            j = subset[jj]
            if weights == 2:
                w_B = fmin(1., w_A * float(counter_[0]) / counter_[j + 1])
            else:
                w_B = fmin(1, w_A * reduction)
            for k in range(n_components):
                D_subset[k, jj] = B_[k, j] * (1. - w_B)
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
    gettimeofday(&tv1, &tz)
    aggregation_time = tv1.tv_usec - tv0.tv_usec
    # printf('Prepare time %i us, coding time %i us, aggregation time %i us\n',
    #        prepare_time, coding_time, aggregation_time)
    return 0

cdef void _update_dict(double[::1, :] D_,
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
                  double[:] proj_temp) nogil:
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

    cdef timeval tv0, tv1
    cdef timezone tz
    cdef suseconds_t gram_time, bcd_time

    for k in range(n_components):
        for jj in range(len_subset):
            j = subset[jj]
            D_subset[k, jj] = D_[k, j]
            R[k, jj] = B_[k, j]

    gettimeofday(&tv0, &tz)
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
    gettimeofday(&tv1, &tz)
    gram_time = tv1.tv_usec - tv0.tv_usec

    gettimeofday(&tv0, &tz)

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
            if A_[k, k] > 1e-20:
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
    gettimeofday(&tv1, &tz)
    bcd_time = tv1.tv_usec - tv0.tv_usec

    gettimeofday(&tv0, &tz)

    if solver == 2:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_subset,
              &oned,
              D_subset_ptr, &n_components,
              D_subset_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )
    gettimeofday(&tv1, &tz)
    gram_time += tv1.tv_usec - tv0.tv_usec

    # printf('Gram time %i us, BCD time %i us\n', gram_time, bcd_time)

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
                    double tol,
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
                    double[::1, :, :] G_average_,
                    long[:] n_iter_,
                    long[:] counter_,
                    long[:] row_counter_,
                    double[::1, :] D_subset,
                    double[::1, :] Dx,
                    double[::1, :] G_temp,
                    double[::1, :] this_X,
                    double[::1, :] full_X,
                    double[:, ::1] H,
                    double[:, ::1] XtA,
                    long[:] subset_range,
                    long[:] subset_temp,
                    long[:] subset_lim,
                    long[:] dict_subset_range,
                    long[:] dict_subset_lim,
                    long[:] this_sample_subset,
                    double[::1, :] R,
                    long[:] D_range,
                    double[:] norm_temp,
                    double[:] proj_temp,
                    long verbose,
                    int num_threads,
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
    with nogil:
        for i in range(n_batches):
            if verbose and n_iter_[0] - old_n_iter >= new_verbose_iter_:
                with gil:
                    print("Iteration %i" % n_iter_[0])
                    new_verbose_iter_ += n_rows // verbose
                    _callback()
            start = i * batch_size
            stop = start + batch_size
            if stop > len_row_range:
                stop = len_row_range
            len_batch = stop - start
            row_batch = row_range[start:stop]
            _update_subset_(subset_sampling == 1,
                           len_subset,
                           subset_range,
                           subset_lim,
                           subset_temp,
                           &random_seed)
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
                         tol,
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
                         H,
                         XtA,
                         &random_seed,
                         num_threads)
            _shuffle(D_range, &random_seed)

            if dict_subset_sampling == 1:
                _update_subset_(subset_sampling == 1,
                               len_subset,
                               dict_subset_range,
                               dict_subset_lim,
                               subset_temp,
                               &random_seed)
                subset = dict_subset_range[dict_subset_lim[0]:dict_subset_lim[1]]

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


def _update_subset(bint replacement,
                   long _len_subset,
                   long[:] _subset_range,
                   long[:] _subset_lim,
                   long[:] _temp_subset,
                   UINT32_t random_seed):
    _update_subset_(replacement,
                   _len_subset,
                   _subset_range,
                   _subset_lim,
                   _temp_subset,
                   &random_seed)


cdef void _update_subset_(bint replacement,
                   long _len_subset,
                   long[:] _subset_range,
                   long[:] _subset_lim,
                   long[:] _temp_subset,
                   UINT32_t* random_seed) nogil:
    cdef long n_cols = _subset_range.shape[0]
    cdef long remainder
    if replacement:
        _shuffle_long(_subset_range, random_seed)
        _subset_lim[0] = 0
        _subset_lim[1] = _len_subset
    else:
        if _len_subset != n_cols:
            _subset_lim[0] = _subset_lim[1]
            remainder = n_cols - _subset_lim[0]
            if remainder == 0:
                _shuffle_long(_subset_range, random_seed)
                _subset_lim[0] = 0
            elif remainder < _len_subset:
                _temp_subset[:remainder] = _subset_range[0:remainder]
                _subset_range[0:remainder] = _subset_range[_subset_lim[0]:]
                _subset_range[_subset_lim[0]:] = _temp_subset[:remainder]
                _shuffle_long(_subset_range[remainder:], random_seed)
                _subset_lim[0] = 0
            _subset_lim[1] = _subset_lim[0] + _len_subset
        else:
            _subset_lim[0] = 0
            _subset_lim[1] = n_cols


def enet_coordinate_descent_gram(double[:] w, double alpha, double beta,
                                 double[::1, :] Q,
                                 double[:] q,
                                 double[:] y,
                                 double[:] H,
                                 double[:] XtA,
                                 int max_iter, double tol, UINT32_t random_seed,
                                 bint random, bint positive):
    cdef int n_samples = y.shape[0]
    cdef double * y_ptr = &y[0]
    cdef double y_norm2 = ddot(&n_samples, y_ptr, &one, y_ptr, &one)

    enet_coordinate_descent_gram_(w, alpha, beta, Q, q, y_norm2, H, XtA, max_iter,
                                 tol,
                                 &random_seed,
                                 random,
                                 positive)


cdef void enet_coordinate_descent_gram_(double[:] w, double alpha, double beta,
                                 double[::1, :] Q,
                                 double[:] q,
                                 double y_norm2,
                                 double[:] H,
                                 double[:] XtA,
                                 int max_iter, double tol, UINT32_t* random_seed,
                                 bint random, bint positive) nogil:
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef int n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    # cdef double[:] XtA = np.zeros(n_features)
    # cdef double[:] H = np.dot(Q, w)

    cdef double tmp
    cdef double w_ii
    cdef double mw_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef unsigned int ii
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter

    cdef double* w_ptr = &w[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* q_ptr = &q[0]
    cdef double* H_ptr = &H[0]
    cdef double* XtA_ptr = &XtA[0]
    cdef double w_norm2
    cdef double const
    cdef double q_dot_w

    tol = tol * y_norm2

    dgemv(&NTRANS,
          &n_features, &n_features,
          &oned,
          Q_ptr, &n_features,
          w_ptr, &one,
          &zerod,
          H_ptr, &one
          )

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            if random:
                ii = rand_int(n_features, random_seed)
            else:
                ii = f_iter

            if Q[ii, ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                # H -= w_ii * Q[ii]
                mw_ii = -w_ii
                daxpy(&n_features, &mw_ii, Q_ptr + ii * n_features, &one,
                      H_ptr, &one)

            tmp = q[ii] - H[ii]

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (Q[ii, ii] + beta)

            if w[ii] != 0.0:
                # H +=  w[ii] * Q[ii] # Update H = X.T X w
                daxpy(&n_features, &w[ii], Q_ptr + ii * n_features, &one,
                      H_ptr, &one)

            # update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if fabs(w[ii]) > w_max:
                w_max = fabs(w[ii])

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            # q_dot_w = np.dot(w, q)
            q_dot_w = ddot(&n_features, w_ptr, &one, q_ptr, &one)

            for ii in range(n_features):
                XtA[ii] = q[ii] - H[ii] - beta * w[ii]
            if positive:
                dual_norm_XtA = max(n_features, XtA_ptr)
            else:
                dual_norm_XtA = abs_max(n_features, XtA_ptr)

            # temp = np.sum(w * H)
            tmp = 0.0
            for ii in range(n_features):
                tmp += w[ii] * H[ii]
            R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w

            # w_norm2 = np.dot(w, w)
            w_norm2 = ddot(&n_features, &w[0], &one, &w[0], &one)

            if dual_norm_XtA > alpha:
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            # The call to dasum is equivalent to the L1 norm of w
            gap += (alpha * dasum(&n_features, &w[0], &one) -
                    const * y_norm2 +  const * q_dot_w +
                    0.5 * beta * (1 + const ** 2) * w_norm2)

            if gap < tol:
                # return if we reached desired tolerance
                break

    # return w, gap, tol, n_iter + 1


cpdef void sparse_coding(double alpha,
                         double l1_ratio,
                         double tol,
                         double[:, ::1] code,
                         double[:, ::1] H,
                         double[:, ::1] XtA,
                         UINT32_t random_seed,
                         double[::1, :] G,
                         double[::1, :] Dx,
                         double[:, ::1] X,
                         int num_threads) nogil:
    cdef int n_rows = code.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int i
    cdef double X_norm
    cdef double * X_ptr = &X[0, 0]
    cdef UINT32_t this_random_seed
    with nogil, parallel(num_threads=num_threads):
        for i in prange(n_rows, schedule='static'):
            X_norm = ddot(&n_cols, X_ptr + i * n_cols, &one, X_ptr +  i * n_cols, &one)
            this_random_seed = random_seed
            enet_coordinate_descent_gram_(
                        code[i], alpha * l1_ratio,
                                    alpha * (1 - l1_ratio),
                        G, Dx[:, i], X_norm,
                        H[i],
                        XtA[i],
                        1000,
                        tol, &this_random_seed, 0, 0)