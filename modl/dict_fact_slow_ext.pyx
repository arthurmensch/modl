# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport numpy as np
ctypedef np.uint32_t UINT32_t
from libc.math cimport pow, ceil, floor, fmin, fmax, fabs
from scipy.linalg.cython_blas cimport dger, daxpy, ddot, dasum, dgemv

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


cpdef double _get_simple_weights(long count, long batch_size,
           double learning_rate, double offset) nogil:
    cdef long i
    cdef double w = 1
    for i in range(count + 1 - batch_size, count + 1):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w

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