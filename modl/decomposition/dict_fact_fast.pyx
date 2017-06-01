# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int ONE = 1

from cython cimport floating

from scipy.linalg.cython_blas cimport saxpy, daxpy, sdot, ddot, sasum, dasum, dgemv, sgemv
from scipy.linalg.cython_lapack cimport dposv, sposv

from libc.math cimport pow, fabs

cimport numpy as np
import numpy as np

from cython cimport view

ctypedef void (*POSV)(char * UPLO, int* N,
                          int* NRHS, floating* A, int* LDA,
                          floating *B, int* LDB, int* INFO)
ctypedef floating (*DOT)(int* N, floating* X, int* incX, floating* Y,
                         int* incY) nogil
ctypedef void (*AXPY)(int* N, floating* alpha, floating* X, int* incX,
                      floating* Y, int* incY) nogil
ctypedef floating (*ASUM)(int* N, floating* X, int* incX) nogil


def _enet_regression_multi_gram(floating[:, :, ::1] G, floating[:, ::1] Dx,
                                floating[:, ::1] X,
                                floating[: , ::1] code,
                                long[:] indices,
                                floating l1_ratio, floating alpha,
                                bint positive,
                                floating tol,
                                int max_iter,
                                ):
    '''
    Perform elastic net regression: for all i in indices,
    find code[i] s.t code[i].dot(G[i]) = Dx[ii], where i = indices[ii].
    G and code are the arrays containing the values for all samples,
    while Dx and X should already be subscripted.

    Parameters
    ----------
    G: array, shape (n_samples x n_components x n_components)
    Dx: array, shape (batch_size x n_components x n_components)
    X: array, shape (batch_size x n_components x n_components
    code: array, shape (n_samples x n_components)
    indices: array, shape (batch_size
    l1_ratio: floating, enet-regression parameter
    alpha: floating, enet-regression paramater
    positive: bint, enet-regression parameter
    '''
    cdef int batch_size = indices.shape[0]
    cdef int n_components = code.shape[1]
    cdef int i, j, info, ii
    cdef floating* G_ptr = <floating*> &G[0, 0, 0]
    cdef floating* code_ptr = <floating*> &code[0, 0]
    cdef POSV posv
    cdef str format


    cdef floating[:] this_code
    cdef floating[::1] this_Dx
    cdef floating[:] this_X
    cdef floating[:, ::1] this_G
    cdef floating[:] H
    cdef floating[:] XtA

    if floating is float:
        posv = sposv
        format = 'f'
    else:
        posv = dposv
        format = 'd'

    if l1_ratio == 0:
        for ii in range(batch_size):
            i = indices[ii]
            code[i, :] = Dx[ii, :]
            for j in range(n_components):
                G[ii, j, j] += alpha
            posv(&UP, &n_components, &ONE,
                G_ptr + ii * n_components ** 2,
                &n_components,
                code_ptr + i * n_components, &n_components,
                &info)
            for j in range(n_components):
                G[ii, j, j] -= alpha
    else:
        H = view.array((n_components, ), sizeof(floating),
                       format=format, mode='c')
        XtA = view.array((n_components, ), sizeof(floating),
                         format=format, mode='c')
        with nogil:
            for ii in range(batch_size):
                i = indices[ii]
                this_G = G[ii, :, :]
                this_Dx = Dx[ii, :]
                this_X = X[ii, :]
                this_code = code[i, :]
                enet_coordinate_descent_gram(
                    this_code,
                    alpha * l1_ratio,
                    alpha * (1 - l1_ratio),
                    this_G, this_Dx, this_X, H, XtA, max_iter, tol,
                    positive)
    return np.asarray(code)

def _batch_weight(long count, long batch_size,
           double learning_rate, double offset):
    cdef long i
    cdef double w = 1
    for i in range(count + 1 - batch_size, count + 1):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w


def _enet_regression_single_gram(floating[:, ::1] G, floating[:, ::1] Dx,
                                floating[:, ::1] X,
                                floating[:, ::1] code,
                                long[:] indices,
                                floating l1_ratio, floating alpha,
                                bint positive,
                                floating tol,
                                int max_iter):
    '''
    Perform elastic net regression: for all i in indices,
    find code[i] s.t code[i].dot(G) = Dx[ii], where i = indices[ii].
    G and code are the arrays containing the values for all samples,
    while Dx and X should already be subscripted.

    Parameters
    ----------
    G: array, shape (n_samples x n_components x n_components)
    Dx: array, shape (batch_size x n_components x n_components)
    X: array, shape (batch_size x n_components x n_components
    code: array, shape (n_samples x n_components)
    indices: array, shape (batch_size
    l1_ratio: floating, enet-regression parameter
    alpha: floating, enet-regression paramater
    positive: bint, enet-regression parameter
    '''
    cdef int batch_size = indices.shape[0]
    cdef int i, j, info, ii
    cdef int n_components = G.shape[0]
    cdef int n_features = X.shape[1]
    cdef floating* G_ptr = <floating*> &G[0, 0]
    cdef floating* Dx_ptr = <floating*> &Dx[0, 0]
    cdef POSV posv
    cdef str format
    cdef floating[:] this_code
    cdef floating[::1] this_Dx
    cdef floating[:] this_X
    cdef floating[:, ::1] G_copy
    cdef floating[:, ::1] code_copy

    cdef floating[:] H
    cdef floating[:] XtA

    if floating is float:
        posv = sposv
        format = 'f'
    else:
        posv = dposv
        format = 'd'

    if l1_ratio == 0:
        # Make it thread-safe
        G_copy = view.array((n_components, n_components),
                                              sizeof(floating),
                                              format=format, mode='c')
        G_copy[:, :] = G[:, :]
        G = G_copy
        G_ptr = &G[0, 0]
        for j in range(n_components):
            G[j, j] += alpha

        code_copy = view.array((batch_size, n_components),
                                      sizeof(floating),
                                      format=format, mode='c')
        posv(&UP, &n_components, &batch_size,
        G_ptr,
        &n_components,
        Dx_ptr, &n_components,
        &info)
        for j in range(n_components):
            G[j, j] -= alpha
        for ii in range(batch_size):
            i = indices[ii]
            code[i, :] = Dx[ii, :]
    else:
        H = view.array((n_components, ), sizeof(floating),
                   format=format, mode='c')
        XtA = view.array((n_components, ), sizeof(floating),
                     format=format, mode='c')
        with nogil:
            for ii in range(batch_size):
                i = indices[ii]
                this_Dx = Dx[ii, :]
                this_X = X[ii, :]
                this_code = code[i, :]
                enet_coordinate_descent_gram(
                    this_code,
                    alpha * l1_ratio,
                    alpha * (1 - l1_ratio),
                    G, this_Dx, this_X, H, XtA, max_iter, tol,
                    positive)
    return np.asarray(code)

def _update_G_average(floating[:, :, ::1] G_average,
                              floating[:, ::1] G,
                              floating[:] w_sample):
    cdef int batch_size = w_sample.shape[0]
    cdef int n_components = G_average.shape[1]
    cdef int ii, i, k, p, q
    for ii in range(batch_size):
        for p in range(n_components):
            for q in range(n_components):
                G_average[ii, p, q] *= (1 - w_sample[ii])
                G_average[ii, p, q] += G[p, q] * w_sample[ii]
    return G_average


# Shamelessly copied from sklearn (no .pxd in sources :-( )
cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m

cdef void enet_coordinate_descent_gram(floating[:] w, floating alpha, floating beta,
                                 floating[:, ::1] Q,
                                 floating[::1] q,
                                 floating[:] y,
                                 floating[:] H,
                                 floating[:] XtA,
                                 int max_iter, floating tol, bint positive) nogil:
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # fused types version of BLAS functions
    cdef DOT dot
    cdef AXPY axpy
    cdef ASUM asum

    if floating is float:
        dot = sdot
        axpy = saxpy
        asum = sasum
        gemv = sgemv
    else:
        dot = ddot
        axpy = daxpy
        asum = dasum
        gemv = dgemv

    # get the data information into easy vars
    cdef int n_samples = y.shape[0]
    cdef int n_features = Q.shape[0]


    cdef floating tmp
    cdef floating w_ii, mw_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating q_dot_w
    cdef floating w_norm2
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef int ii
    cdef int n_iter = 0
    cdef int f_iter

    cdef floating* w_ptr = <floating*>&w[0]
    cdef floating* Q_ptr = &Q[0, 0]
    cdef floating* q_ptr = <floating*>&q[0]
    cdef floating* H_ptr = &H[0]
    cdef floating* XtA_ptr = &XtA[0]
    cdef floating one = 1
    cdef floating zero = 0

    cdef floating y_norm2

    y_norm2 = dot(&n_samples, &y[0], &ONE, &y[0], &ONE)

    tol = tol * y_norm2

    # initial value "Q w" which will be kept of up to date in the iterations
    # H = np.dot(Q, w)
    gemv(&NTRANS,
      &n_features, &n_features,
      &one,
      Q_ptr, &n_features,
      w_ptr, &ONE,
      &zero,
      H_ptr, &ONE
      )

    XtA[:] = 0

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            ii = f_iter

            if Q[ii, ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value
            if w_ii != 0.0:
                # H -= w_ii * Q[ii]
                mw_ii = -w[ii]
                axpy(&n_features, &mw_ii, Q_ptr + ii * n_features, &ONE,
                     H_ptr, &ONE)

            tmp = q[ii] - H[ii]

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                    / (Q[ii, ii] + beta)

            if w[ii] != 0.0:
                # H +=  w[ii] * Q[ii] # Update H = X.T X w
                axpy(&n_features, &w[ii], Q_ptr + ii * n_features, &ONE,
                     H_ptr, &ONE)

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
            q_dot_w = dot(&n_features, w_ptr, &ONE, q_ptr, &ONE)

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
            w_norm2 = dot(&n_features, &w[0], &ONE, &w[0], &ONE)

            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            # The call to dasum is equivalent to the L1 norm of w
            gap += (alpha * asum(&n_features, &w[0], &ONE) -
                    const * y_norm2 +  const * q_dot_w +
                    0.5 * beta * (1 + const ** 2) * w_norm2)

            if gap < tol:
                # return if we reached desired tolerance
                break