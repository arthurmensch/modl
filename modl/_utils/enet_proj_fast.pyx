# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""Projection on the elastic-net ball (Cython version)
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
"""


import cython

from libc.math cimport sqrt, fabs

from .enet_proj_fast cimport UINT32_t

from cython cimport view

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# The following two functions are shamelessly copied from the tree code.
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t randint(UINT32_t end,
                              UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline double positive(double a) nogil:
    if a > 0:
        return a
    else:
        return 0


cdef inline double sign(double a) nogil:
    if a >= 0.:
        return 1.
    else:
        return -1.


cdef inline void swap(double[:] b, unsigned int i, unsigned int j,
                      double * buf) nogil:
    buf[0] = b[i]
    b[i] = b[j]
    b[j] = buf[0]
    return


cpdef void enet_projection_fast(double[:] v, double[:] b, double radius,
                             double l1_ratio) nogil:
    cdef unsigned int m = v.shape[0]
    cdef UINT32_t random_state = 0
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int size_U
    cdef unsigned int start_U
    cdef double buf = 0
    cdef double gamma
    cdef unsigned int pivot
    cdef unsigned int rho
    cdef unsigned int drho
    cdef double s
    cdef double ds
    cdef double a
    cdef double d
    cdef double c
    cdef double l
    cdef double norm = 0
    if radius == 0:
        b[:] = 0
        return

    # L2 projection
    if l1_ratio == 0:
        for i in range(m):
            norm += v[i] ** 2
        if norm <= radius:
            norm = 1
        else:
            norm = sqrt(norm / radius)
        for i in range(m):
            b[i] = v[i] / norm
    else:
        # Scaling by 1 / l1_ratio
        gamma = 2 / l1_ratio - 2
        radius /= l1_ratio
        # Preparing data
        for j in range(m):
            b[j] = fabs(v[j])
            norm += b[j] * (1 + gamma / 2 * b[j])
        if norm <= radius:
            b[:] = v[:]
        else:
            # s and rho computation
            s = 0
            rho = 0
            start_U = 0
            size_U = m
            while size_U > 0:
                pivot = randint(size_U, &random_state) + start_U
                # Putting pivot at the beginning
                swap(b, pivot, start_U, &buf)
                pivot = start_U
                drho = 1
                ds = b[pivot] * (1 + gamma / 2 * b[pivot])
                # Ordering : [pivot, >=, <], using Lobato quicksort
                for i in range(start_U + 1, start_U + size_U):
                    if b[i] >= b[pivot]:
                        ds += b[i] * (1 + gamma / 2 * b[i])
                        swap(b, i, start_U + drho, &buf)
                        drho += 1
                if s + ds - (rho + drho) * (1 + gamma / 2 * b[pivot])\
                        * b[pivot] < radius * (1 + gamma * b[pivot]) ** 2:
                    # U <- L : [<]
                    start_U += drho
                    size_U -= drho
                    rho += drho
                    s += ds
                else:
                    # U <- G \ k : [>=]
                    start_U += 1
                    size_U = drho - 1

            # Projection
            if gamma != 0:
                a = gamma ** 2 * radius + gamma * rho * 0.5
                d = 2 * radius * gamma + rho
                c = radius - s
                l = (-d + sqrt(d ** 2 - 4 * a * c)) / (2 * a)
            else:
                l = (s - radius) / rho
            for i in range(m):
                b[i] = sign(v[i]) * positive(fabs(v[i]) - l) / (1 + l * gamma)
    return


cpdef double enet_norm_fast(double[:] v, double l1_ratio) nogil:
    """Returns the elastic net norm of a vector

    Parameters
    -----------------------------------------
    v: double memory-view,
        Vector

    l1_gamma: float,
        Ratio of l1 norm (between 0 and 1)

    Returns
    ------------------------------------------
    norm: float,
        Elastic-net norm
    """
    cdef int n = v.shape[0]
    cdef double res = 0
    cdef double v_abs
    cdef unsigned int i
    for i in range(n):
        v_abs = fabs(v[i])
        res += v_abs * (l1_ratio + (1 - l1_ratio) * v_abs)
    return res

cpdef void enet_scale_fast(double[:] X,
                           double l1_ratio, double radius=1) nogil:
    cdef int n_features = X.shape[0]
    cdef double l1_norm = 0
    cdef double l2_norm = 0
    cdef double S = 0

    for j in range(n_features):
        l1_norm += fabs(X[j])
        l2_norm += X[j] ** 2
    l1_norm *= l1_ratio
    l2_norm *= (1 - l1_ratio)
    if l2_norm != 0:
        S = (- l1_norm + sqrt(l1_norm ** 2
                              + 4 * radius * l2_norm)) / (2 * l2_norm)
    elif l1_norm != 0:
        S = radius / l1_norm
    for j in range(n_features):
        X[j] *= S

cpdef void enet_scale_matrix_fast(double[:, :] X,
                           double l1_ratio, double radius=1) nogil:
    cdef int n_vectors = X.shape[0]
    cdef int i
    for i in range(n_vectors):
        enet_scale_fast(X[i], l1_ratio, radius=radius)