# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""Projection on the elastic-net ball (Cython version)
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
"""
from libc.math cimport sqrt, fabs

from cython cimport floating

cdef inline floating positive(floating a) nogil:
    if a > 0:
        return a
    else:
        return 0


cdef inline floating sign(floating a) nogil:
    if a >= 0.:
        return 1.
    else:
        return -1.


cdef inline void swap(floating[:] b, unsigned int i, unsigned int j,
                      floating * buf) nogil:
    buf[0] = b[i]
    b[i] = b[j]
    b[j] = buf[0]
    return


cpdef void enet_projection(floating[:] v, floating[:] out, floating radius,
                             floating l1_ratio) nogil:
    cdef unsigned int m = v.shape[0]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int size_U
    cdef unsigned int start_U
    cdef floating buf = 0
    cdef floating gamma
    cdef unsigned int pivot
    cdef unsigned int rho
    cdef unsigned int drho
    cdef floating s
    cdef floating ds
    cdef floating a
    cdef floating d
    cdef floating c
    cdef floating l
    cdef floating norm = 0
    if radius == 0:
        out[:] = 0
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
            out[i] = v[i] / norm
    else:
        # Scaling by 1 / l1_ratio
        gamma = 2 / l1_ratio - 2
        radius /= l1_ratio
        # Preparing data
        for j in range(m):
            out[j] = fabs(v[j])
            norm += out[j] * (1 + gamma / 2 * out[j])
        if norm <= radius:
            out[:] = v[:]
        else:
            # s and rho computation
            s = 0
            rho = 0
            start_U = 0
            size_U = m
            while size_U > 0:
                pivot = start_U + size_U / 2
                # Putting pivot at the beginning
                swap(out, pivot, start_U, &buf)
                pivot = start_U
                drho = 1
                ds = out[pivot] * (1 + gamma / 2 * out[pivot])
                # Ordering : [pivot, >=, <], using Lobato quicksort
                for i in range(start_U + 1, start_U + size_U):
                    if out[i] >= out[pivot]:
                        ds += out[i] * (1 + gamma / 2 * out[i])
                        swap(out, i, start_U + drho, &buf)
                        drho += 1
                if s + ds - (rho + drho) * (1 + gamma / 2 * out[pivot])\
                        * out[pivot] < radius * (1 + gamma * out[pivot]) ** 2:
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
                out[i] = sign(v[i]) * positive(fabs(v[i]) - l) / (1 + l * gamma)
    return


cpdef floating enet_norm(floating[:] v, floating l1_ratio) nogil:
    """Returns the elastic net norm of a vector

    Parameters
    -----------------------------------------
    v: floating memory-view,
        Vector

    l1_gamma: float,
        Ratio of l1 norm (between 0 and 1)

    Returns
    ------------------------------------------
    norm: float,
        Elastic-net norm
    """
    cdef int n = v.shape[0]
    cdef floating res = 0
    cdef floating v_abs
    cdef unsigned int i
    for i in range(n):
        v_abs = fabs(v[i])
        res += v_abs * (l1_ratio + (1 - l1_ratio) * v_abs)
    return res

cpdef void enet_scale(floating[:] X,
                           floating l1_ratio, floating radius=1) nogil:
    cdef int n_features = X.shape[0]
    cdef floating l1_norm = 0
    cdef floating l2_norm = 0
    cdef floating S = 0

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