
"""Projection on the elastic-net ball
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
"""
import numpy as np
from sklearn.utils import check_array

from .enet_proj_fast import enet_norm_fast, enet_projection_fast, \
    enet_scale_fast, enet_scale_matrix_fast


def enet_projection(v, radius=1., l1_ratio=1, check_input=False):
    if check_input:
        v = check_array(v, dtype=np.float64, copy=False,
                        ensure_2d=False)
    b = np.zeros_like(v)
    if v.ndim == 1:
        enet_projection_fast(v, b, radius, l1_ratio)
    else:
        for i in range(v.shape[0]):
            enet_projection_fast(v[i], b[i], radius, l1_ratio)
    return b


def enet_norm(v, l1_ratio=1):
    v = check_array(v, dtype=np.float64, order='C', copy=False,
                    ensure_2d=False)
    if v.ndim == 1:
        return enet_norm_fast(v, l1_ratio)
    else:
        m = v.shape[0]
        norms = np.zeros(m, dtype=np.float64)
        for i in range(m):
            norms[i] = enet_norm_fast(v[i], l1_ratio)
    return norms


def enet_scale(v, radius=1, l1_ratio=1):
    v = check_array(v, dtype=np.float64,
                    order='F',
                    ensure_2d=False,
                    copy=True)
    if v.ndim == 1:
        enet_scale_fast(v, l1_ratio=l1_ratio, radius=radius)
        return v
    else:
        enet_scale_matrix_fast(v, l1_ratio=l1_ratio, radius=radius)
        return v


def enet_threshold(v, l1_ratio=1, radius=1, inplace=False):
    if not inplace:
        v = v.copy()
    Sv = np.sqrt(np.sum(v ** 2, axis=1)) / radius
    Sv[Sv == 0] = 1
    v[:] = enet_projection(v / Sv[:, np.newaxis], l1_ratio=l1_ratio,
                           radius=radius)
    Sb = np.sqrt(np.sum(v ** 2, axis=1)) / radius
    Sb[Sb == 0] = 1
    v *= (Sv / Sb)[:, np.newaxis]
    return v


def l2_sphere_projection(v, radius=1, inplace=False):
    if not inplace:
        v = v.copy()
    Sv = np.sqrt(np.sum(v ** 2, axis=1)) / radius
    Sv[Sv == 0] = 1
    v /= Sv[:, np.newaxis]
    return v
