# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport view
import numpy as np
cimport numpy as np

from cython cimport floating

def clean_mask(floating[:, :, :, :, :, :] patches,
          floating[:, :, :] image):
    """
    Given the patches extracted from image using gen_patches, return the indices
    for which patch are clean (i.e. with non-negative values)
    Parameters
    ----------
    patches: float/double ndarray, shape (*patch_indices, *patch_shape)
        Extracted from sklearn.feature_extraction.image.gen_batches
    image: float/double ndarray, shape (width, height, n_channel)

    Returns
    -------
    indices: int ndarray, shape = (n_good_patches, 3)
        Coordinates of the clean patches
    """
    cdef long p = patches.shape[0]
    cdef long q = patches.shape[1]
    cdef long r = patches.shape[2]
    cdef long x = patches.shape[3]
    cdef long y = patches.shape[4]
    cdef long z = patches.shape[5]
    cdef long size = p * q * r
    cdef long pp, qq, rr, xx, yy, zz
    cdef long l = 0
    cdef long n_samples = 0
    cdef int[:, :, :] take = view.array((p, q, r), sizeof(int), format='i', mode='c')
    cdef long[:, :] indices = view.array((size, 3), sizeof(long), format='l')
    take[:] = 1
    for pp in range(p + x - 1):
        for qq in range(q + y -1):
            for rr in range(r + z - 1):
                if image[pp, qq, rr] == -1:
                    for xx in range(max(0, pp - x + 1), min(p, pp + 1)):
                        for yy in range(max(0, qq - y + 1), min(q, qq + 1)):
                            for zz in range(max(0, rr - y + 1), min(r, rr + 1)):
                                take[xx, yy, zz] = 0
    for pp in range(p):
        for qq in range(q):
            for rr in range(r):
                if take[pp, qq, rr]:
                    indices[l, 0] = pp
                    indices[l, 1] = qq
                    indices[l, 2] = rr
                    l +=1
    return np.asarray(indices[:l])

def fill(long p, long q, long r):
    """
    Faster np.c_[np.where(np.ones(p, q, r))]
    """
    cdef long size = p * q * r
    cdef long pp, qq, rr
    cdef long l = 0
    cdef long[:, :] indices = view.array((size, 3), sizeof(long), format='l')
    for pp in range(p):
        for qq in range(q):
            for rr in range(r):
                indices[l, 0] = pp
                indices[l, 1] = qq
                indices[l, 2] = rr
                l +=1
    return np.asarray(indices)
