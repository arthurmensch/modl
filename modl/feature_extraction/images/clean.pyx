# encoding: utf-8
from cython cimport view
import numpy as np
cimport numpy as np

def clean_mask(double[:, :, :, :, :, :] patches,
          double[:, :, :] image):
    cdef int p = patches.shape[0]
    cdef int q = patches.shape[1]
    cdef int r = patches.shape[2]
    cdef int x = patches.shape[3]
    cdef int y = patches.shape[4]
    cdef int z = patches.shape[5]
    cdef int pp, qq, rr, xx, yy, zz
    cdef int n_samples = 0
    cdef int[:, :, :] take = view.array((p, q, r), sizeof(int),
              format='i', mode='c')
    take[:] = 1
    for pp in range(p + x - 1):
        for qq in range(q + y -1):
            for rr in range(r + z - 1):
                if image[pp, qq, rr] == -1:
                    for xx in range(max(0, pp - x + 1), min(p, pp + 1)):
                        for yy in range(max(0, qq - y + 1), min(q, qq + 1)):
                            for zz in range(max(0, rr - y + 1), min(r, rr + 1)):
                                take[xx, yy, zz] = 0
    return np.array(take, dtype=bool)