# encoding: utf-8
from cython cimport view
import numpy as np
from .._utils.randomkit.random_fast cimport RandomState
from libc.stdio cimport printf
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
    cdef bint[:, :, :] take = view.array((p, q, r), sizeof(int),
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
#
# def index_array(double[:, :, :, :, :, :] patches,
#                 int[:, :, :] take = None,
#                 int max_samples=-1,
#                 int random_seed=0,):
#     cdef int p = patches.shape[0]
#     cdef int q = patches.shape[1]
#     cdef int r = patches.shape[2]
#     cdef int x = patches.shape[3]
#     cdef int y = patches.shape[4]
#     cdef int z = patches.shape[5]
#     cdef int n_samples = 0
#     cdef int i, j, k, u, v, w
#     cdef int[:] mask
#     cdef long[:, :] results
#     cdef int idx = 0
#     cdef int masked_idx = 0
#     cdef RandomState random_state = RandomState(random_seed)
#
#     mask = view.array((n_samples, ), sizeof(int), format='i',
#                                       mode='c')
#     results = view.array((max_samples, 3), sizeof(long),
#                          format='l', mode='c')
#
#     for i in range(n_samples):
#         if i < max_samples:
#             mask[i] = 1
#         else:
#             mask[i] = 0
#     random_state.shuffle(mask)
#
#     for i in range(p):
#         for j in range(q):
#             for k in range(r):
#                 if take[i, j, k]:
#                     if mask[idx]:
#                         # printf("%i\n", masked_idx)
#                         results[masked_idx, 0] = i
#                         results[masked_idx, 1] = j
#                         results[masked_idx, 2] = k
#                         masked_idx += 1
#                     idx += 1
#     return np.asarray(results)