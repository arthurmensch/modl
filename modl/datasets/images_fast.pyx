# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from cython cimport view
import numpy as np
from .._utils.randomkit.random_fast cimport RandomState
from libc.stdio cimport printf
cimport numpy as np

def index_array(double[:, :, :, :, :, :] patches,
                int max_samples=-1,
                int random_seed=0,
                bint clean=False):
    cdef int p = patches.shape[0]
    cdef int q = patches.shape[1]
    cdef int r = patches.shape[2]
    cdef int x = patches.shape[3]
    cdef int y = patches.shape[4]
    cdef int z = patches.shape[5]
    cdef int n_samples = 0
    cdef int i, j, k, u, v, w
    cdef int[:] mask
    cdef long[:, :] results
    cdef int[:, :, :] take
    cdef int idx = 0
    cdef int masked_idx = 0
    cdef RandomState random_state = RandomState(random_seed)

    take = view.array((p, q, r), sizeof(int),
              format='i', mode='c')
    take[:] = 1
    if clean:
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    for u in range(x):
                        for v in range(y):
                            for w in range(z):
                                if patches[i, j, k, u, v, w] == -1:
                                    take[i, j, k] = 0
                                    break
                            if not take[i, j, k]:
                                break
                        if not take[i, j, k]:
                                break
                    if take[i, j, k]:
                        n_samples += 1
    else:
        n_samples = p * q * r
    printf('%i\n', n_samples)

    if max_samples <= 0:
        max_samples = n_samples

    mask = view.array((n_samples, ), sizeof(int), format='i',
                                      mode='c')
    results = view.array((max_samples, 3), sizeof(long),
                         format='l', mode='c')
    for i in range(n_samples):
        if i < max_samples:
            mask[i] = 1
        else:
            mask[i] = 0
    random_state.shuffle(mask)

    for i in range(p):
        for j in range(q):
            for k in range(r):
                if take[i, j, k]:
                    if mask[idx]:
                        # printf("%i\n", masked_idx)
                        results[masked_idx, 0] = i
                        results[masked_idx, 1] = j
                        results[masked_idx, 2] = k
                        masked_idx += 1
                    idx += 1
    return np.asarray(results)