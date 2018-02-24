from modl.utils.randomkit.sampler import Sampler
import numpy as np
from numpy.testing import assert_array_equal, assert_equal


def test_sampler():
    sampler = Sampler(100, rand_size=True,
                      replacement=True,
                      random_seed=0)
    A = sampler.yield_subset(10)
    assert_array_equal(A, np.array([14, 58, 11, 49, 36, 62, 87,
                                   45, 72, 47, 48, 13, 98, 97, 25, 93]))
    a = np.mean(np.array([sampler.yield_subset(10).shape[0]
                          for t in range(100)]))
    assert_equal(a, 10.19)

    # Without replacement, with fixed size
    sampler = Sampler(100, rand_size=False,
                      replacement=False,
                      random_seed=0)
    A = np.concatenate([sampler.yield_subset(10) for t in range(10)])
    assert_array_equal(np.sort(A), np.arange(100))

    # With replacement, with random size
    sampler = Sampler(100, rand_size=False,
                      replacement=True,
                      random_seed=0)
    A = sampler.yield_subset(10)
    assert_array_equal(A, np.array([6, 55,  1, 25, 87, 49, 69, 63, 13,  8]))
    a = np.mean(np.array([sampler.yield_subset(10).shape[0]
                          for t in range(100)]))
    assert_equal(a, 10)
    A = []
    for t in range(100):
        A.append(sampler.yield_subset(10))
    A = np.concatenate(A)
    m = np.mean(np.bincount(A))
    assert_equal(m, 10)

    # Without replacement, with random size
    sampler = Sampler(100, rand_size=True,
                      replacement=False,
                      random_seed=0)
    A = np.concatenate([sampler.yield_subset(10) for t in range(20)])
    assert_array_equal(np.sort(A[:100]), np.arange(100))
