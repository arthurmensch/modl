# Adapted from lightning

import pickle
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_equal)
from modl.utils.randomkit import RandomState


def test_random():
    rs = RandomState(seed=0)
    vals = [rs.randint(10) for t in range(10000)]
    assert_almost_equal(np.mean(vals), 5.018)
    vals = [rs.binomial(1000, 0.8) for t in range(10000)]
    assert_almost_equal(np.mean(vals), 799.8564)


def test_shuffle():
    ind = np.arange(10)
    rs = RandomState(seed=0)
    rs.shuffle(ind)
    assert_array_equal(ind, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])


def test_shuffle_with_trace():
    ind = np.arange(10)
    ind2 = np.arange(9, -1, -1)
    rs = RandomState(seed=0)
    perm = rs.shuffle_with_trace([ind, ind2])
    assert_array_equal(ind, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])
    assert_array_equal(ind2, [7, 1, 5, 0, 8, 3, 2, 6, 9, 4])
    assert_array_equal(ind, perm)


def test_permutation():
    rs = RandomState(seed=0)
    perm = rs.permutation(10)
    assert_array_equal(perm, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])


def test_random_state_pickle():
    rs = RandomState(seed=0)
    random_integer = rs.randint(5)
    pickle_rs = pickle.dumps(rs)
    pickle_rs = pickle.loads(pickle_rs)
    pickle_random_integer = pickle_rs.randint(5)
    assert_equal(random_integer, pickle_random_integer)
