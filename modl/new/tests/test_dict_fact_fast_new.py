import numpy as np
from modl.new.dict_fact_new import DictFactNew
from modl.new.dict_fact_new import Sampler
from numpy.testing import assert_array_equal

def test_sampler():
    s = Sampler(1000, 100, False, 0)
    res = np.empty(1000, dtype='long')
    for i in range(10):
        res[100 * i: 100 * (i + 1)] = s.yield_subset()
    res = np.sort(res)
    assert_array_equal(res, np.arange(1000))

    s = Sampler(1000, 150, False, 0)
    res = np.empty(1000, dtype='long')
    for i in range(6):
        res[150 * i: 150 * (i + 1)] = s.yield_subset()
    i = 6
    subset = s.yield_subset()
    res[150 * i:] = subset[:100]
    res = np.sort(res)
    assert_array_equal(res, np.arange(1000))
    assert(np.unique(subset[100:]).shape[0] == 50)

def test_dict_fact_fast_new():
    # Smoke test
    dict_init = np.random.randn(100, 200)
    dl = DictFactNew(dict_init, n_samples=200)
    X = np.random.randn(100, 200)

    dl.partial_fit(X, np.arange(200))
