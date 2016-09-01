import numpy as np
from modl.dict_fact_fast import DictFactImpl
from numpy.testing import assert_array_equal


def test_dict_fact_fast_new():
    # Smoke test
    dict_init = np.random.randn(200, 100).T
    dl = DictFactImpl(dict_init, n_samples=200)
    X = np.random.randn(100, 200)

    dl.partial_fit(X, np.arange(200))