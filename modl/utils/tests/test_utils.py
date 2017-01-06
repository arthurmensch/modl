from numpy.testing import assert_array_equal

from modl.utils import get_sub_slice

import numpy as np


def test_get_subslice():
    a = slice(10, 100)
    b = slice(20, 30)
    assert_array_equal(get_sub_slice(a, b), np.arange(30, 40))
    assert_array_equal(get_sub_slice(None, b), np.arange(20, 30))
    a = np.arange(10, 100)
    assert_array_equal(get_sub_slice(a, b), np.arange(30, 40))