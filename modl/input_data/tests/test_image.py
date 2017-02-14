import numpy as np
from modl.input_data.image import scale_patches
from modl.input_data.image_fast import clean_mask, fill
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state


def test_scale_patches():
    patch_size = (8, 8, 3)
    n = 100
    shape = (n, ) + patch_size
    rs = check_random_state(0)
    X = rs.randn(*shape)
    Y = scale_patches(X, with_mean=True, with_std=True, channel_wise=True)
    assert_array_almost_equal(Y.mean(axis=(1, 2)), 0)
    assert_array_almost_equal(np.sum(Y ** 2, axis=(1, 2)), 1 / 3)

    scale_patches(X, with_mean=True, with_std=True, channel_wise=True,
                  copy=False)
    assert_array_equal(X, Y)

    X = rs.randn(*shape)

    Y = scale_patches(X, with_mean=False, with_std=True, channel_wise=True)
    assert_array_almost_equal(np.sum(Y ** 2, axis=(1, 2)), 1 / 3)

    Y = scale_patches(X, with_mean=True, with_std=False, channel_wise=True)
    assert_array_almost_equal(Y.mean(axis=(1, 2)), 0)

    Y = scale_patches(X, with_mean=True, with_std=True, channel_wise=False)
    assert_array_almost_equal(Y.mean(axis=(1, 2, 3)), 0)
    assert_array_almost_equal(np.sum(Y ** 2, axis=(1, 2, 3)), 1)


def test_clean():
    A = np.ones((64, 64, 3))
    A[:2, :, :] = -1
    A[-2:, :, :] = -1
    A[:, :2, :] = -1
    A[:, -2:, :] = -1
    patches = extract_patches(A, (8, 8, 3))
    idx = clean_mask(patches, A)
    mask = np.zeros((64, 64, 3))
    mask[2:55, 2:55, 0] = 1
    true_idx = np.c_[np.where(mask)]
    assert_array_almost_equal(idx, true_idx)


def test_fill():
    p, q, r = 10, 10, 10
    assert_array_equal(np.c_[np.where(np.ones((p, q, r)))], fill(p, q, r))
