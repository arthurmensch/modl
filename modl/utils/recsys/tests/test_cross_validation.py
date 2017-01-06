import scipy.sparse as sp
from numpy.testing import assert_equal

from modl.utils.recsys.cross_validation import ShuffleSplit


def test_shuffle_split():
    X = [[3, 0, 0, 1],
         [2, 0, 5, 0],
         [0, 4, 3, 0],
         [0, 0, 2, 0]]
    X = sp.coo_matrix(X)

    cv = ShuffleSplit(n_iter=10)
    for X_tr, X_te in cv.split(X):
        assert_equal(X.shape, X_tr.shape)
        assert_equal(X.shape, X_te.shape)
        assert_equal(X.data.shape[0],
                     X_tr.data.shape[0] + X_te.data.shape[0])