from math import sqrt

import numpy as np
import scipy.sparse as sp
from modl.decomposition.recsys import RecsysDictFact, compute_biases
from modl.utils.recsys.cross_validation import train_test_split
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_array


def test_dict_completion():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = RecsysDictFact(n_components=3, n_epochs=1, alpha=1e-3,
                        random_state=0,
                        detrend=False,
                        verbose=0, )

    mf.fit(X)

    Y = np.dot(mf.code_, mf.components_)
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


def test_dict_completion_normalise():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = RecsysDictFact(n_components=3, n_epochs=1, alpha=1e-3,
                        random_state=0,
                        verbose=0, detrend=True)

    mf.fit(X)

    Y = np.dot(mf.code_, mf.components_)
    Y += mf.col_mean_[np.newaxis, :]
    Y += mf.row_mean_[:, np.newaxis]
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


def test_dict_completion_missing():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(100, 4)
    V = rng.rand(4, 20)
    X = np.dot(U, V)
    X = sp.csr_matrix(X)
    X_tr, X_te = train_test_split(X, train_size=0.95)
    X_tr = sp.csr_matrix(X_tr)
    X_te = sp.csr_matrix(X_te)

    mf = RecsysDictFact(n_components=4, n_epochs=1, alpha=1,
                        random_state=0,
                        detrend=True,
                        verbose=0, )

    mf.fit(X_tr)
    X_pred = mf.predict(X_te)
    rmse = sqrt(np.sum((X_te.data - X_pred.data) ** 2) / X_te.data.shape[0])
    X_te_centered = check_array(X_te, accept_sparse='csr', copy=True)
    compute_biases(X_te_centered, inplace=True)
    rmse_c = sqrt(np.sum((X_te.data
                          - X_te_centered.data) ** 2) / X_te.data.shape[0])
    assert (rmse < rmse_c)