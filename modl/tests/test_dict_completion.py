import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from spira.cross_validation import train_test_split

from modl.dict_completion import DictCompleter


def test_dict_completion():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = DictCompleter(n_components=3, max_n_iter=100, alpha=1e-3,
                       random_state=0,
                       normalize=False,
                       verbose=0, )

    mf.fit(X)

    Y = np.dot(mf.P_.T, mf.Q_)
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

    mf = DictCompleter(n_components=3, max_n_iter=100, alpha=1e-3,
                       random_state=0,
                       verbose=0, normalize=True)

    mf.fit(X)

    Y = np.dot(mf.P_.T, mf.Q_)
    Y += mf.col_mean_[np.newaxis, :]
    Y += mf.row_mean_[:, np.newaxis]
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


def test_dict_completion():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 2)
    V = rng.rand(2, 20)
    X = np.dot(U, V)
    X = sp.csr_matrix(X)
    X_tr, X_te = train_test_split(X, train_size=0.9)

    mf = DictCompleter(n_components=3, max_n_iter=100, alpha=1e-3,
                       random_state=0,
                       normalize=False,
                       verbose=0, )

    mf.fit(X_tr)
    X_pred = mf.predict(X_te)
    assert_array_almost_equal(X_te, X_pred)
