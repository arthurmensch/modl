from math import sqrt

import pytest
import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from spira.cross_validation import train_test_split

from modl.dict_completion import DictCompleter, csr_center_data

backends = ['c', 'python']


@pytest.mark.parametrize("backend", backends)
def test_dict_completion(backend):
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = DictCompleter(n_components=3, max_n_iter=100, alpha=1e-3,
                       random_state=0,
                       detrend=False,
                       backend=backend,
                       verbose=0, )

    mf.fit(X)

    Y = np.dot(mf.P_, mf.Q_)
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


@pytest.mark.parametrize("backend", backends)
def test_dict_completion_normalise(backend):
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = DictCompleter(n_components=3, max_n_iter=100, alpha=1e-3,
                       random_state=0,
                       backend=backend,
                       verbose=0, detrend=True)

    mf.fit(X)

    Y = np.dot(mf.P_, mf.Q_)
    Y += mf.col_mean_[np.newaxis, :]
    Y += mf.row_mean_[:, np.newaxis]
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


@pytest.mark.parametrize("backend", backends)
def test_dict_completion_missing(backend):
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(100, 4)
    V = rng.rand(4, 20)
    X = np.dot(U, V)
    X = sp.csr_matrix(X)
    X_tr, X_te = train_test_split(X, train_size=0.95)
    X_tr = sp.csr_matrix(X_tr)
    X_te = sp.csr_matrix(X_te)

    mf = DictCompleter(n_components=4, max_n_iter=400, alpha=1,
                       random_state=0,
                       backend=backend,
                       detrend=True,
                       verbose=0, )

    mf.fit(X_tr)
    X_pred = mf.predict(X_te)
    rmse = sqrt(np.sum((X_te.data - X_pred.data) ** 2) / X_te.data.shape[0])
    X_te_c, _, _ = csr_center_data(X_te)
    rmse_c = sqrt(np.sum((X_te.data - X_te_c.data) ** 2) / X_te.data.shape[0])
    assert(rmse < rmse_c)
    # assert_array_almost_equal(X_te.data, X_pred.data)
