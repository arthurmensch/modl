import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from modl.externals.spira.matrix_fact import ExplicitMF


def test_matrix_fact_cd():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = ExplicitMF(n_components=3, max_iter=10, alpha=1e-3, random_state=0,
                    verbose=0)

    mf.fit(X)

    Y = np.dot(mf.P_, mf.Q_)
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)
