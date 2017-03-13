import pytest

from sklearn.utils import check_random_state

from modl.classification import Projector


def test_projector():
    rng = check_random_state(0)
    X = rng.randn(10000, 1000)
    W = rng.randn(50, 1000)
    projector = Projector(W, n_jobs=3)
    Xp = projector.fit_transform(X)