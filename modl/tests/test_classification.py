import pytest

from sklearn.utils import check_random_state

from modl.classification import FactoredLogistic
import numpy as np

# def test_projector():
#     rng = check_random_state(0)
#     X = rng.randn(10000, 1000)
#     W = rng.randn(50, 1000)
#     projector = Projector(W, n_jobs=3)
#     Xp = projector.fit_transform(X)

def test_factored_projection():
    rng = check_random_state(0)
    X = rng.randn(1000, 1000)
    y = rng.randint(10, size=1000)
    dataset = rng.randint(2, size=1000)[:, np.newaxis]
    X = np.concatenate([X, dataset], axis=1)
    factored_lr = FactoredLogistic(latent_dim=10, random_state=0,
                                   max_iter=1)
    factored_lr.fit(X, y)

    y_pred = factored_lr.predict(X)
    y_pred = factored_lr.predict_proba(X[:, :-1])
    y_pred = factored_lr.predict_proba(X[:, :-1], dataset=0)
    X[:, -1] = -1
    y_pred = factored_lr.predict(X)
