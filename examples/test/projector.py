from sklearn.utils import check_random_state

from modl.classification import Projector

rng = check_random_state(0)
X = rng.randn(10000, 10000)
W = rng.randn(100, 10000)
projector = Projector(W, n_jobs=3)
Xp = projector.fit_transform(X)