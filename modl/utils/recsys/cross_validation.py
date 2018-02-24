# Author: Mathieu Blondel
# From spira
# License: BSD

import numpy as np
import scipy.sparse as sp

class ShuffleSplit(object):

    def __init__(self, n_iter=5, train_size=0.75, random_state=None):
        self.n_iter = n_iter
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X):
        X = sp.coo_matrix(X)
        rng = np.random.RandomState(self.random_state)
        shape = X.shape
        n_data = len(X.data)
        n_train = int(self.train_size * n_data)

        for it in range(self.n_iter):
            ind = rng.permutation(n_data)
            train_ind = ind[:n_train]
            test_ind = ind[n_train:]
            X_tr = sp.coo_matrix((X.data[train_ind],
                                  (X.row[train_ind], X.col[train_ind])),
                                 shape=shape)
            X_te = sp.coo_matrix((X.data[test_ind],
                                  (X.row[test_ind], X.col[test_ind])),
                                 shape=shape)
            yield X_tr, X_te

    def __len__(self):
        return self.n_iter


def train_test_split(X, train_size=0.75, random_state=None):
    cv = ShuffleSplit(n_iter=1, train_size=train_size,
                      random_state=random_state)
    return next(cv.split(X))


def cross_val_score(estimator, X, cv):
    scores = []
    for X_tr, X_te in cv.split(X):
        estimator.fit(X_tr)
        scores.append(estimator.score(X_te))
    return np.array(scores)