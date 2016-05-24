# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve

# FIXME: don't depend on scikit-learn.
from sklearn.base import BaseEstimator
from modl.dict_completion import compute_biases

from .matrix_fact_fast import _cd_fit, _predict
from modl.dict_completion import rmse

class ExplicitMF(BaseEstimator):
    def __init__(self, alpha=1.0, beta=0., n_components=30, max_iter=10, tol=1e-3,
                 callback=None, random_state=None, detrend=False,
                 verbose=0):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.random_state = random_state
        self.detrend = detrend
        self.beta = beta
        self.verbose = verbose

    def _init(self, X, rng):
        n_rows, n_cols = X.shape
        P = np.zeros((n_rows, self.n_components), order="C")
        Q = rng.rand(self.n_components, n_cols)
        Q = np.asfortranarray(Q)
        return P, Q

    def fit(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)

        if self.detrend:
            self.row_mean_, self.col_mean_ = compute_biases(X, beta=self.beta)
            for i in range(X.shape[0]):
                X.data[X.indptr[i]:X.indptr[i + 1]] -= self.row_mean_[i]
            X.data -= self.col_mean_.take(X.indices, mode='clip')

        n_rows, n_cols = X.shape
        n_data = len(X.data)

        # Initialization.
        rng = np.random.RandomState(self.random_state)
        self.P_, self.Q_ = self._init(X, rng)

        residuals = np.empty(n_data, dtype=np.float64)
        n_max = max(n_rows, n_cols)
        g = np.empty(n_max, dtype=np.float64)
        h = np.empty(n_max, dtype=np.float64)
        delta = np.empty(n_max, dtype=np.float64)
        if self.callback is not None:
            self.callback(self)
        # Model estimation.
        _cd_fit(self, X.data, X.indices, X.indptr, self.P_, self.Q_, residuals,
                g, h, delta, self.n_components, self.alpha, self.max_iter,
                self.tol, self.callback, self.verbose)

        return self

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_, self.Q_)

        if self.detrend:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


class ImplicitMF(BaseEstimator):
    def __init__(self, alpha=1.0, n_components=30, max_iter=10, tol=1e-3,
                 callback=None, random_state=None, verbose=0):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.random_state = random_state
        self.verbose = verbose

    def _init(self, X, rng):
        n_rows, n_cols = X.shape
        P = rng.rand(n_rows, self.n_components)
        Q = np.zeros((self.n_components, n_cols), order="F")
        return P, Q

    def fit(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)
        self.P_, self.Q_ = self._init(X, rng)

        for it in range(self.max_iter):
            PX = self.P_.T * X  # sparse dot
            PP = np.dot(self.P_.T, self.P_)
            PP.flat[::PP.shape[0] + 1] += self.alpha
            self.Q_ = solve(PP, PX)

            QX = self.Q_ * X.T  # sparse dot
            QQ = np.dot(self.Q_, self.Q_.T)
            QQ.flat[::QQ.shape[0] + 1] += self.alpha
            self.P_ = solve(QQ, QX).T

            if self.callback is not None:
                self.callback(self)

        return self

    def decision_function(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_, self.Q_)
        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def predict(self, X):
        X = self.decision_function(X)
        X.data = (X.data > 0.5).astype(np.int32)
        return X