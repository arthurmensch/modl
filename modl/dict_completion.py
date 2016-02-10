import numpy as np
import scipy.sparse as sp
from scipy import linalg
from spira.impl.matrix_fact_fast import _predict

from modl.dict_fact import DictMF, online_dl
from spira.metrics import rmse


class DictCompleter(DictMF):
    def __init__(self, alpha=1.0,
                 n_components=30,
                 # Hyper-parameters
                 learning_rate=1.,
                 batch_size=1,
                 offset=0,
                 reduction=1,
                 n_iter=None,
                 # Preproc parameters
                 fit_intercept=True,
                 normalize=True,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 impute=True,
                 max_n_iter=10000,
                 # Generic parameters
                 random_state=None,
                 verbose=0,
                 backend='python',
                 debug=False,
                 callback=None):
        DictMF.__init__(self, alpha,
                        n_components,
                        # Hyper-parameters
                        learning_rate,
                        batch_size,
                        offset,
                        reduction,
                        n_iter,
                        # Preproc parameters
                        fit_intercept,
                        # Dict parameter
                        dict_init,
                        l1_ratio,
                        impute,
                        max_n_iter,
                        # Generic parameters
                        random_state,
                        verbose,
                        backend,
                        debug,
                        callback)
        self.normalize = normalize

    def fit(self, X, y=None):
        X = sp.csr_matrix(X, dtype='float')
        n_rows = X.shape[0]
        self.P_ = np.zeros((n_rows, self.n_components), order='F',
                           dtype='float')

        if self.normalize:
            X_c, self.row_mean_, self.col_mean_ = csr_center_data(X)

        DictMF.fit(self, X_c)

    def partial_fit(self, X, y=None):
        # Overriding to keep P in memory
        if not self._check_init():
            self._init(X)
        (self.n_iter_,
         self.num_verbose_call_) = online_dl(
            X, self.Q_, self.P_, alpha=float(self.alpha),
            l1_ratio=self.l1_ratio,
            learning_rate=float(self.learning_rate), offset=float(self.offset),
            stat=self._stat, freeze_first_col=self.fit_intercept,
            batch_size=self.batch_size, random_state=self.random_state_,
            verbose=self.verbose, impute=self.impute,
            max_n_iter=self.max_n_iter, n_iter=self.n_iter_,
            num_verbose_call=self.num_verbose_call_,
            reduction=self.reduction,
            subset_stat=self._subset_stat,
            debug=self.debug,
            loss_stat=self._loss_stat,
            callback=self._callback)

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_.T,
                 self.Q_)

        if self.normalize:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')
        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


def _online_refit_slow(X, alpha, P, Q, verbose):
    row_range = X.getnnz(axis=1).nonzero()[0]
    n_cols = X.shape[1]
    n_components = P.shape[0]

    if verbose:
        print('Refitting code')

    for j in row_range:
        nnz = X.indptr[j + 1] - X.indptr[j]
        idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
        x = X.data[X.indptr[j]:X.indptr[j + 1]]

        Q_idx = Q[:, idx]
        C = Q_idx.dot(Q_idx.T)
        Qx = Q_idx.dot(x)
        C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        P[:, j] = linalg.solve(C, Qx, sym_pos=True,
                               overwrite_a=True, check_finite=False)

def csr_center_data(X, inplace=False):
    if not inplace:
        X = X.copy()

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(10):
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return X, acc_u, acc_m
