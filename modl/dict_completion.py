import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_array
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
                 detrend=True,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 impute=False,
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
        self.detrend = detrend

    def fit(self, X, y=None):
        X = sp.csr_matrix(X, dtype='float')

        if self.detrend:
            X, self.row_mean_, self.col_mean_ = csr_center_data(X,
                                                                inplace=False)
        n_rows = X.shape[0]
        self.P_ = np.zeros((n_rows, self.n_components), order='C',
                           dtype='float')

        DictMF.fit(self, X)

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_,
                 self.Q_)

        if self.detrend:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')

        out[out > 5] = 5
        out[out < 1]

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


def csr_center_data(X, inplace=False):
    if not inplace:
        X = X.copy()
    X = sp.csr_matrix(X)

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(4):
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return X, acc_u, acc_m
