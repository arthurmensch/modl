from math import pow

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from spira.impl.dict_fact_fast import _update_code_full_fast, _online_dl_fast
from spira.impl.matrix_fact_fast import _predict

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array
from spira.metrics import rmse


class DictMF(BaseEstimator):
    def __init__(self, alpha=1.0, learning_rate=1.,
                 offset=0,
                 n_components=30, n_epochs=2,
                 normalize=False,
                 fit_intercept=False,
                 callback=None, random_state=None, verbose=0,
                 impute=False,
                 batch_size=1,
                 dict_init=None,
                 partial=False,
                 backend='c'):
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.callback = callback
        self.learning_rate = learning_rate
        self.offset = offset
        self.alpha = alpha
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.impute = impute
        self.dict_init = dict_init
        self.partial = partial
        self.backend = backend

    def _init(self, X, random_state):
        n_rows, n_cols = X.shape

        # P code
        self.P_ = np.zeros((self.n_components, n_rows), order='F')

        # Q dictionary
        if self.dict_init is not None:
            if self.fit_intercept:
                raise NotImplementedError('Dict init and fit intercept are'
                                          'not yet compatible')
            if self.dict_init.shape != (self.n_components, n_cols):
                raise ValueError('Wrong shape for dict init')

            self.Q_ = check_array(self.dict_init, order='F',
                                  dtype='float')
        else:
            self.Q_ = np.empty((self.n_components, n_cols), order='F')

            if self.fit_intercept:
                # Intercept on first line
                self.Q_[0] = 1
                self.Q_[1:] = random_state.randn(self.n_components - 1, n_cols)
            else:
                self.Q_[:] = random_state.randn(self.n_components, n_cols)
        self.Q_mult_ = np.zeros(self.n_components)
        S = np.sqrt(np.sum(self.Q_ ** 2, axis=1))
        self.Q_ /= S[:, np.newaxis]

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.counter_ = np.zeros(n_cols + 1, dtype='int')

        if self.impute:
            self.G_ = self.Q_.dot(self.Q_.T).T
            self.T_ = np.zeros((self.n_components, n_cols + 1), order="F")
        else:
            self.G_ = None
            self.T_ = None

    def _refit_code(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        row_range = X.getnnz(axis=1).nonzero()[0]
        if self.backend == 'c':
            _online_refit(X, self.alpha, self.P_, self.Q_, self.Q_mult_,
                          row_range,
                          self.verbose)
        else:
            _online_refit_slow(X, self.alpha, self.P_, self.Q_, self.verbose)

    def fit(self, X, y=None):
        X = sp.csr_matrix(X, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self._init(X, random_state)

        if self.normalize:
            (X, self.row_mean_, self.col_mean_) = csr_center_data(X)
        self._callback()
        if self.backend == 'c':
            _online_dl(X,
                       float(self.alpha), float(self.learning_rate),
                       float(self.offset),
                       self.A_, self.B_,
                       self.counter_,
                       self.G_, self.T_,
                       self.P_, self.Q_,
                       self.Q_mult_,
                       self.fit_intercept,
                       self.n_epochs,
                       self.batch_size,
                       random_state,
                       self.verbose,
                       self.impute,
                       self.partial,
                       True,
                       self._callback,
                       )
        else:
            _online_dl_slow(X,
                            float(self.alpha), float(self.learning_rate),
                            self.A_, self.B_,
                            self.counter_,
                            self.G_, self.T_,
                            self.P_, self.Q_,
                            self.fit_intercept,
                            self.n_epochs,
                            self.batch_size,
                            random_state,
                            self.verbose,
                            self.impute,
                            self._callback)
        self._callback()

    def _callback(self):
        if self.callback is not None:
            self.callback(self)
            return self.callback.rmse[-1]
        else:
            return -1

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        mult = self.Q_mult_[:, np.newaxis] if False else np.exp(
            self.Q_mult_[:, np.newaxis])
        _predict(out, X.indices, X.indptr, self.P_.T,
                 self.Q_ * mult)

        if self.normalize:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')
        out[out > 5] = 5
        out[out < 1] = 1
        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


def _online_refit(X, alpha, P, Q, Q_mult, row_range, verbose):
    n_rows, n_cols = X.shape
    n_components = P.shape[0]

    row_nnz = X.getnnz(axis=1)
    max_idx_size = row_nnz.max()

    Q_idx = np.zeros((n_components, max_idx_size), order='F')
    C = np.zeros((n_components, n_components), order='F')
    if verbose:
        print('Refitting code')
    _update_code_full_fast(X.data, X.indices, X.indptr, n_rows, n_cols,
                           row_range,
                           alpha, P, Q, Q_mult, Q_idx, C, True)


def _online_dl(X,
               alpha, learning_rate,
               offset,
               A, B, counter,
               G, T,
               P, Q,
               Q_mult,
               fit_intercept, n_epochs, batch_size, random_state, verbose,
               impute,
               partial,
               mult_exp,
               callback):
    row_nnz = X.getnnz(axis=1)
    n_cols = X.shape[1]
    max_idx_size = min(row_nnz.max() * batch_size, n_cols)
    row_range = row_nnz.nonzero()[0]

    n_rows, n_cols = X.shape

    random_seed = random_state.randint(0, np.iinfo(np.uint32).max)
    if not impute:
        G = np.zeros((1, 1), order='F')
        T = np.zeros((1, 1), order='F')
    _online_dl_fast(X.data, X.indices,
                    X.indptr, n_rows, n_cols,
                    row_range,
                    max_idx_size,
                    alpha, learning_rate,
                    offset,
                    A, B,
                    counter,
                    G, T,
                    P, Q,
                    Q_mult,
                    n_epochs, batch_size,
                    random_seed,
                    verbose, fit_intercept, partial, impute, mult_exp, callback)


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


def _update_code_slow(X, alpha, learning_rate,
                      A, B, G, T, counter,
                      P, Q, row_batch, impute=False):
    len_batch = len(row_batch)
    n_cols = X.shape[1]
    n_components = P.shape[0]
    for j in row_batch:
        idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
        x = X.data[X.indptr[j]:X.indptr[j + 1]]
        nnz = X.indptr[j + 1] - X.indptr[j]

        if impute:
            T[:, 0] -= T[:, idx + 1].sum(axis=1)
            T[:, idx + 1] = Q[:, idx] * x
            red_Qx = T[:, idx + 1].sum(axis=1)
            T[:, 0] += red_Qx

            v = 0  # nnz / n_cols
            Qx = v * red_Qx + (1 - v) * T[:, 0]
            G.flat[::n_components + 1] += 2 * alpha
        else:
            Qx = Q[:, idx].dot(x)
            G = Q[:, idx].dot(Q[:, idx].T)
            G.flat[::n_components + 1] += 2 * alpha * nnz / n_cols

        P[:, j] = linalg.solve(G,
                               Qx, sym_pos=True,
                               overwrite_a=True, check_finite=False)

        if impute:
            G.flat[::n_components + 1] -= 2 * alpha

        counter[0] += 1
        counter[idx + 1] += 1
        w_A = pow(1 / counter[0], learning_rate)
        A *= 1 - w_A
        A += np.outer(P[:, j], P[:, j]) * w_A
        w_B = np.power(1 / counter[idx + 1], learning_rate)

        B[:, idx] *= 1 - w_B
        B[:, idx] += np.outer(P[:, j], x) * w_B

    X_indices = np.concatenate([X.indices[X.indptr[j]:X.indptr[j + 1]]
                                for j in row_batch])

    if len_batch > 1:
        idx = np.unique(X_indices)
    else:
        idx = X_indices

    return idx  # , x


def _update_dict_slow(X, A, B, G, Q, Q_idx, idx, fit_intercept,
                      components_range, norm, impute=True):
    Q_idx = Q[:, idx]

    if impute:
        old_sub_G = Q_idx.dot(Q_idx.T)

    ger, = linalg.get_blas_funcs(('ger',), (A, Q_idx))
    R = B[:, idx] - np.dot(Q_idx.T, A).T

    # norm = np.sqrt(np.sum(Q_idx ** 2, axis=1))
    norm = np.sqrt(np.sum(Q ** 2, axis=1))
    print('Old norm : %.8f' % norm[1])
    # Intercept on first column
    for j in components_range:
        ger(1.0, A[j], Q_idx[j], a=R, overwrite_a=True)
        Q_idx[j] = R[j] / A[j, j]
        # new_norm = np.sqrt(np.sum(Q_idx[j] ** 2))
        # if new_norm > norm[j]:
        #     Q_idx[j] /= new_norm / norm[j]
        Q[j, idx] = Q_idx[j]
        new_norm = np.sqrt(np.sum(Q[j] ** 2))
        if j == 1:
            print('New norm : %.8f' % new_norm)
        if new_norm > 1:
            Q_idx[j] /= new_norm
            Q[j] /= new_norm

        ger(-1.0, A[j], Q_idx[j], a=R, overwrite_a=True)

    Q[:, idx] = Q_idx

    if impute:
        G += Q_idx.dot(Q_idx.T) - old_sub_G


def _online_dl_slow(X,
                    alpha, learning_rate,
                    A, B, counter,
                    G, T,
                    P, Q,
                    fit_intercept, n_epochs, batch_size, random_state, verbose,
                    impute,
                    callback):
    row_nnz = X.getnnz(axis=1)
    max_idx_size = row_nnz.max() * batch_size
    row_range = row_nnz.nonzero()[0]

    n_rows, n_cols = X.shape
    n_components = P.shape[0]
    Q_idx = np.zeros((n_components, max_idx_size), order='F')

    last_call = 0

    norm = np.zeros(n_components)

    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)

    for e in range(n_epochs):
        random_state.shuffle(row_range)
        batches = gen_batches(len(row_range), batch_size)
        for batch in batches:
            row_batch = row_range[batch]
            idx = _update_code_slow(X, alpha, learning_rate,
                                    A, B, G, T, counter,
                                    P, Q, row_batch,
                                    impute=impute)
            random_state.shuffle(components_range)

            _update_dict_slow(X, A, B, G, Q, Q_idx, idx, fit_intercept,
                              components_range, norm, impute=impute)

            # assert_array_almost_equal(Q.dot(Q.T), G)
            if verbose and counter[0] // (n_rows // verbose) == last_call + 1:
                print("Iteration %i" % (counter[0]))
                last_call += 1
                callback()


def csr_center_data(X, inplace=False):
    if not inplace:
        X = X.copy()

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(2):
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return X, acc_u, acc_m
