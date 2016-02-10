from math import pow, ceil

import numpy as np
import scipy.sparse as sp
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array
from .enet_proj import enet_projection, enet_scale, enet_norm


class DictMF(BaseEstimator):
    def __init__(self, alpha=1.0,
                 n_components=30,
                 # Hyper-parameters
                 learning_rate=1.,
                 batch_size=1,
                 offset=0,
                 reduction=1,
                 n_iter=None,
                 # Preproc parameters
                 fit_intercept=False,
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

        self.fit_intercept = fit_intercept

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.reduction = reduction

        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.dict_init = dict_init
        self.n_components = n_components

        self.impute = impute
        self.max_n_iter = max_n_iter

        self.random_state = random_state
        self.verbose = verbose

        self.backend = backend
        self.debug = debug

        self.callback = callback

    def _init(self, X):
        _, n_cols = X.shape

        self.random_state_ = check_random_state(self.random_state)

        # Q dictionary
        if self.dict_init is not None:
            if self.dict_init.shape != (self.n_components, n_cols):
                raise ValueError(
                    'Initial dictionary and X shape mismatch: %r != %r' % (
                        self.dict_init.shape,
                        (self.n_components, n_cols)))
            self.Q_ = check_array(self.dict_init, order='C',
                                  dtype='float')
            if self.fit_intercept and not (
                    np.all(self.Q_[0] == self.Q_[0].mean())):
                raise ValueError('When fitting intercept and providing '
                                 'initial dictionary, first component of the '
                                 'dictionary should be '
                                 'proportional to [1, ..., 1]')
            self.Q_[0] = 1
        else:
            self.Q_ = np.empty((self.n_components, n_cols), order='C')

            if self.fit_intercept:
                self.Q_[0] = 1
                self.Q_[1:] = self.random_state_.randn(self.n_components - 1,
                                                       n_cols)
            else:
                self.Q_[:] = self.random_state_.randn(self.n_components,
                                                      n_cols)
        # Fix this
        self.Q_ = np.asfortranarray(enet_scale(self.Q_, l1_ratio=self.l1_ratio, radius=1))

        self._stat = _init_sufficient_stat(self.Q_, impute=self.impute)

        if sp.isspmatrix_csr(X):
            self._subset_stat = _init_subset_stat(n_cols, self.reduction,
                                                  self.random_state_)
        else:
            self._subset_stat = None
        if self.debug:
            self._loss_stat = _init_loss_stat()
        else:
            self._loss_stat = None

        self.n_iter_ = 0
        self.num_verbose_call_ = 0

    def _check_init(self):
        return hasattr(self, 'Q_')

    def fit(self, X, y=None):
        while not hasattr(self, 'n_iter_') or (self.n_iter_ < self.max_n_iter):
            self.partial_fit(X)

    def partial_fit(self, X, y=None):
        if not self._check_init():
            self._init(X)
        self.n_iter_, self.num_verbose_call_ = online_dl(X, self.Q_,
                  alpha=float(self.alpha),
                  l1_ratio=self.l1_ratio,
                  learning_rate=float(self.learning_rate),
                  offset=float(self.offset),
                  stat=self._stat,
                  freeze_first_col=self.fit_intercept,
                  batch_size=self.batch_size,
                  random_state=self.random_state_,
                  verbose=self.verbose,
                  impute=self.impute,
                  max_n_iter=self.max_n_iter,
                  n_iter=self.n_iter_,
                  num_verbose_call=self.num_verbose_call_,
                  reduction=self.reduction,
                  subset_stat=self._subset_stat,
                  debug=self.debug,
                  loss_stat=self._loss_stat,
                  callback=self._callback)

    def _callback(self):
        if self.callback is not None:
            self.callback(self)


def _init_sufficient_stat(Q, impute=False):
    n_components, n_cols = Q.shape
    A = np.zeros((n_components, n_components),
                 order='F')
    B = np.zeros((n_components, n_cols), order="F")

    counter = np.zeros(n_cols + 1, dtype='int')

    if not impute:
        G = None
        T = None
    else:
        G = Q.dot(Q.T).T
        T = np.zeros((n_components, n_cols + 1), order="F")
    return A, B, counter, G, T


def _init_subset_stat(n_cols, reduction, random_state):
    subset_stat_array = random_state.permutation(n_cols)
    subset_size = ceil(int(n_cols / reduction))
    subset_stat_slice = slice(0, subset_size)
    return subset_stat_array, subset_stat_slice


def _init_loss_stat():
    return [0, []]


def _get_weights(idx, counter, batch_size, learning_rate, offset):
    idx_len = idx.shape[0]
    count = counter[0]
    w_A = 1
    for i in range(count + 1, count + 1 + batch_size):
        w_A *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w_A = 1 - w_A
    w_B = np.zeros(idx_len)
    for jj in range(idx_len):
        j = idx[jj]
        count = counter[j + 1]
        w_B[jj] = 1
        for i in range(count + 1, count + 1 + batch_size):
            w_B[jj] *= (1 - pow((1 + offset) / (offset + i), learning_rate))
        w_B[jj] = 1 - w_B[jj]
    return w_A, w_B


def _update_subset_stat(subset_stat, random_state):
    n_cols = subset_stat[0].shape[0]
    subset_size = subset_stat[1].start - subset_stat[0].end
    if subset_stat[0].end + subset_size <= n_cols:
        subset_stat[1].end += subset_size
        subset_stat[1].start += subset_size
    else:
        buffer_end = subset_stat[0][subset_stat[1].start:].copy()
        buffer_start = subset_stat[0][:subset_stat[1].start].copy()
        len_buffer_end = buffer_end.shape[0]
        random_state.shuffle(buffer_start)
        random_state.shuffle(buffer_end)
        subset_stat[:len_buffer_end] = buffer_end
        subset_stat[len_buffer_end:] = buffer_start
        subset_stat[1].start = 0
        subset_stat[1].end = subset_size


def online_dl(X, Q,
              P=None,
              alpha=1.,
              learning_rate=1.,
              offset=0.,
              batch_size=1,
              reduction=1,
              l1_ratio=1.,
              stat=None,
              subset_stat=None,
              impute=False,
              n_iter=0,
              num_verbose_call=0,
              max_n_iter=10000,
              freeze_first_col=False,
              random_state=None,
              verbose=0,
              debug=False,
              loss_stat=None,
              callback=None):
    n_rows, n_cols = X.shape
    X = check_array(X, accept_sparse='csr', dtype='float')
    if sp.isspmatrix_csr(X):
        X = check_array(X, accept_sparse='csr', order='F')
    Q = check_array(Q, order='F', dtype='float')

    if Q.shape[1] != n_cols:
        raise ValueError('X and Q shape mismatch: %r != %r' % (n_cols,
                                                               Q.shape[1]))
    if P is not None:
        P = check_array(P, order='F', dtype='float')
        if P.shape != (n_rows, Q.shape[0]):
            raise ValueError('Bad P shape: expected %r, got %r' %
                             ((n_rows, Q.shape[1]), P.shape))

    random_state = check_random_state(random_state)

    if stat is None:
        stat = _init_sufficient_stat(Q, impute=impute)
    A, B, counter, G, T = stat
    if sp.isspmatrix_csr(X):
        row_nnz = X.getnnz(axis=1)
        row_range = row_nnz.nonzero()[0]
    else:
        if subset_stat is None:
            subset_stat = _init_subset_stat(n_cols, reduction, random_state)
        row_range = np.arange(n_rows)

    if debug:
        if loss_stat is None:
            loss_stat = _init_loss_stat()

    random_state.shuffle(row_range)
    batches = gen_batches(len(row_range), batch_size)
    for batch in batches:
        row_batch = row_range[batch]
        if sp.isspmatrix_csr(X):
            assert(isinstance(X, sp.csr_matrix))
            dict_subset = np.array([], dtype='int')
            for i in row_batch:
                subset = X.indices[X.indptr[i]:X.indptr[i + 1]]
                reg = alpha * subset.shape[0] / n_cols
                this_X = X.data[X.indptr[i]:X.indptr[i + 1]]
                this_X = this_X[np.newaxis, :]
                this_P = _update_code_slow(this_X, subset,
                                           reg, learning_rate,
                                           offset,
                                           Q, A, B, counter, G, T,
                                           impute)
                if P is not None:
                    P[i] = this_P[:, 0]
                dict_subset = np.union1d(dict_subset, subset)
        else:
            subset = subset_stat[0][subset_stat[1]]
            reg = alpha * subset.shape[0] / n_cols
            _update_subset_stat(subset_stat, random_state)
            this_X = X[row_batch]
            _update_code_slow(this_X, subset, alpha, learning_rate,
                              offset,
                              Q, A, B, counter, G, T,
                              impute)
            dict_subset = subset
            if P is not None:
                P[row_batch] = this_P
        _update_dict_slow(Q, dict_subset, freeze_first_col,
                          l1_ratio,
                          A, B, G,
                          impute,
                          random_state)
        # print((Q ** 2).sum(axis=1))
        if debug:
            dict_loss = .5 * np.trace(Q.dot(Q.T) * A) - np.trace(Q.dot(B.T))
            loss_stat[0] += .5 * np.sum(this_X ** 2)
            loss_stat[0] += alpha * np.sum(this_P ** 2)
            loss_stat[1].append(loss_stat[0] + dict_loss)
        n_iter += len(row_batch)
        if verbose and n_iter // ceil(
                int(n_rows / verbose)) == num_verbose_call + 1:
            print("Iteration %i" % n_iter)
            num_verbose_call += 1
            if callback is not None:
                callback()
        if n_iter > max_n_iter:
            return n_iter, num_verbose_call
    return n_iter, num_verbose_call


def _update_code_slow(X, subset, alpha, learning_rate,
                      offset,
                      Q, A, B, counter, G, T,
                      impute):
    batch_size, n_cols = X.shape
    n_components = Q.shape[0]

    Q_subset = Q[:, subset]
    w_A, w_B = _get_weights(subset, counter, batch_size, learning_rate, offset)

    counter[0] += batch_size
    counter[subset + 1] += batch_size

    if impute:
        T[:, 0] -= T[:, subset + 1].sum(axis=1)
        Qx = T[:, 0][:, np.newaxis]
        Qx += Q_subset.dot(X.T)
        T[:, subset + 1] = Q_subset * X.mean(axis=0)
        T[:, 0] += T[:, subset + 1].sum(axis=1)
        G.flat[::n_components + 1] += 2 * alpha
    else:
        Qx = Q_subset.dot(X.T)
        G = Q_subset.dot(Q_subset.T)
        G.flat[::n_components + 1] += 2 * alpha

    P = linalg.solve(G, Qx, sym_pos=True, overwrite_a=True, check_finite=False)

    A *= 1 - w_A
    A += P.dot(P.T) * w_A / batch_size

    B[:, subset] *= 1 - w_B
    B[:, subset] += P.dot(X) * w_B / batch_size
    if Q_subset.ndim == 1:
        P = P[:, 0]
    return P


def _update_dict_slow(Q, subset,
                      freeze_first_col,
                      l1_ratio,
                      A, B, G,
                      impute,
                      random_state):
    n_components = Q.shape[0]
    Q_idx = Q[:, subset]
    norm = enet_norm(Q_idx, l1_ratio)
    if impute:
        G -= Q_idx.dot(Q_idx.T)

    R = B[:, subset] - np.dot(Q_idx.T, A).T

    ger, = linalg.get_blas_funcs(('ger',), (A, Q_idx))
    # Intercept on first column
    if freeze_first_col:
        components_range = np.arange(1, n_components)
    else:
        components_range = np.arange(n_components)
    random_state.shuffle(components_range)
    for j in components_range:
        ger(1.0, A[j], Q_idx[j], a=R, overwrite_a=True)
        Q_idx[j] = R[j] / A[j, j]
        Q_idx[j] = enet_projection(Q_idx[j], norm[j], l1_ratio)
        ger(-1.0, A[j], Q_idx[j], a=R, overwrite_a=True)

    Q[:, subset] = Q_idx

    if impute:
        G += Q_idx.dot(Q_idx.T)
