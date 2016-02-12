from math import pow, ceil, sqrt

import numpy as np
import scipy.sparse as sp
from numba import autojit
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array
from .enet_proj import enet_scale
from .enet_proj import enet_norm, enet_projection_inplace
from .dict_fact_fast import _online_dl_main_loop_sparse_fast


class DictMFStats:
    def __init__(self, A=None, B=None, counter=None, G=None, T=None, loss=None,
                 loss_indep=0., subset_array=None, subset_start=None,
                 subset_stop=None,
                 n_iter=0,
                 n_verbose_call=0):
        self.loss = loss
        self.loss_indep = loss_indep
        self.subset_stop = subset_stop
        self.subset_start = subset_start
        self.subset_array = subset_array
        self.T = T
        self.G = G
        self.counter = counter
        self.B = B
        self.A = A
        self.n_iter = n_iter
        self.n_verbose_call = n_verbose_call


def _init_stats(Q, impute=False, reduction=1, max_n_iter=0,
                random_state=None):
    random_state = check_random_state(random_state)
    n_components, n_cols = Q.shape

    subset_size = int(ceil(n_cols / reduction))

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

    subset_array = random_state.permutation(n_cols)
    subset_start = 0
    subset_stop = subset_size

    loss = np.empty(max_n_iter)
    loss_indep = 0.

    return DictMFStats(A, B, counter, G, T, loss, loss_indep, subset_array,
                       subset_start,
                       subset_stop, 0, 0)


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
                 impute=False,
                 max_n_iter=50,
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
                                  dtype='float', copy=True)
            if self.fit_intercept:
                if not (np.all(self.Q_[0] == self.Q_[0].mean())):
                    raise ValueError('When fitting intercept and providing '
                                     'initial dictionary, first component of'
                                     ' the dictionary should be '
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
        self.Q_ = np.asfortranarray(
            enet_scale(self.Q_, l1_ratio=self.l1_ratio, radius=1))

        self._stat = _init_stats(self.Q_, impute=self.impute,
                                 max_n_iter=self.max_n_iter,
                                 reduction=self.reduction,
                                 random_state=self.random_state_)

    def _check_init(self):
        return hasattr(self, 'Q_')

    def fit(self, X, y=None):
        while not self._check_init() or self._stat.n_iter < self.max_n_iter:
            self.partial_fit(X)

    def _refit(self, X):
        self.P_ = self.transform(X)

    def transform(self, X, y=None):
        return compute_code(X, self.Q_, self.alpha)

    def partial_fit(self, X, y=None):
        if not self._check_init():
            self._init(X)
        self.P_, self.Q_ = online_dl(X, self.Q_,
                                     P=getattr(self, 'P_', None),
                                     alpha=float(
                                         self.alpha),
                                     l1_ratio=self.l1_ratio,
                                     learning_rate=float(
                                         self.learning_rate),
                                     offset=float(
                                         self.offset),
                                     stat=self._stat,
                                     freeze_first_col=self.fit_intercept,
                                     batch_size=self.batch_size,
                                     random_state=self.random_state_,
                                     verbose=self.verbose,
                                     impute=self.impute,
                                     max_n_iter=self.max_n_iter,
                                     reduction=self.reduction,
                                     debug=self.debug,
                                     callback=self._callback,
                                     backend=self.backend)

    def _callback(self):
        if self.callback is not None:
            self.callback(self)


def compute_code(X, Q, alpha):
    X = check_array(X, accept_sparse='csr', order='F')
    Q = check_array(Q, order='F')
    n_components = Q.shape[0]
    if not sp.isspmatrix_csr(X):
        G = Q.dot(Q.T)
        Qx = Q.dot(X.T)
        G.flat[::n_components + 1] += 2 * alpha
        P = linalg.solve(G, Qx, sym_pos=True,
                         overwrite_a=True, check_finite=False)
        return P
    else:
        row_range = X.getnnz(axis=1).nonzero()[0]
        n_rows, n_cols = X.shape
        P = np.zeros((n_rows, n_components), order='F')
        for j in row_range:
            nnz = X.indptr[j + 1] - X.indptr[j]
            idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
            x = X.data[X.indptr[j]:X.indptr[j + 1]]

            Q_idx = Q[:, idx]
            C = Q_idx.dot(Q_idx.T)
            Qx = Q_idx.dot(x)
            C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
            P[j] = linalg.solve(C, Qx, sym_pos=True,
                                overwrite_a=True, check_finite=False)
        return P


@autojit(nopython=True)
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


def _update_subset_stat(stat, random_state):
    n_cols = stat.subset_array.shape[0]
    subset_size = stat.subset_stop - stat.subset_start
    if stat.subset_stop + subset_size < n_cols:
        stat.subset_start += subset_size
        stat.subset_stop += subset_size
    else:
        buffer_end = stat.subset_array[stat.subset_start:].copy()
        buffer_start = stat.subset_array[:stat.subset_start].copy()
        len_buffer_end = buffer_end.shape[0]
        random_state.shuffle(buffer_start)
        random_state.shuffle(buffer_end)
        stat.subset_array[:len_buffer_end] = buffer_end
        stat.subset_array[len_buffer_end:] = buffer_start
        stat.subset_start = 0
        stat.subset_stop = subset_size


def online_dl(X, Q,
              P=None,
              alpha=1.,
              learning_rate=1.,
              offset=0.,
              batch_size=1,
              reduction=1,
              l1_ratio=1.,
              stat=None,
              impute=False,
              max_n_iter=10000,
              freeze_first_col=False,
              random_state=None,
              verbose=0,
              debug=False,
              callback=None,
              backend='python'):
    n_rows, n_cols = X.shape
    X = check_array(X, accept_sparse='csr', dtype='float', order='F')

    if Q.shape[1] != n_cols:
        Q = check_array(Q, order='F', dtype='float')
        raise ValueError('X and Q shape mismatch: %r != %r' % (n_cols,
                                                               Q.shape[1]))
    if P is not None:
        P = check_array(P, order='C', dtype='float')
        if P.shape != (n_rows, Q.shape[0]):
            raise ValueError('Bad P shape: expected %r, got %r' %
                             ((n_rows, Q.shape[0]), P.shape))

    random_state = check_random_state(random_state)

    if stat is None:
        stat = _init_stats(Q, impute=impute, reduction=reduction,
                           max_n_iter=max_n_iter,
                           random_state=random_state)

    if backend == 'python':
        if sp.isspmatrix_csr(X):
            row_range = X.getnnz(axis=1).nonzero()[0]
        else:
            row_range = np.arange(n_rows)

        random_state.shuffle(row_range)
        batches = gen_batches(len(row_range), batch_size)

        for batch in batches:
            row_batch = row_range[batch]
            if sp.isspmatrix_csr(X):
                for j in row_batch:
                    if stat.n_iter >= max_n_iter:
                        return
                    subset = X.indices[X.indptr[j]:X.indptr[j + 1]]
                    reg = alpha * subset.shape[0] / n_cols
                    this_X = X.data[X.indptr[j]:X.indptr[j + 1]]
                    this_X = this_X[np.newaxis, :]
                    this_P = _update_code_slow(this_X, subset,
                                               reg, learning_rate,
                                               offset,
                                               Q, stat.A, stat.B, stat.counter,
                                               stat.G, stat.T,
                                               impute,
                                               debug)
                    if P is not None:
                        P[j] = this_P
                    stat.n_iter += 1
                dict_subset = np.concatenate([X.indices[
                                              X.indptr[j]:X.indptr[j + 1]]
                                              for j in row_batch])
                dict_subset = np.unique(dict_subset)
            else:
                if stat.n_iter + len(row_batch) - 1 >= max_n_iter:
                    return
                subset = stat.subset_array[stat.subset_start:stat.subset_stop]
                reg = alpha * subset.shape[0] / n_cols
                _update_subset_stat(stat, random_state)
                this_X = X[row_batch][:, subset]
                this_P = _update_code_slow(this_X, subset, reg, learning_rate,
                                           offset,
                                           Q, stat.A, stat.B, stat.counter,
                                           stat.G, stat.T,
                                           impute,
                                           debug)
                dict_subset = subset
                if P is not None:
                    P[row_batch] = this_P.copy()
                stat.n_iter += len(row_batch)
            _update_dict_slow(Q, dict_subset, freeze_first_col,
                              l1_ratio,
                              stat.A, stat.B, stat.G,
                              impute)

            if verbose and stat.n_iter // ceil(
                    int(n_rows / verbose)) == stat.n_verbose_call + 1:
                print("Iteration %i" % stat.n_iter)
                stat.n_verbose_call += 1
                if callback is not None:
                    callback()
    else:
        if sp.isspmatrix_csr(X):
            random_seed = random_state.randint(0, np.iinfo(np.uint32).max)
            _online_dl_main_loop_sparse_fast(X, Q, P if P is not None else
            np.zeros((1, 1), order='F'),
                                             stat,
                                             alpha, learning_rate,
                                             offset,
                                             freeze_first_col,
                                             batch_size,
                                             max_n_iter,
                                             impute,
                                             True,
                                             verbose,
                                             random_seed,
                                             callback)
        else:
            raise NotImplementedError
            _online_dl_dense_main_loop_fast(X, P, Q,

                                            stat)
    return P, Q

def _update_code_slow(X, subset, alpha, learning_rate,
                      offset,
                      Q, A, B, counter, G, T,
                      impute, debug):
    batch_size, n_cols = X.shape
    n_components = Q.shape[0]

    Q_subset = Q[:, subset]

    w_A, w_B = _get_weights(subset, counter, batch_size,
                            learning_rate, offset)

    counter[0] += batch_size
    counter[subset + 1] += batch_size

    if impute:
        T[:, 0] -= T[:, subset + 1].sum(axis=1)
        Qx = T[:, 0][:, np.newaxis]
        Qx += Q_subset.dot(X.T)
        T[:, subset + 1] = Q_subset * X.mean(axis=0)
        T[:, 0] += T[:, subset + 1].sum(axis=1)
        G_temp = G.copy()
    else:
        Qx = Q_subset.dot(X.T)
        G_temp = Q_subset.dot(Q_subset.T)
    G_temp.flat[::n_components + 1] += alpha
    P = linalg.solve(G_temp, Qx, sym_pos=True, overwrite_a=True, check_finite=False)

    if debug:
        dict_loss = .5 * np.sum(Q.dot(Q.T) * A) - np.sum(Q * B)
        loss_indep *= (1 - w_A)
        loss_indep += (.5 * np.sum(X ** 2) +
                            alpha * np.sum(P ** 2)) * w_A / batch_size
        loss[n_iter] = loss_indep + dict_loss

    A *= 1 - w_A
    A += P.dot(P.T) * w_A / batch_size
    B[:, subset] *= 1 - w_B
    B[:, subset] += P.dot(X) * w_B / batch_size
    if batch_size == 1:
        P = P[:, 0]

    return P


@autojit(nopython=True)
def _update_dict_slow(Q, subset,
                      freeze_first_col,
                      l1_ratio,
                      A, B, G, 
                      impute):
    n_components = Q.shape[0]
    Q_subset = Q[:, subset]
    norm = enet_norm(Q_subset, l1_ratio)
    buffer = np.zeros(subset.shape[0])
    if impute:
        G -= Q_subset.dot(Q_subset.T)

    ger, = linalg.get_blas_funcs(('ger',), (A, Q_subset))
    # Intercept on first column
    if freeze_first_col:
        components_range = np.arange(1, n_components)
    else:
        components_range = np.arange(n_components)
    np.random.shuffle(components_range)
    R = B[:, subset] - np.dot(Q_subset.T, A).T
    for j in components_range:
        ger(1.0, A[j], Q_subset[j], a=R, overwrite_a=True)
        Q_subset[j] = R[j] / A[j, j]
        enet_projection_inplace(Q_subset[j], buffer, norm[j], l1_ratio)
        Q_subset[j] = buffer
        ger(-1.0, A[j], Q_subset[j], a=R, overwrite_a=True)

    Q[:, subset] = Q_subset

    if impute:
        G += Q_subset.dot(Q_subset.T)
