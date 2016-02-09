import itertools
from copy import deepcopy
from math import pow, floor, ceil

import numpy as np
from scipy import linalg
from sklearn.utils.enet_proj_fast import enet_projection_inplace
from spira.impl.dict_fact_fast import _online_dl_fast

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array, \
    gen_cycling_subsets
from sklearn.utils.enet_projection import enet_scale, enet_norm


class DictMF(BaseEstimator):
    def __init__(self, alpha=1.0, learning_rate=1.,
                 n_components=30, n_epochs=2,
                 normalize=False,
                 fit_intercept=False,
                 random_state=None, verbose=0,
                 impute=True,
                 batch_size=1,
                 dict_init=None,
                 reduction=1,
                 l1_ratio=1,
                 debug=False):

        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.impute = impute
        self.reduction = reduction
        self.dict_init = dict_init
        self.l1_ratio = l1_ratio
        self.debug = debug

    def _init(self, X, random_state):
        n_rows, n_cols = X.shape

        self.random_state_ = check_random_state(self.random_state)

        # Q dictionary
        if self.dict_init is not None:
            if self.fit_intercept:
                if self.dict_init.shape != (self.n_components - 1, n_cols):
                    raise ValueError('Wrong shape for dict init')
                self.Q_ = np.empty((self.n_components, n_cols), order='F')
                self.Q_[1:] = check_array(self.dict_init, order='F',
                                          dtype='float')
                self.Q_[0] = 1
            else:
                if self.dict_init.shape != (self.n_components, n_cols):
                    raise ValueError('Wrong shape for dict init')
                self.Q_ = check_array(self.dict_init, order='F',
                                      dtype='float')
        else:
            self.Q_ = np.empty((self.n_components, n_cols), order='F')

            if self.fit_intercept:
                # Intercept on first line
                self.Q_[0] = 1
                self.Q_[1:] = self.random_state_.randn(self.n_components - 1,
                                                       n_cols)
            else:
                self.Q_[:] = self.random_state_.randn(self.n_components,
                                                      n_cols)

            S = np.sqrt(np.sum(self.Q_ ** 2, axis=1))
            self.Q_ /= S[:, np.newaxis]

        self.Q_ = enet_scale(self.Q_, l1_ratio=self.l1_ratio, radius=1)

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

        if self.reduction > 1:
            self.subsets_ = gen_cycling_subsets(n_cols,
                                                int(ceil(
                                                        n_cols / self.reduction)),
                                                random=True,
                                                random_state=random_state)
            if self.debug:
                self.subsets_ = [self.subsets_] + [
                    gen_cycling_subsets(n_cols,
                                        int(floor(n_cols / self.reduction)),
                                        random=True,
                                        random_state=rng)
                    for rng in range(10)]
        else:
            self.subsets_ = itertools.repeat(np.arange(n_cols))
        if self.debug:
            self.loss_ = [0]
            self.diff_ = []
        else:
            self.loss_ = None
            self.diff_ = None

    def fit(self, X, y=None):
        self.partial_fit(X)

    def partial_fit(self, X, y=None):
        if not hasattr(self, 'Q_'):
            self._init(X, self.random_state)
        if self.backend == 'c':
            _online_dl(X,
                       float(self.alpha), float(self.learning_rate),
                       self.A_, self.B_,
                       self.counter_,
                       self.G_, self.T_,
                       self.Q_,
                       self.fit_intercept,
                       self.batch_size,
                       self.random_state_,
                       self.verbose,
                       self.impute,
                       1 if self.counter_[0] < 1000 else self.reduction, )
        else:
            _online_dl_slow(X,
                            float(self.alpha), float(self.learning_rate),
                            self.A_, self.B_,
                            self.counter_,
                            self.G_, self.T_,
                            self.Q_,
                            self.fit_intercept,
                            self.batch_size,
                            self.random_state_,
                            self.verbose,
                            self.impute,
                            self.reduction,
                            self.l1_ratio,
                            self.subsets_,
                            self.loss_,
                            self.debug,
                            self.diff_)


def _online_dl(X,
               alpha, learning_rate,
               A, B, counter,
               G, T,
               Q,
               fit_intercept, n_epochs, batch_size, random_state, verbose,
               impute):
    row_nnz = X.getnnz(axis=1)
    n_cols = X.shape[1]
    max_idx_size = min(row_nnz.max() * batch_size, n_cols)
    row_range = row_nnz.nonzero()[0]

    n_rows, n_cols = X.shape

    random_seed = random_state.randint(0, np.iinfo(np.uint32).max)
    if not impute:
        G = np.zeros((0, 0), order='F')
        T = np.zeros((0, 0), order='F')
    _online_dl_fast(X.data, X.indices,
                    X.indptr, n_rows, n_cols,
                    row_range,
                    max_idx_size,
                    alpha, learning_rate,
                    A, B,
                    counter,
                    G, T,
                    Q,
                    n_epochs, batch_size,
                    random_seed,
                    verbose, fit_intercept, impute)


def get_w(w, idx, counter, batch_size, learning_rate):
    idx_len = idx.shape[0]
    count = counter[0]
    w[0] = 1
    for i in range(count + 1, count + 1 + batch_size):
        w[0] *= (1 - pow(i, - learning_rate))
    w[0] = 1 - w[0]

    for jj in range(idx_len):
        j = idx[jj]
        count = counter[j + 1]
        w[jj + 1] = 1
        for i in range(count + 1, count + 1 + batch_size):
            w[jj + 1] *= (1 - pow(i, - learning_rate))
        w[jj + 1] = 1 - w[jj + 1]


def _update_code_slow(X, idx, alpha, learning_rate,
                      A, B, counter, G, T,
                      Q, row_batch,
                      x, Q_idx, H, Qx, P, w,
                      loss, impute):
    n_cols = X.shape[1]
    n_components = Q.shape[0]
    len_idx = idx.shape[0]
    len_batch = row_batch.shape[0]

    Q_idx[:, :len_idx] = Q[:, idx]
    Q_idx = Q_idx[:, :len_idx]
    x[:len_batch, :len_idx] = X[row_batch][:, idx]
    x = x[:len_batch, :len_idx]

    # w[0] = 1 - np.product(1 - np.power(
    #     np.arange(counter[0], counter[0] + len_batch) + 1,
    #     - learning_rate))

    get_w(w, idx, counter, len_batch, learning_rate)

    counter[0] += len_batch
    counter[idx + 1] += len_batch

    if impute:
        T[:, 0] -= T[:, idx + 1].sum(axis=1)
        Qx[:, :len_batch] = T[:, 0][:, np.newaxis]
        Qx[:, :len_batch] += Q_idx.dot(x.T)
        Qx = Qx[:, :len_batch]
        T[:, idx + 1] = Q_idx * x.mean(axis=0)
        T[:, 0] += T[:, idx + 1].sum(axis=1)
        H[:] = G
        H.flat[::n_components + 1] += 2 * alpha
    else:
        Qx[:, :len_batch] = Q_idx.dot(x.T)
        Qx = Qx[:, :len_batch]
        H[:] = Q_idx.dot(Q_idx.T)
        H.flat[::n_components + 1] += 2 * alpha * len_idx / n_cols

    P[:, :len_batch] = linalg.solve(H, Qx, sym_pos=True,
                                    overwrite_a=True, check_finite=False)
    P = P[:, :len_batch]

    A *= 1 - w[0]
    A += P.dot(P.T) * w[0] / len_batch

    B[:, idx] *= 1 - w[1:(len_idx + 1)]
    B[:, idx] += P.dot(x) * w[1:(len_idx + 1)] / len_batch

    return idx


def _update_dict_slow(A, B, G, Q, Q_idx, R, idx, fit_intercept,
                      components_range, norm, buffer, impute,
                      l1_ratio,
                      full_update=False):
    ger, = linalg.get_blas_funcs(('ger',), (A, Q_idx))
    len_idx = idx.shape[0]

    Q_idx[:, :len_idx] = Q[:, idx]
    Q_idx = Q_idx[:, :len_idx]

    if full_update:
        for j in components_range:
            norm[j] = enet_norm(Q[j], l1_ratio)
    else:
        for j in components_range:
            norm[j] = enet_norm(Q_idx[j], l1_ratio)

    if impute and not full_update:
        G -= Q_idx.dot(Q_idx.T)

    R[:, :len_idx] = B[:, idx] - np.dot(Q_idx.T, A).T
    R = R[:, :len_idx]

    # Intercept on first column
    for j in components_range:
        ger(1.0, A[j], Q_idx[j], a=R, overwrite_a=True)
        Q_idx[j] = R[j] / A[j, j]
        if full_update:
            Q[j, idx] = Q_idx[j]
            enet_projection_inplace(Q[j], buffer, norm[j], l1_ratio)
            Q[j] = buffer
            Q_idx[j] = Q[j, idx]
        else:
            enet_projection_inplace(Q_idx[j], buffer[:len_idx],
                                    norm[j], l1_ratio)
            Q_idx[j] = buffer[:len_idx]
        ger(-1.0, A[j], Q_idx[j], a=R, overwrite_a=True)

    Q[:, idx] = Q_idx

    if impute:
        if not full_update:
            G += Q_idx.dot(Q_idx.T)
        else:
            G[:] = Q.dot(Q.T)


def _online_dl_slow(X,
                    alpha, learning_rate,
                    A, B, counter,
                    G, T,
                    Q,
                    fit_intercept, batch_size, random_state, verbose,
                    impute,
                    reduction,
                    l1_ratio,
                    subsets,
                    loss,
                    debug,
                    diff=None):
    n_rows, n_cols = X.shape
    n_components = Q.shape[0]

    if debug:
        max_idx_size = n_cols
    else:
        max_idx_size = int(ceil(n_cols / reduction))
    row_range = np.arange(n_rows)

    x = np.zeros((batch_size, max_idx_size), order='F')
    Q_idx = np.zeros((n_components, max_idx_size), order='F')
    R = np.zeros((n_components, max_idx_size), order='F')
    H = np.zeros((n_components, n_components), order='F')
    Qx = np.zeros((n_components, batch_size), order='F')
    P = np.zeros((n_components, batch_size), order='F')

    w = np.zeros(max_idx_size + 1)
    buffer = np.zeros(n_cols)
    norm = np.zeros(n_components)
    last_call = 0

    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)

    random_state.shuffle(row_range)
    batches = gen_batches(len(row_range), batch_size)

    if debug:
        main_subsets = subsets[0]
    else:
        main_subsets = subsets

    for subset, batch in zip(main_subsets, batches):
        row_batch = row_range[batch]
        if debug:
            Q_old = Q.copy()
        idx = _update_code_slow(X, subset,
                                alpha, learning_rate,
                                A, B, counter, G, T,
                                Q, row_batch,
                                x, Q_idx, H, Qx, P, w,
                                loss,
                                impute)
        random_state.shuffle(components_range)

        _update_dict_slow(A, B, G, Q, Q_idx, R, idx, fit_intercept,
                          components_range, norm, buffer, impute,
                          l1_ratio)

        if debug:

            this_loss = .5 * np.trace(Q.dot(Q.T) * A) - np.trace(Q.dot(B.T))
            this_loss += loss[0]
            loss.append(this_loss)

            Q_full = Q_old.copy()
            A_copy, B_copy, counter_copy, G_copy, T_copy = deepcopy(
                    (A, B, counter, G, T))
            idx = _update_code_slow(X, np.arange(n_cols),
                                    alpha, learning_rate,
                                    A_copy, B_copy, counter_copy, G_copy,
                                    T_copy,
                                    Q_full, row_batch,
                                    x, Q_idx, H, Qx, P, w,
                                    loss,
                                    impute)
            random_state.shuffle(components_range)

            _update_dict_slow(A_copy, B_copy, G_copy, Q_full, Q_idx, R, idx,
                              fit_intercept,
                              components_range, norm, buffer, impute,
                              l1_ratio)

            Q_mean = np.zeros_like(Q_old)
            for i in range(1, 11):
                this_Q = Q_old.copy()
                subset = next(subsets[i])

                idx = _update_code_slow(X, subset,
                                        alpha, learning_rate,
                                        A, B, counter, G, T,
                                        this_Q, row_batch,
                                        x, Q_idx, H, Qx, P, w,
                                        loss,
                                        impute)
                random_state.shuffle(components_range)

                _update_dict_slow(A, B, G, this_Q, Q_idx, R, idx,
                                  fit_intercept,
                                  components_range, norm, buffer, impute,
                                  l1_ratio)
                Q_mean += this_Q
            Q_mean /= 10
            diff.append(np.sum((Q_mean - Q_full) ** 2))
            print(diff[-1])
        if verbose and counter[0] // (n_rows // verbose) == last_call + 1:
            print("Iteration %i" % (counter[0]))
            last_call += 1
