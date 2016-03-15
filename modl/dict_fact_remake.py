"""
Author: Arthur Mensch (2016)
Dictionary learning with masked data
"""
from math import pow, ceil

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array

from modl._utils.enet_proj import enet_projection, enet_scale, enet_norm
from .dict_fact_fast import _update_dict, _update_code, \
    _update_code_sparse_batch


class DictMFStats:
    """ Data structure holding all the necessary variables to perform online
    dictionary learning with masks.
    """

    def __init__(self, A, B, counter, sample_counter,
                 G, beta, E, E_mult, F, reg, weights, P, multiplier, loss,
                 loss_indep, subset_array, subset_start,
                 subset_stop,
                 n_iter,
                 ):
        self.loss = loss
        self.loss_indep = loss_indep
        self.subset_stop = subset_stop
        self.subset_start = subset_start
        self.subset_array = subset_array
        self.beta = beta
        self.G = G
        self.counter = counter
        self.sample_counter = sample_counter
        self.B = B
        self.A = A
        self.E = E
        self.E_mult = E_mult
        self.F = F
        self.P = P
        self.multiplier = multiplier
        self.reg = reg
        self.weights = weights
        self.n_iter = n_iter

    def reset(self):
        for elem in ['A', 'B', 'beta', 'E', 'E_mult', 'F', 'reg', 'weights']:
            setattr(self, elem, getattr(self, elem) * self.multiplier)
        self.multiplier = 1

def _init_stats(Q,
                n_rows=None,
                impute=False, reduction=1, max_n_iter=0,
                random_state=None):
    """

    Parameters
    ----------
    Q: ndarray (n_component, n_cols)
        Initial dictionary
    impute: boolean
        Initialize variables to perform online update of G and Qx
    reduction: float
        Reduction factor
    max_n_iter:
        Max number of iteration (useful for debugging)
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    stat: DictMFStats
        Data structure used by dictionary learning algorithm
    """
    random_state = check_random_state(random_state)
    n_components, n_cols = Q.shape

    subset_size = int(ceil(n_cols / reduction))

    A = np.zeros((n_components, n_components),
                 order='F')
    B = np.zeros((n_components, n_cols), order="F")

    counter = np.zeros(n_cols + 1, dtype='int')

    if not impute:
        sample_counter = np.zeros(1, dtype='int')
        G = np.zeros((1, 1), order='F')
        T = np.zeros((1, 1), order='F')
    else:
        sample_counter = np.zeros(n_rows, dtype='int')
        G = Q.dot(Q.T).T
        T = np.zeros((n_rows, n_components), order="F")

    subset_array = random_state.permutation(n_cols).astype('i4')
    subset_start = 0
    subset_stop = subset_size

    loss = np.empty(max_n_iter)
    loss_indep = 0.

    weights = np.zeros(n_rows)
    reg = np.zeros(n_rows)
    E = np.zeros((n_components, n_cols))
    E_mult = 0
    F = np.zeros(n_components)
    P = np.zeros((n_rows, n_components))

    return DictMFStats(A, B, counter, sample_counter, G, T, E, E_mult, F, reg, weights,
                       P, 1,
                       loss, loss_indep, subset_array,
                       subset_start,
                       subset_stop, 0, )


class DictMFRemake(BaseEstimator):
    """Matrix factorization estimator based on masked online dictionary
     learning.

    Parameters
    ----------
    alpha: float,
        Regularization of the code (ridge penalty)
    n_components: int,
        Number of components for the dictionary
    learning_rate: float in [0.5, 1],
        Controls the sequence of weights in
         the update of the surrogate function
    batch_size: int,
        Number of samples to consider between each dictionary update
    offset: float,
        Offset in the
    reduction: float,
        Sets how much the data is masked during the algorithm
    fit_intercept: boolean,
        Fixes the first dictionary atom to [1, .., 1]
    dict_init: ndarray (n_components, n_cols),
        Initial dictionary
    l1_ratio: float in [0, 1]:
        Controls the sparsity of the dictionary
    impute: boolean,
        Updates the Gram matrix online (Experimental, non tested)
    max_n_iter: int,
        Number of samples to visit before stopping. If None, fit performs
         a single epoch on data
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.
    verbose: boolean,
        Degree of output the procedure will print.
    backend: str in {'c', 'python'},
        'c' is fastter, but 'python' is easier to hack
    debug: boolean,
        Keep tracks of the surrogate loss during the procedure
    callback: callable,
        Function to be called when printing information

    Attributes
    -------
        self.Q_: ndarray (n_components, n_cols):
            Learned dictionary
    """

    def __init__(self, alpha=1.0,
                 n_components=30,
                 # Hyper-parameters
                 learning_rate=1.,
                 batch_size=1,
                 offset=0,
                 reduction=1,
                 full_projection=False,
                 # Preproc parameters
                 fit_intercept=False,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 impute=False,
                 impute_lr=-1,
                 n_samples=None,
                 max_n_iter=0,
                 # Generic parameters
                 random_state=None,
                 verbose=0,
                 backend='c',
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
        self.impute_lr = impute_lr
        self.n_samples = n_samples
        self.max_n_iter = max_n_iter

        self.random_state = random_state
        self.verbose = verbose

        self.backend = backend
        self.debug = debug

        self.callback = callback

        self.full_projection = full_projection

    def _init(self, X):
        """Initialize statistic and dictionary"""
        n_rows, n_cols = X.shape

        if self.n_samples is not None:
            n_rows = self.n_samples

        self._random_state = check_random_state(self.random_state)

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
                self.Q_[1:] = self._random_state.randn(self.n_components - 1,
                                                       n_cols)
            else:
                self.Q_[:] = self._random_state.randn(self.n_components,
                                                      n_cols)
        # Fix this
        self.Q_ = np.asfortranarray(
            enet_scale(self.Q_, l1_ratio=self.l1_ratio, radius=1))

        self._stat = _init_stats(self.Q_,
                                 n_rows=n_rows,
                                 impute=self.impute,
                                 max_n_iter=self.max_n_iter,
                                 reduction=self.reduction,
                                 random_state=self._random_state)

    def _check_init(self):
        return hasattr(self, 'Q_')

    @property
    def components_(self):
        return self.Q_

    def fit(self, X, y=None):
        """Use X to learn a dictionary Q_. The algorithm cycles on X
        until it reaches the max number of iteration

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the dictionary from
        """
        if self.max_n_iter > 0:
            while not self._check_init() or \
                            self._stat.n_iter < self.max_n_iter:
                self.partial_fit(X)
        else:
            self.partial_fit(X)

    def _refit(self, X):
        """Use X and Q to learn a code P"""
        self.P_ = self.transform(X)

    def transform(self, X, y=None):
        """Computes the loadings to reconstruct dataset X
        from the dictionary Q

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the code from

        Returns
        -------
        code: ndarray(n_samples, n_components)
            Code obtained projecting X on the dictionary
        """
        return compute_code(X, self.Q_, self.alpha)

    def partial_fit(self, X, y=None, sample_idx=None):
        """Stream data X to update the estimator dictionary

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the code from

        """
        if not self._check_init():
            self._init(X)
        self.P_, self.Q_ = online_dl(X, self.Q_,
                                     sample_idx=sample_idx,
                                     P=getattr(self, 'P_', None),
                                     alpha=float(
                                         self.alpha),
                                     l1_ratio=self.l1_ratio,
                                     learning_rate=float(
                                         self.learning_rate),
                                     offset=float(
                                         self.offset),
                                     full_projection=self.full_projection,
                                     stat=self._stat,
                                     freeze_first_col=self.fit_intercept,
                                     batch_size=self.batch_size,
                                     random_state=self._random_state,
                                     verbose=self.verbose,
                                     impute=self.impute,
                                     impute_lr=self.impute_lr,
                                     max_n_iter=self.max_n_iter,
                                     reduction=self.reduction,
                                     debug=self.debug,
                                     callback=self._callback,
                                     backend=self.backend)

    def _callback(self):
        if self.callback is not None:
            self.callback(self)


def compute_code(X, Q, alpha):
    """Ridge with missing value.

    Useful to relearn code from a given dictionary

    Parameters
    ----------
    X: ndarray( n_samples, n_features)
        Data matrix
    Q: ndarray (n_components, n_features)
        Dictionary
    alpha: float,
        Regularization parameter

    Returns
    -------
    P: ndarray (n_components, n_samples)
        Code for each of X sample
    """
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


def _get_weights(idx, counter, batch_size, learning_rate, offset,
                 learning_scheme='new'):
    """Utility function to get the update weights at a given iteration
    """
    if learning_scheme == 'deprecated':
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
                w_B[jj] *= (
                    1 - pow((1 + offset) / (offset + i), learning_rate))
            w_B[jj] = 1 - w_B[jj]
        return w_A, w_B
    else:
        idx_len = idx.shape[0]
        full_count = counter[0]
        w_A = 1
        for i in range(full_count + 1, full_count + 1 + batch_size):
            w_A *= (1 - pow((1 + offset) / (offset + i), learning_rate))
        w_A = 1 - w_A
        w_B = np.zeros(idx_len)
        for jj in range(idx_len):
            j = idx[jj]
            count = counter[j + 1]
            w_B[jj] = 1
            for i in range(1, batch_size + 1):
                w_B[jj] *= (1 - (full_count + i) / (count + i) *
                            pow((1 + offset) / (offset + full_count + i),
                                learning_rate))
            w_B[jj] = 1 - w_B[jj]
        return w_A, w_B


def online_dl(X, Q,
              sample_idx=None,
              P=None,
              alpha=1.,
              learning_rate=1.,
              offset=0.,
              batch_size=1,
              reduction=1,
              l1_ratio=1.,
              full_projection=False,
              stat=None,
              impute=False,
              impute_lr=1,
              max_n_iter=0,
              freeze_first_col=False,
              random_state=None,
              verbose=0,
              debug=False,
              callback=None,
              backend='c'):
    """Matrix factorization estimation based on masked online dictionary
     learning.

    Parameters
    ----------
    alpha: float,
        Regularization of the code (ridge penalty)
    learning_rate: float in [0.5, 1],
        Controls the sequence of weights in
         the update of the surrogate function
    batch_size: int,
        Number of samples to consider between each dictionary update
    offset: float,
        Offset in the sequence of weights in
         the update of the surrogate function
    reduction: float,
        Sets how much the data is masked during the algorithm
    freeze_first_col: boolean,
        Fixes the first dictionary atom
    Q: ndarray (n_components, n_features),
        Initial dictionary
    P: ndarray (n_components, n_samples), optional
        Array where the rolling code is kept (for matrix completion)
    l1_ratio: float in [0, 1]:
        Controls the sparsity of the dictionary
    impute: boolean,
        Updates the Gram matrix online (Experimental, non tested)
    max_n_iter: int,
        Number of samples to visit before stopping. If None, fit performs
         a single epoch on data
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.
    verbose: boolean,
        Degree of output the procedure will print.
    backend: str in {'c', 'python'},
        'c' is faster, but 'python' is easier to hack
    debug: boolean,
        Keep tracks of the surrogate loss during the procedure
    callback: callable,
        Function to be called when printing information
    """
    n_rows, n_cols = X.shape
    n_components = Q.shape[0]
    X = check_array(X, accept_sparse='csr', dtype='float', order='F')

    if sample_idx is None:
        sample_idx = np.arange(n_rows)

    if Q.shape[1] != n_cols:
        Q = check_array(Q, order='F', dtype='float')
        raise ValueError('X and Q shape mismatch: %r != %r' % (n_cols,
                                                               Q.shape[1]))
    if P is not None:
        P = check_array(P, order='C', dtype='float')
        if P.shape != (n_rows, Q.shape[0]):
            raise ValueError('Bad P shape: expected %r, got %r' %
                             ((n_rows, Q.shape[0]), P.shape))

    if debug and backend == 'c':
        raise NotImplementedError("Recording objective loss is only available"
                                  "with backend == 'python'")
    random_state = check_random_state(random_state)

    if stat is None:
        stat = _init_stats(Q, n_rows=n_rows,
                           impute=impute, reduction=reduction,
                           max_n_iter=max_n_iter,
                           random_state=random_state)

    old_n_iter = stat.n_iter
    n_verbose_call = 0

    if sp.isspmatrix_csr(X):
        row_range = X.getnnz(axis=1).nonzero()[0]
        max_subset_size = min(n_cols, batch_size * X.getnnz(axis=1).max())
    else:
        row_range = np.arange(n_rows)
        max_subset_size = stat.subset_stop - stat.subset_start

    random_state.shuffle(row_range)
    batches = gen_batches(len(row_range), batch_size)

    if backend == 'c':
        R = np.empty((n_components, n_cols), order='F')
        Q_subset = np.empty((n_components, max_subset_size), order='F')
        norm = np.zeros(n_components)
        if full_projection:
            buffer = np.zeros(n_cols)
        else:
            buffer = np.zeros(max_subset_size)

        G_temp = np.empty((n_components, n_components), order='F')
        if sp.isspmatrix_csr(X):
            P_temp = np.empty((n_components, batch_size), order='F')
        else:
            P_temp = np.empty((n_components, batch_size), order='F')
        if freeze_first_col:
            components_range = np.arange(1, n_components)
        else:
            components_range = np.arange(n_components)
        weights = np.zeros(max_subset_size + 1)
        subset_mask = np.zeros(n_cols, dtype='i1')
        dict_subset_temp = np.zeros(max_subset_size, dtype='i4')
        dict_subset_lim = np.zeros(1, dtype='i4')
        this_X = np.zeros((1, max_subset_size), order='F')
        P_dummy = np.zeros((1, 1), order='C')

    for batch in batches:
        row_batch = row_range[batch]
        if 0 < max_n_iter <= stat.n_iter + len(row_batch) - 1:
            # Stop algorithm
            stat.n_iter = max_n_iter
            return P, Q
        if sp.isspmatrix_csr(X):
            if backend == 'c':
                stat.n_iter = _update_code_sparse_batch(X.data, X.indices,
                                                        X.indptr,
                                                        n_rows,
                                                        n_cols,
                                                        row_batch,
                                                        sample_idx,
                                                        alpha,
                                                        learning_rate,
                                                        offset,
                                                        Q,
                                                        P if P is not None else
                                                        P_dummy,
                                                        stat.A,
                                                        stat.B,
                                                        stat.counter,
                                                        stat.sample_counter,
                                                        stat.G,
                                                        stat.beta,
                                                        impute,
                                                        impute_lr,
                                                        Q_subset,
                                                        P_temp,
                                                        G_temp,
                                                        this_X,
                                                        subset_mask,
                                                        dict_subset_temp,
                                                        dict_subset_lim,
                                                        weights,
                                                        stat.n_iter,
                                                        max_n_iter,
                                                        P is not None
                                                        )
                # This is hackish, but np.where becomes a
                # bottleneck for low batch size otherwise
                dict_subset = dict_subset_temp[:dict_subset_lim[0]]
            else:
                for j in row_batch:
                    if 0 < max_n_iter <= stat.n_iter:
                        return P, Q
                    subset = X.indices[X.indptr[j]:X.indptr[j + 1]]
                    if impute:
                        reg = alpha
                    else:
                        reg = alpha * subset.shape[0] / n_cols
                    this_X = np.empty((1, subset.shape[0]), order='F')
                    this_X[:] = X.data[X.indptr[j]:X.indptr[j + 1]]
                    this_P = _update_code_slow(this_X, subset,
                                               sample_idx[j:(j + 1)],
                                               reg, learning_rate,
                                               offset,
                                               Q, stat,
                                               impute,
                                               full_projection,
                                               debug)
                    if P is not None:
                        P[j] = this_P
                    stat.n_iter += 1
                dict_subset = np.concatenate([X.indices[
                                              X.indptr[j]:X.indptr[j + 1]]
                                              for j in row_batch])
                dict_subset = np.unique(dict_subset)
        else:  # X is a dense matrix : we force masks
            subset = stat.subset_array[stat.subset_start:stat.subset_stop]
            if impute:
                reg = alpha
            else:
                reg = alpha * subset.shape[0] / n_cols
            this_X = X[row_batch][:, subset]
            if backend == 'python':
                this_P = _update_code_slow(this_X,
                                           subset,
                                           sample_idx[row_batch],
                                           reg, learning_rate,
                                           offset,
                                           Q, stat,
                                           impute,
                                           full_projection,
                                           debug)
            else:
                _update_code(this_X, subset, sample_idx[row_batch],
                             reg, learning_rate,
                             offset, Q, stat.A, stat.B,
                             stat.counter,
                             stat.sample_counter,
                             stat.G,
                             stat.beta,
                             impute,
                             impute_lr,
                             Q_subset,
                             P_temp,
                             G_temp,
                             subset_mask,
                             weights)
                this_P = P_temp.T
            dict_subset = subset
            if P is not None:
                P[row_batch] = this_P[:len(row_batch)]
            stat.n_iter += len(row_batch)
            _update_subset_stat(stat, random_state)
        if stat.multiplier < 1e-50:
            stat.reset()
        # Dictionary update
        if backend == 'python':
            _update_dict_slow(Q, dict_subset, freeze_first_col,
                              l1_ratio,
                              full_projection,
                              stat,
                              impute,
                              random_state)
        else:
            random_state.shuffle(components_range)
            _update_dict(Q, dict_subset, freeze_first_col,
                         l1_ratio,
                         full_projection,
                         stat.A,
                         stat.B,
                         stat.G,
                         impute,
                         R,
                         Q_subset,
                         norm,
                         buffer,
                         components_range)
        if verbose and (stat.n_iter - old_n_iter) // ceil(
                int(n_rows / verbose)) == n_verbose_call:
            print("Iteration %i" % stat.n_iter)
            n_verbose_call += 1
            if callback is not None:
                callback()
    return P, Q


def _update_subset_stat(stat, random_state):
    """Utility function to track forced masks, using a permutation array
    with rolling limits

    Parameters
    ----------
    stat: DictMFStats,
        stat holding the subset array and limits
    random_state:

    """
    # n_cols = stat.subset_array.shape[0]
    # subset_size = stat.subset_stop - stat.subset_start
    # if stat.subset_stop + subset_size < n_cols:
    #     stat.subset_start += subset_size
    #     stat.subset_stop += subset_size
    # else:
    #     buffer_end = stat.subset_array[stat.subset_start:].copy()
    #     buffer_start = stat.subset_array[:stat.subset_start].copy()
    #     len_buffer_end = buffer_end.shape[0]
    #     random_state.shuffle(buffer_start)
    #     random_state.shuffle(buffer_end)
    #     stat.subset_array[:len_buffer_end] = buffer_end
    #     stat.subset_array[len_buffer_end:] = buffer_start
    #     stat.subset_start = 0
    #     stat.subset_stop = subset_size
    random_state.shuffle(stat.subset_array)


def _update_code_slow(X, subset, sample_idx, alpha, learning_rate,
                      offset,
                      Q, stat,
                      impute, full_projection, debug):
    """Compute code for a mini-batch and update algorithm statistics accordingly

    Parameters
    ----------
    X: ndarray, (batch_size, len_subset)
        Mini-batch of masked data to perform the update from
    subset: ndarray (len_subset),
        Mask used on X
    alpha: float,
        Regularization of the code (ridge penalty)
    learning_rate: float in [0.5, 1],
        Controls the sequence of weights in
         the update of the surrogate function
    offset: float,
        Offset in the sequence of weights in
         the update of the surrogate function
    Q: ndarray (n_components, n_features):
        Dictionary to perform ridge regression
    stat: DictMFStats,
        Statistics kept by the algorithm, to be updated by the function
    impute: boolean,
        Online update of the Gram matrix (Experimental)
    debug: boolean,
        Keeps track of the surrogate loss function
    Returns
    -------
    P: ndarray,
        Code for the mini-batch X
    """
    batch_size, _ = X.shape
    n_components, n_cols = Q.shape

    Q_subset = Q[:, subset]

    stat.counter[0] += batch_size
    stat.counter[subset + 1] += batch_size

    X /= stat.counter[subset + 1] / stat.counter[0]

    Qx = np.dot(Q_subset, X.T)

    w = pow((1. + offset) / (offset + stat.counter[0]),
              learning_rate)

    if impute:
        G = stat.G.copy()
    else:
        G = np.dot(Q_subset, Q_subset.T).T
    if impute:
        stat.multiplier *= 1 - w
        w_norm = w / stat.multiplier
        reg_strength = np.sum(stat.P[sample_idx] ** 2, axis=1)
        inv_reg_strength = np.where(reg_strength, 1. / reg_strength, 0)
        if full_projection:
            stat.E += w_norm / batch_size * Q * np.sum(reg_strength)
        else:
            stat.E_mult += w_norm / batch_size * np.sum(reg_strength)

        stat.F += w_norm / batch_size * np.sum(reg_strength)

        stat.reg[sample_idx] += w_norm * (alpha + .5 * np.sum(X ** 2, axis=1) * inv_reg_strength)
        stat.weights[sample_idx] += w_norm

        stat.beta[sample_idx] += w_norm * (Qx.T + stat.P[sample_idx] * (np.sum(X ** 2, axis=1) * inv_reg_strength)[:, np.newaxis])
        this_beta = stat.beta[sample_idx].copy()

    for ii, i in enumerate(sample_idx):
        reg = stat.reg[i] / stat.weights[i]
        G.flat[::n_components + 1] += reg
        P = linalg.solve(G, this_beta[ii] * stat.multiplier, sym_pos=True, overwrite_a=True,
                         check_finite=False)
        G.flat[::n_components + 1] -= reg
        P /= stat.weights[i] * stat.multiplier
        stat.P[i] = P.T

    this_P = stat.P[sample_idx].T

    stat.A += this_P.dot(this_P.T) * w_norm / batch_size
    stat.B[:, subset] += this_P.dot(X) * w_norm / batch_size

    if debug:
        dict_loss = .5 * np.sum(Q.dot(Q.T) * stat.A) - np.sum(Q * stat.B)
        stat.loss_indep *= (1 - w)
        stat.loss_indep += (.5 * np.sum(X ** 2) +
                            alpha * np.sum(P ** 2)) * w
        stat.loss[stat.n_iter] = stat.loss_indep + dict_loss

    return P.T


def _update_dict_slow(Q, subset,
                      freeze_first_col,
                      l1_ratio,
                      full_projection,
                      stat,
                      impute,
                      random_state):
    """Update dictionary from statistic
    Parameters
    ----------
    subset: ndarray (len_subset),
        Mask used on X
    Q: ndarray (n_components, n_features):
        Dictionary to perform ridge regression
    l1_ratio: float in [0, 1]:
        Controls the sparsity of the dictionary
    stat: DictMFStats,
        Statistics kept by the algorithm, to be updated by the function
    impute: boolean,
        Online update of the Gram matrix (Experimental)
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    """
    n_components = Q.shape[0]
    len_subset = subset.shape[0]
    Q_subset = np.zeros((n_components, len_subset))
    Q_subset[:] = Q[:, subset]

    if impute and not full_projection:
        stat.G -= Q_subset.dot(Q_subset.T)

    if full_projection:
        norm = enet_norm(Q, l1_ratio)
    else:
        norm = enet_norm(Q_subset, l1_ratio)

    ger, = linalg.get_blas_funcs(('ger',), (stat.A, Q_subset))
    # Intercept on first column
    if freeze_first_col:
        components_range = np.arange(1, n_components)
    else:
        components_range = np.arange(n_components)
    random_state.shuffle(components_range)

    stat.A.flat[::(n_components + 1)] += stat.F
    if full_projection:
        R = stat.B[:, subset] + stat.E[:, subset] - np.dot(Q_subset.T, stat.A).T
    else:
        R = stat.B[:, subset] + stat.E_mult * Q[:, subset] - np.dot(Q_subset.T, stat.A).T
    for j in components_range:
        ger(1.0, stat.A[j], Q_subset[j], a=R, overwrite_a=True)
        # R += np.dot(stat.A[:, j].reshape(n_components, 1),
        #  Q_subset[j].reshape(len_subset, 1).T)
        Q_subset[j] = R[j] / stat.A[j, j]
        if full_projection:
            Q[j][subset] = Q_subset[j]
            Q[j] = enet_projection(Q[j], norm[j], l1_ratio)
            Q_subset[j] = Q[j][subset]
        else:
            Q_subset[j] = enet_projection(Q_subset[j], norm[j], l1_ratio)
        ger(-1.0, stat.A[j], Q_subset[j], a=R, overwrite_a=True)
        # R -= np.dot(stat.A[:, j].reshape(n_components, 1),
        #  Q_subset[j].reshape(len_subset, 1).T)
    stat.A -= np.eye(n_components) * stat.F
    if not full_projection:
        Q[:, subset] = Q_subset

    if impute:
        if not full_projection:
            stat.G += Q_subset.dot(Q_subset.T)
        else:
            stat.G = Q.dot(Q.T).T
