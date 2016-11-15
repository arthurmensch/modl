"""
Author: Arthur Mensch (2016)
Dictionary learning with masked data
"""
from math import floor

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array

from modl._utils.enet_proj import enet_projection, enet_scale, enet_norm
from .dict_fact_fast import _get_weights, \
    dict_learning_sparse, \
    dict_learning_dense, _update_subset


class DictMF(BaseEstimator):
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
                 # Reduction parameter
                 reduction=1,
                 projection='partial',
                 fit_intercept=False,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 n_samples=None,
                 # Generic parameters
                 max_n_iter=0,
                 n_epochs=1,
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

        self.n_samples = n_samples
        self.max_n_iter = max_n_iter

        self.random_state = random_state
        self.verbose = verbose

        self.backend = backend
        self.debug = debug

        self.callback = callback

        self.projection = projection

        self.n_epochs = n_epochs

    @property
    def components_(self):
        return self.D_

    def _get_projection(self):
        projections = {
            'full': 1,
            'partial': 2,
        }
        return projections[self.projection]

    def _init(self, X):
        """Initialize statistic and dictionary"""
        if self.projection not in ['partial', 'full']:
            raise ValueError("projection should be in {'partial', 'full'},"
                             " got %s" % self.projection)

        X = check_array(X, dtype='float', order='F', accept_sparse='csr')

        self.sparse_ = sp.issparse(X)

        n_rows, n_cols = X.shape

        if self.n_samples is not None:
            self.n_samples_ = self.n_samples
        else:
            self.n_samples_ = n_rows

        self.random_state_ = check_random_state(self.random_state)

        # D dictionary
        if self.dict_init is not None:
            if self.dict_init.shape != (self.n_components, n_cols):
                raise ValueError(
                    'Initial dictionary and X shape mismatch: %r != %r' % (
                        self.dict_init.shape,
                        (self.n_components, n_cols)))
            self.D_ = check_array(self.dict_init, order='C',
                                  dtype='float', copy=True)
            if self.fit_intercept:
                if not (np.all(self.D_[0] == self.D_[0].mean())):
                    raise ValueError('When fitting intercept and providing '
                                     'initial dictionary, first component of'
                                     ' the dictionary should be '
                                     'proportional to [1, ..., 1]')
                self.D_[0] = 1
        else:
            self.D_ = np.empty((self.n_components, n_cols), order='C')

            if self.fit_intercept:
                self.D_[0] = 1
                U = self.random_state_.randn(
                    n_cols,
                    self.n_components - 1,
                )
                Q, _ = np.linalg.qr(U)
                self.D_[1:] = Q.T
            else:
                self.D_[:] = self.random_state_.randn(
                    self.n_components,
                    n_cols)

        self.D_ = np.asfortranarray(
            enet_scale(self.D_, l1_ratio=self.l1_ratio, radius=1))

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.counter_ = np.zeros(n_cols + 1, dtype='int')

        self.n_iter_ = np.zeros(1, dtype='int64')

        self.code_ = np.zeros((self.n_samples_, self.n_components))


    def _init_arrays(self, X):
        n_rows, n_cols = X.shape
        if self.sparse_:
            self._len_subset = min(n_cols,
                                   self.batch_size * X.getnnz(axis=1).max())
        else:
            self._len_subset = int(floor(n_cols / self.reduction))
        if self.fit_intercept:
            self._D_range = np.arange(1, self.n_components)
        else:
            self._D_range = np.arange(self.n_components)
        if self.sparse_:
            self._this_X = np.empty((1, self._len_subset), order='F')
        else:
            self._this_X = np.empty((self.batch_size, self._len_subset),
                                    order='F')
        self._this_sample_subset = np.empty(self.batch_size, dtype='int')
        if self.backend == 'c':
            # Init various arrays for efficiency
            self._D_subset = np.empty((self.n_components, self._len_subset),
                                      order='F')
            self._G_temp = np.empty((self.n_components, self.n_components),
                                    order='F')
            self._code_temp = np.empty((self.n_components, self.batch_size),
                                       order='F')
            self._w_temp = np.zeros(self._len_subset + 1)
            self._R = np.empty((self.n_components, n_cols), order='F')
            self._norm_temp = np.zeros(self.n_components)
            if self.projection == 'full':
                self._proj_temp = np.zeros(n_cols)
            else:
                self._proj_temp = np.zeros(self._len_subset)
            if self.sparse_:
                self._subset_mask = np.zeros(n_cols, dtype='i1')
                self._dict_subset = np.zeros(self._len_subset, dtype='i4')
                self._dict_subset_lim = np.zeros(1, dtype='i4')
        if not self.sparse_:
            self._subset_range = np.arange(n_cols, dtype='i4')
            if self.reduction >= 1:
                self.random_state_.shuffle(self._subset_range)
            self._temp_subset = np.empty(n_cols, dtype='i4')
            self._subset_lim = np.zeros(2, dtype='i4')

    def _is_initialized(self):
        return hasattr(self, 'D_')

    def fit(self, X, y=None):
        """Use X to learn a dictionary Q_. The algorithm cycles on X
        until it reaches the max number of iteration

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the dictionary from
        """
        X = self._prefit(X, reset=True)
        if self.max_n_iter > 0:
            while self.n_iter_[0] + self.batch_size - 1 < self.max_n_iter:
                self.partial_fit(X, check_input=False)
        else:
            for _ in range(self.n_epochs):
                self.partial_fit(X, check_input=False)

    def _prefit(self, X, reset=False, check_input=True):
        if reset or not self._is_initialized():
            if self.backend not in ['python', 'c']:
                raise ValueError("Invalid backend %s" % self.backend)

            if self.debug and self.backend == 'c':
                raise NotImplementedError(
                    "Recording objective loss is only available"
                    "with backend == 'python'")

            self._init(X)
            self._init_arrays(X)
        if check_input:
            X = check_array(X, dtype='float', order='C',
                            accept_sparse='csr' if self.sparse_ else None)
        return X

    def _refit(self, X):
        """Use X and Q to learn a code P"""
        self.code_ = self.transform(X)

    def _check_fitted(self):
        if not hasattr(self, 'D_'):
            raise ValueError('DictLearning object has not been'
                             ' fitted before transform')

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
        self._check_fitted()
        if self.sparse_:
            return self._sparse_transform(X)
        else:
            return self._dense_transform(X)

    def _dense_transform(self, X, y=None):
        X = check_array(X, order='F')
        G = self.D_.dot(self.D_.T)
        Dx = self.D_.dot(X.T)
        G.flat[::self.n_components + 1] += 2 * self.alpha
        code = linalg.solve(G, Dx, sym_pos=True,
                            overwrite_a=True, check_finite=False)
        return code

    def _sparse_transform(self, X):
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
        row_range = X.getnnz(axis=1).nonzero()[0]
        n_rows, n_cols = X.shape
        code = np.zeros((n_rows, self.n_components), order='C')
        for j in row_range:
            nnz = X.indptr[j + 1] - X.indptr[j]
            idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
            x = X.data[X.indptr[j]:X.indptr[j + 1]]

            Q_idx = self.D_[:, idx]
            G = Q_idx.dot(Q_idx.T)
            Qx = Q_idx.dot(x)
            G.flat[::self.n_components + 1] += self.alpha * nnz / n_cols
            code[j] = linalg.solve(G, Qx, sym_pos=True,
                                   overwrite_a=True, check_finite=False)
        return code

    def partial_fit(self, X, y=None, sample_subset=None, check_input=True):
        """Stream data X to update the estimator dictionary

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the code from

        """
        X = self._prefit(X, check_input=check_input)
        n_rows, n_cols = X.shape
        # Sample related variables
        if sample_subset is None:
            sample_subset = np.arange(n_rows, dtype='int')

        row_range = np.arange(n_rows)

        self.random_state_.shuffle(row_range)

        if self.backend == 'c':
            random_seed = self.random_state_.randint(np.iinfo(np.uint32).max)
            if self.sparse_:
                dict_learning_sparse(X.data, X.indices,
                                     X.indptr,
                                     n_rows,
                                     n_cols,
                                     row_range,
                                     sample_subset,
                                     self.batch_size,
                                     self.alpha,
                                     self.learning_rate,
                                     self.offset,
                                     self.fit_intercept,
                                     self.l1_ratio,
                                     self._get_projection(),
                                     self.D_,
                                     self.code_,
                                     self.A_,
                                     self.B_,
                                     self.counter_,
                                     self._D_subset,
                                     self._code_temp,
                                     self._G_temp,
                                     self._this_X,
                                     self._w_temp,
                                     self._subset_mask,
                                     self._dict_subset,
                                     self._dict_subset_lim,
                                     self._this_sample_subset,
                                     self._R,
                                     self._D_range,
                                     self._norm_temp,
                                     self._proj_temp,
                                     random_seed,
                                     self.verbose,
                                     self.n_iter_,
                                     self._callback,
                                     )
            else:
                dict_learning_dense(X,
                                    row_range,
                                    sample_subset,
                                    self.batch_size,
                                    self.alpha,
                                    self.learning_rate,
                                    self.offset,
                                    self.fit_intercept,
                                    self.l1_ratio,
                                    self._get_projection(),
                                    self.D_,
                                    self.code_,
                                    self.A_,
                                    self.B_,
                                    self.counter_,
                                    self._D_subset,
                                    self._code_temp,
                                    self._G_temp,
                                    self._this_X,
                                    self._w_temp,
                                    self._len_subset,
                                    self._subset_range,
                                    self._temp_subset,
                                    self._subset_lim,
                                    self._this_sample_subset,
                                    self._R,
                                    self._D_range,
                                    self._norm_temp,
                                    self._proj_temp,
                                    random_seed,
                                    self.verbose,
                                    self.n_iter_,
                                    self._callback,
                                    )

        else:
            new_verbose_iter_ = 0
            old_n_iter = self.n_iter_[0]

            batches = gen_batches(len(row_range), self.batch_size)

            for batch in batches:
                if self.verbose:
                    if self.n_iter_[0] - old_n_iter >= new_verbose_iter_:
                        print("Iteration %i" % self.n_iter_[0])
                        new_verbose_iter_ += n_rows // self.verbose
                        self._callback()

                row_batch = row_range[batch]
                len_batch = row_batch.shape[0]

                self._this_sample_subset[:len_batch] = sample_subset[row_batch]

                if 0 < self.max_n_iter <= self.n_iter_[0] + len_batch - 1:
                    return
                if self.sparse_:
                    for j in row_batch:
                        subset = X.indices[X.indptr[j]:X.indptr[j + 1]]
                        if len(subset) == 0:
                            continue
                        self._this_X[0, :subset.shape[0]] = X.data[
                                                            X.indptr[j]:
                                                            X.indptr[
                                                                j + 1]]
                        self._update_code_slow(
                            self._this_X[:, :subset.shape[0]],
                            subset,
                            sample_subset[j:j + 1])
                    dict_subset = np.concatenate([X.indices[
                                                  X.indptr[j]:X.indptr[
                                                      j + 1]]
                                                  for j in row_batch])
                    dict_subset = np.unique(dict_subset)
                # End if self.sparse_
                else:
                    random_seed = self.random_state_.randint(
                        np.iinfo(np.uint32).max)
                    _update_subset(False,
                                   self._len_subset,
                                   self._subset_range,
                                   self._subset_lim,
                                   self._temp_subset,
                                   random_seed)
                    subset = self._subset_range[
                             self._subset_lim[0]:self._subset_lim[1]]
                    self._this_X[:len_batch] = X[row_batch][:, subset]

                    self._update_code_slow(self._this_X,
                                           subset,
                                           sample_subset[row_batch],
                                           )
                    dict_subset = subset

                # End else
                self.random_state_.shuffle(self._D_range)
                # Dictionary update
                self._update_dict_slow(dict_subset, self._D_range)
                self.n_iter_[0] += len(row_batch)

    def _update_code_slow(self, this_X,
                          subset,
                          sample_subset):
        """Compute code for a mini-batch and update algorithm statistics accordingly

        Parameters
        ----------
        this_X: ndarray, (batch_size, len_subset)
            Mini-batch of masked data to perform the update from
        this_subset: ndarray (len_subset),
            Mask used on X
        sample_subset: ndarray (batch_size),
            Sample indices of this_X within X
        """
        len_batch = sample_subset.shape[0]

        len_subset = subset.shape[0]

        if len_batch != self.batch_size:
            this_X = this_X[:len_batch]

        _, n_cols = self.D_.shape

        D_subset = self.D_[:, subset]

        self.counter_[0] += len_batch

        reduction = n_cols / len_subset

        self.counter_[subset + 1] += len_batch
        w = np.zeros(len(subset) + 1)
        _get_weights(w, subset, self.counter_, len_batch,
                     self.learning_rate, self.offset)
        w_A = w[0]
        w_B = w[1:]
        Dx = np.dot(D_subset, this_X.T)
        this_G = D_subset.dot(D_subset.T)
        this_G.flat[::self.n_components + 1] += self.alpha / reduction
        this_beta = Dx
        this_code = linalg.solve(this_G,
                                 this_beta,
                                 sym_pos=True, overwrite_a=True,
                                 check_finite=False)
        self.A_ *= 1 - w_A
        self.A_ += this_code.dot(this_code.T) * w_A / len_batch
        self.B_[:, subset] *= 1 - w_B
        self.B_[:, subset] += this_code.dot(this_X) * w_B / len_batch

        self.code_[sample_subset] = this_code.T

    def _update_dict_slow(self, subset, D_range):
        """Update dictionary from statistic
        Parameters
        ----------
        subset: ndarray (len_subset),
            Mask used on X

        """
        D_subset = self.D_[:, subset]

        if self.projection == 'full':
            norm = enet_norm(self.D_, self.l1_ratio)
        else:
            norm = enet_norm(D_subset, self.l1_ratio)
        R = self.B_[:, subset] - np.dot(D_subset.T, self.A_).T

        ger, = linalg.get_blas_funcs(('ger',), (self.A_, D_subset))
        for k in D_range:
            ger(1.0, self.A_[k], D_subset[k], a=R, overwrite_a=True)
            # R += np.dot(stat.A[:, j].reshape(n_components, 1),
            D_subset[k] = R[k] / (self.A_[k, k])
            if self.projection == 'full':
                self.D_[k][subset] = D_subset[k]
                self.D_[k] = enet_projection(self.D_[k],
                                             norm[k],
                                             self.l1_ratio)
                D_subset[k] = self.D_[k][subset]
            else:
                D_subset[k] = enet_projection(D_subset[k], norm[k],
                                              self.l1_ratio)
            ger(-1.0, self.A_[k], D_subset[k], a=R, overwrite_a=True)
            # R -= np.dot(stat.A[:, j].reshape(n_components, 1),
        if self.projection == 'partial':
            self.D_[:, subset] = D_subset

    def _callback(self):
        if self.callback is not None:
            self.callback(self)
