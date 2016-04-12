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
    var_red: boolean,
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
                 full_projection=True,
                 exact_E=None,
                 # Preproc parameters
                 fit_intercept=False,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 sparse_data=False,
                 full_G=True,
                 # Variance reduction related
                 var_red=False,
                 var_red_surr='homogeneous',
                 alpha_var_red_surr=1,
                 n_samples=None,
                 persist_P=False,
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

        self.sparse_data = sparse_data

        self.dict_init = dict_init
        self.n_components = n_components

        self.var_red = var_red
        self.var_red_surr = var_red_surr
        self.alpha_var_red_surr = alpha_var_red_surr

        self.n_samples = n_samples
        self.max_n_iter = max_n_iter

        self.persist_P = persist_P

        self.full_G = full_G

        self.random_state = random_state
        self.verbose = verbose

        self.backend = backend
        self.debug = debug

        self.callback = callback

        self.full_projection = full_projection
        self.exact_E = exact_E

    def _reset_stat(self):
        multiplier = self.var_red_mult_[0]
        for elem in ['A_', 'B_', 'beta_', 'E_', 'var_red_mult_',
                     'reg_', 'weights_']:
            if hasattr(self, elem):
                setattr(self, elem, getattr(self, elem) * multiplier)
        self.var_red_mult_[0] = 1

    def _init(self, X):
        """Initialize statistic and dictionary"""
        X = check_array(X, accept_sparse=self.sparse_data,
                        dtype='float', order='F')

        n_rows, n_cols = X.shape

        if self.n_samples is not None:
            n_rows = self.n_samples

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

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.counter_ = np.zeros(n_cols + 1, dtype='int')

        self.n_iter_ = 0

        if self.full_G:
            self.G_ = self.Q_.dot(self.Q_.T).T
        else:
            self.G_ = np.zeros((1, 1), order='F')

        self.var_red_mult_ = np.array([1, 0.])  # multiplier_, F_
        if self.var_red:
            self.beta_ = np.zeros((n_rows, self.n_components), order="F")
            self.weights_ = np.zeros(n_rows)
            self.reg_ = np.zeros(n_rows)
            self.E_ = np.zeros((self.n_components, n_cols), order='F')
        else:
            # Init dummy matrices
            self.beta_ = np.zeros((1, 1), order="F")
            self.weights_ = np.zeros(1)
            self.reg_ = np.zeros(1)
            self.E_ = np.zeros((1, 1), order='F')

        if self.persist_P or self.var_red:
            self.P_ = np.zeros((n_rows, self.n_components))
        else:
            self.P_ = np.zeros((0, 0), order='C')
        self.seen_samples_ = np.zeros(n_rows, dtype='i1')
        self.exact_E_ = self.exact_E or (
            self.exact_E is None and self.full_projection)
        if self.debug:
            self.loss_ = np.empty(self.max_n_iter)
            self.loss_indep_ = 0.

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
            while (not self._check_init() or
                                   self.n_iter_ + self.batch_size - 1 < self.max_n_iter):
                self.partial_fit(X)
        else:
            # Default to one pass
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
        X = check_array(X, accept_sparse=self.sparse_data, order='F')
        if not sp.issparse(X):
            G = self.components_.dot(self.components_.T)
            Qx = self.components_.dot(X.T)
            G.flat[::self.n_components + 1] += 2 * self.alpha
            P = linalg.solve(G, Qx, sym_pos=True,
                             overwrite_a=True, check_finite=False)
            return P
        else:
            row_range = X.getnnz(axis=1).nonzero()[0]
            n_rows, n_cols = X.shape
            P = np.zeros((n_rows, self.n_components), order='F')
            for j in row_range:
                nnz = X.indptr[j + 1] - X.indptr[j]
                idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
                x = X.data[X.indptr[j]:X.indptr[j + 1]]

                Q_idx = self.components_[:, idx]
                C = Q_idx.dot(Q_idx.T)
                Qx = Q_idx.dot(x)
                C.flat[::self.n_components + 1] += self.alpha * nnz / n_cols
                P[j] = linalg.solve(C, Qx, sym_pos=True,
                                    overwrite_a=True, check_finite=False)
            return P

    def _reduced_transform(self, X):
        # For dense matrices only
        if not self.sparse_data:
            n_rows, n_cols = X.shape
            G_temp = self.G_.copy()
            G_temp.flat[::self.n_components + 1] += 2 * self.alpha
            subset_size = int(ceil(n_cols / self.reduction))
            batches = gen_batches(len(X), self.batch_size)
            subset_range = np.arange(n_cols, dtype='i4')
            P = np.zeros((n_rows, self.n_components), order='C')
            for batch in batches:
                this_X = X[batch]
                self.random_state_.shuffle(subset_range)
                subset = subset_range[:subset_size]
                Qx = self.components_[:, subset].dot(this_X[:, subset].T)
                P[batch] = linalg.solve(G_temp, Qx, sym_pos=True,
                                        overwrite_a=True,
                                        check_finite=False).T
            return P
        else:
            return self.transform(X)

    def partial_fit(self, X, y=None, sample_subset=None):
        """Stream data X to update the estimator dictionary

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the code from

        """
        if self.backend not in ['python', 'c']:
            raise ValueError("Invalid backend %s" % self.backend)

        if self.debug and self.backend == 'c':
            raise NotImplementedError(
                "Recording objective loss is only available"
                "with backend == 'python'")

        if not self._check_init():
            self._init(X)

        X = check_array(X, accept_sparse=self.sparse_data,
                        dtype='float', order='F')
        n_rows, n_cols = X.shape

        if sample_subset is None:
            sample_subset = np.arange(n_rows)

        old_n_iter = self.n_iter_
        n_verbose_call = 0

        if self.sparse_data:
            row_range = X.getnnz(axis=1).nonzero()[0]
            max_subset_size = min(n_cols,
                                  self.batch_size * X.getnnz(axis=1).max())
        else:
            row_range = np.arange(n_rows)
            subset_size = int(ceil(n_cols / self.reduction))
            max_subset_size = subset_size

        self.random_state_.shuffle(row_range)
        batches = gen_batches(len(row_range), self.batch_size)

        if self.fit_intercept:
            components_range = np.arange(1, self.n_components)
        else:
            components_range = np.arange(self.n_components)

        subset_range = np.arange(n_cols, dtype='i4')

        if self.backend == 'c':
            # Init various arrays for efficiency
            R = np.empty((self.n_components, n_cols), order='F')
            Q_subset = np.empty((self.n_components, max_subset_size),
                                order='F')
            norm = np.zeros(self.n_components)
            if self.full_projection:
                buffer = np.zeros(n_cols)
            else:
                buffer = np.zeros(max_subset_size)

            G_temp = np.empty((self.n_components, self.n_components),
                              order='F')
            P_temp = np.empty((self.n_components, self.batch_size),
                              order='F')
            subset_mask = np.zeros(n_cols, dtype='i1')
            dict_subset_temp = np.zeros(max_subset_size, dtype='i4')
            dict_subset_lim = np.zeros(1, dtype='i4')
            X_temp = np.zeros((1, max_subset_size), order='F')

        if self.var_red and np.all(self.seen_samples_[sample_subset] == 0):
            self.P_[sample_subset] = self._reduced_transform(X)
            self.seen_samples_[sample_subset] = 1

        for batch in batches:
            row_batch = row_range[batch]

            if 0 < self.max_n_iter <= self.n_iter_ + len(row_batch) - 1:
                return
            if self.sparse_data:
                if self.backend == 'c':
                    _update_code_sparse_batch(X.data,
                                              X.indices,
                                              X.indptr,
                                              n_rows,
                                              n_cols,
                                              row_batch,
                                              sample_subset,
                                              self.alpha,
                                              self.learning_rate,
                                              self.offset,
                                              self.Q_,
                                              self.P_,
                                              self.A_, self.B_,
                                              self.counter_,
                                              self.E_,
                                              self.reg_,
                                              self.weights_,
                                              self.G_,
                                              self.beta_,
                                              self.var_red_mult_,
                                              self.var_red,
                                              self.exact_E_,
                                              self.persist_P,
                                              Q_subset,
                                              P_temp,
                                              G_temp,
                                              X_temp,
                                              subset_mask,
                                              dict_subset_temp,
                                              dict_subset_lim
                                              )
                    self.n_iter_ += row_batch.shape[0]
                    dict_subset = dict_subset_temp[:dict_subset_lim[0]]
                else:
                    for j in row_batch:
                        subset = X.indices[X.indptr[j]:X.indptr[j + 1]]

                        X_temp = np.empty((1, subset.shape[0]), order='F')
                        X_temp[:] = X.data[X.indptr[j]:X.indptr[j + 1]]
                        self._update_code_slow(X_temp, subset,
                                               sample_subset[
                                               j:(j + 1)],
                                               )
                        self.n_iter_ += 1
                    dict_subset = np.concatenate([X.indices[
                                                  X.indptr[j]:X.indptr[j + 1]]
                                                  for j in row_batch])
                    dict_subset = np.unique(dict_subset)
            else:  # X is a dense matrix : we force masks
                self.random_state_.shuffle(subset_range)
                subset = subset_range[:subset_size]
                this_X = X[row_batch][:, subset]  # Trigger copy
                if self.backend == 'python':
                    self._update_code_slow(this_X,
                                           subset,
                                           sample_subset[row_batch])
                else:
                    _update_code(this_X,
                                 subset,
                                 sample_subset[row_batch],
                                 self.alpha,
                                 self.learning_rate,
                                 self.offset, self.Q_,
                                 self.P_,
                                 self.A_,
                                 self.B_,
                                 self.counter_,
                                 self.E_,
                                 self.reg_,
                                 self.weights_,
                                 self.G_,
                                 self.beta_,
                                 self.var_red_mult_,
                                 self.var_red,
                                 self.reduction,
                                 self.exact_E_,
                                 self.persist_P,
                                 Q_subset,
                                 P_temp,
                                 G_temp,
                                 subset_mask)
                dict_subset = subset
                self.n_iter_ += len(row_batch)

            if not self.sparse_data and self.var_red_mult_[0] < 1e-50:
                self._reset_stat()

            self.random_state_.shuffle(components_range)
            # Dictionary update
            if self.backend == 'python':
                self._update_dict_slow(dict_subset, components_range)
            else:
                _update_dict(self.Q_,
                             dict_subset,
                             self.fit_intercept,
                             self.l1_ratio,
                             self.full_projection,
                             self.A_,
                             self.B_,
                             self.E_,
                             self.G_,
                             self.var_red_mult_,
                             self.var_red,
                             self.exact_E_,
                             R,
                             Q_subset,
                             norm,
                             buffer,
                             components_range)

            if self.verbose and (self.n_iter_ - old_n_iter) // ceil(
                    int(n_rows / self.verbose)) == n_verbose_call:
                print("Iteration %i" % self.n_iter_)
                n_verbose_call += 1
                if self.callback is not None:
                    self.callback(self)

    def _update_code_slow(self, this_X, this_subset, sample_subset):
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
        batch_size, _ = this_X.shape
        _, n_cols = self.Q_.shape

        Q_subset = self.Q_[:, this_subset]

        self.counter_[0] += batch_size
        self.counter_[this_subset + 1] += batch_size

        if self.sparse_data:
            this_alpha = self.alpha * this_subset.shape[0] / n_cols
        else:
            this_alpha = self.alpha
            this_X *= self.reduction
        Qx = np.dot(Q_subset, this_X.T)

        w = pow((1. + self.offset) / (self.offset + self.counter_[0]),
                self.learning_rate)

        if self.var_red:
            this_G = self.G_.copy()
            if w != 1:
                self.var_red_mult_[0] *= 1 - w
            w_norm = w / self.var_red_mult_[0]

            if self.var_red_surr == 'equilibrated':
                norm_X = np.sqrt(np.sum(this_X ** 2, axis=1))
                reg_dict = np.sum(norm_X) / self.alpha_var_red_surr
                reg_code = norm_X * self.alpha_var_red_surr
            elif self.var_red_surr == 'homogeneous':
                reg_dict = np.sum(this_X ** 2) / self.alpha_var_red_surr
                reg_code = np.ones(batch_size) * self.alpha_var_red_surr
            if self.exact_E_:
                self.E_ += w_norm / batch_size * self.Q_ * reg_dict

            self.var_red_mult_[1] += w_norm / batch_size * reg_dict
            self.reg_[sample_subset] += w_norm * (2 * this_alpha + reg_code)
            self.weights_[sample_subset] += w_norm

            self.beta_[sample_subset] += w_norm * (
                Qx.T + self.P_[sample_subset] * reg_code[:, np.newaxis])
            this_beta = self.beta_[sample_subset].copy()

            for ii, i in enumerate(sample_subset):
                this_sample_reg = self.reg_[i] / self.weights_[i]
                this_G.flat[::self.n_components + 1] += this_sample_reg
                this_P = linalg.solve(this_G, this_beta[ii] / self.weights_[i],
                                      sym_pos=True,
                                      overwrite_a=False,
                                      check_finite=False)
                this_G.flat[::self.n_components + 1] -= this_sample_reg
                self.P_[i] = this_P.T
            this_P = self.P_[sample_subset].T
            self.A_ += this_P.dot(this_P.T) * w_norm / batch_size
            self.B_[:, this_subset] += this_P.dot(this_X) * w_norm / batch_size
        else:
            if self.full_G:
                this_G = self.G_.copy()
            else:
                this_G = np.dot(Q_subset, Q_subset.T).T

            this_G.flat[::self.n_components + 1] += this_alpha
            this_P = linalg.solve(this_G, Qx, sym_pos=True, overwrite_a=True,
                                  check_finite=False)
            if self.persist_P:
                self.P_[sample_subset] = this_P.T

            if self.sparse_data:
                self.A_ *= 1 - w
                w_B = np.power(
                    (1 + self.offset) / (
                        self.offset + self.counter_[this_subset + 1]),
                    self.learning_rate)
                self.B_[:, this_subset] *= 1 - w_B
                self.B_[:, this_subset] += this_P.dot(
                    this_X) * w_B / self.reduction / batch_size
                self.A_ += this_P.dot(this_P.T) * w / batch_size
            else:
                if w != 1:
                    self.var_red_mult_[0] *= 1 - w
                w_norm = w / self.var_red_mult_[0]
                self.B_[:, this_subset] += this_P.dot(
                    this_X) * w_norm / batch_size
                self.A_ += this_P.dot(this_P.T) * w_norm / batch_size

        if self.debug:
            dict_loss = .5 * np.sum(self.Q_.dot(self.Q_.T) * self.A_) - np.sum(
                self.Q_ * self.B_)
            self.loss_indep_ *= (1 - w)
            self.loss_indep_ += (.5 * np.sum(this_X ** 2) +
                                 self.alpha * np.sum(this_P ** 2)) * w
            self.loss_[self.n_iter_] = self.loss_indep_ + dict_loss

    def _update_dict_slow(self, subset, components_range):
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
        var_red: boolean,
            Online update of the Gram matrix (Experimental)
        random_state: int or RandomState
            Pseudo number generator state used for random sampling.

        """
        n_components = self.Q_.shape[0]
        Q_subset = self.Q_[:, subset]
        if self.full_G and not self.full_projection:
            self.G_ -= Q_subset.dot(Q_subset.T)

        if self.full_projection:
            norm = enet_norm(self.Q_, self.l1_ratio)
        else:
            norm = enet_norm(Q_subset, self.l1_ratio)

        if self.var_red:
            self.A_.flat[::(n_components + 1)] += self.var_red_mult_[1]
            if self.exact_E_:
                R = self.B_[:, subset] + self.E_[:, subset] - np.dot(
                    Q_subset.T,
                    self.A_).T
            else:
                R = self.B_[:, subset] + self.var_red_mult_[1] \
                                         * Q_subset - Q_subset.T.dot(self.A_).T
        else:
            R = self.B_[:, subset] - np.dot(Q_subset.T, self.A_).T

        ger, = linalg.get_blas_funcs(('ger',), (self.A_, Q_subset))
        for j in components_range:
            ger(1.0, self.A_[j], Q_subset[j], a=R, overwrite_a=True)
            # R += np.dot(stat.A[:, j].reshape(n_components, 1),
            #  Q_subset[j].reshape(len_subset, 1).T)
            Q_subset[j] = R[j] / (self.A_[j, j])
            if self.full_projection:
                self.Q_[j][subset] = Q_subset[j]
                self.Q_[j] = enet_projection(self.Q_[j], norm[j],
                                             self.l1_ratio)
                Q_subset[j] = self.Q_[j][subset]
            else:
                Q_subset[j] = enet_projection(Q_subset[j], norm[j],
                                              self.l1_ratio)
            ger(-1.0, self.A_[j], Q_subset[j], a=R, overwrite_a=True)
            # R -= np.dot(stat.A[:, j].reshape(n_components, 1),
            #  Q_subset[j].reshape(len_subset, 1).T)
        if not self.full_projection:
            self.Q_[:, subset] = Q_subset

        if self.var_red:
            self.A_.flat[::(n_components + 1)] -= self.var_red_mult_[1]

        if self.full_G:
            if not self.full_projection:
                self.G_ += Q_subset.dot(Q_subset.T)
            else:
                self.G_ = self.Q_.dot(self.Q_.T).T

    def _callback(self):
        if self.callback is not None:
            self.callback(self)
