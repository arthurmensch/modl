"""
Author: Arthur Mensch (2016)
Dictionary learning with masked data
"""
from math import ceil

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array

from modl._utils.enet_proj import enet_projection, enet_scale, enet_norm
from .dict_fact_fast import _update_dict, _update_code, _get_weights, \
    _get_simple_weights


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
                 # Preproc parameters
                 fit_intercept=False,
                 # Dict parameter
                 dict_init=None,
                 l1_ratio=0,
                 # Variance reduction related
                 var_red=None,
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

        self.var_red = var_red

        self.n_samples = n_samples
        self.max_n_iter = max_n_iter

        self.random_state = random_state
        self.verbose = verbose

        self.backend = backend
        self.debug = debug

        self.callback = callback

        self.full_projection = full_projection

    def _reset_stat(self):
        self.A_ *= self.multiplier_
        if self.var_red != 'past_based':
            self.B_ *= self.multiplier_
        self.multiplier_ = 1.

    def _init(self, X):
        """Initialize statistic and dictionary"""
        X = check_array(X,
                        dtype='float', order='F')

        n_rows, n_cols = X.shape

        if self.n_samples is not None:
            n_samples = self.n_samples
        else:
            n_samples = n_rows

        self.random_state_ = check_random_state(self.random_state)

        # Q dictionary
        if self.dict_init is not None:
            if self.dict_init.shape != (self.n_components, n_cols):
                raise ValueError(
                    'Initial dictionary and X shape mismatch: %r != %r' % (
                        self.dict_init.shape,
                        (self.n_components, n_cols)))
            self.components_ = check_array(self.dict_init, order='C',
                                           dtype='float', copy=True)
            if self.fit_intercept:
                if not (
                        np.all(self.components_[0] == self.components_[
                            0].mean())):
                    raise ValueError('When fitting intercept and providing '
                                     'initial dictionary, first component of'
                                     ' the dictionary should be '
                                     'proportional to [1, ..., 1]')
                self.components_[0] = 1
        else:
            self.components_ = np.empty((self.n_components, n_cols), order='C')

            if self.fit_intercept:
                self.components_[0] = 1
                self.components_[1:] = self.random_state_.randn(
                    self.n_components - 1,
                    n_cols)
            else:
                self.components_[:] = self.random_state_.randn(
                    self.n_components,
                    n_cols)

        self.components_ = np.asfortranarray(
            enet_scale(self.components_, l1_ratio=self.l1_ratio, radius=1))

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.counter_ = np.zeros(n_cols + 1, dtype='int')

        self.n_iter_ = 0

        if (self.var_red is not None and not
            self.var_red in ['code_only', 'past_based',
                             'sample_based', 'two_epochs', 'legacy']):
            raise ValueError("var_red should be in {None, 'code_only',"
                             " 'past_based', 'two_epochs', 'sample_based',"
                             "'legacy'],"
                             " got %s" % self.var_red)

        if self.var_red != 'legacy':
            self.G_ = self.components_.dot(self.components_.T).T
            self.multiplier_ = 1
            if self.var_red:
                self.row_counter_ = np.zeros(n_samples, dtype='int')
                self.beta_ = np.zeros((n_samples, self.n_components),
                                      order="F")
                if self.var_red == 'past_based':
                    self.subsets_ = np.zeros((n_samples, n_cols), dtype=bool)

        self.code_ = np.zeros((n_samples, self.n_components))

        if self.debug:
            self.loss_ = np.empty(self.max_n_iter)
            self.loss_indep_ = 0.

    def _check_init(self):
        return hasattr(self, 'components_')

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
        self.code_ = self.transform(X)

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
        X = check_array(X, order='F')
        if self.var_red != 'legacy':
            G = self.G_.copy()
        else:
            G = self.components_.dot(self.components_.T)
        Qx = self.components_.dot(X.T)
        G.flat[::self.n_components + 1] += 2 * self.alpha
        P = linalg.solve(G, Qx, sym_pos=True,
                         overwrite_a=True, check_finite=False)
        return P

    def _reduced_transform(self, X):
        n_rows, n_cols = X.shape
        G = self.G_.copy()
        G.flat[::self.n_components + 1] += 2 * self.alpha
        subset_size = int(ceil(n_cols / self.reduction))
        batches = gen_batches(len(X), self.batch_size)
        subset_range = np.arange(n_cols, dtype='i4')
        row_range = self.random_state_.permutation(n_rows)
        P = np.zeros((n_rows, self.n_components), order='C')
        for batch in batches:
            sample_subset = row_range[batch]
            self.random_state_.shuffle(subset_range)
            subset = subset_range[:subset_size]

            if self.var_red == 'past_based':
                these_subsets = np.zeros((sample_subset.shape[0], n_cols),
                                         dtype=bool)
                these_subsets[:, subset] = True
                self.subsets_[sample_subset] = these_subsets

            self.row_counter_[sample_subset] += 1
            this_X = X[sample_subset][:, subset] * self.reduction

            Qx = self.components_[:, subset].dot(this_X.T)
            P[batch] = linalg.solve(G, Qx, sym_pos=True,
                                    overwrite_a=True,
                                    check_finite=False).T
        return P

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

        X = check_array(X, dtype='float', order='F')
        n_rows, n_cols = X.shape

        if sample_subset is None:
            sample_subset = np.arange(n_rows)

        old_n_iter = self.n_iter_
        n_verbose_call = 0

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

        if self.var_red and self.var_red != 'legacy':
            unseen_subset = self.row_counter_[sample_subset] == 0
            self.code_[sample_subset][unseen_subset] = \
                self._reduced_transform(X[unseen_subset])

        for batch in batches:
            row_batch = row_range[batch]
            if 0 < self.max_n_iter <= self.n_iter_ + len(row_batch) - 1:
                return
            self.random_state_.shuffle(subset_range)
            subset = subset_range[:subset_size]
            this_X = X[row_batch]
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
                             self.offset, self.components_,
                             self.code_,
                             self.A_,
                             self.B_,
                             self.counter_,
                             self.E_,
                             self.reg_,
                             self.weights_,
                             self.G_,
                             self.beta_,
                             self.multiplier_,
                             self.var_red,
                             self.reduction,
                             self.exact_E_,
                             self.persist_P,
                             Q_subset,
                             P_temp,
                             G_temp,
                             subset_mask)
            dict_subset = subset

            if self.var_red != 'legacy' and self.multiplier_ < 1e-50:
                self._reset_stat()

            self.random_state_.shuffle(components_range)
            # Dictionary update
            if self.backend == 'python':
                self._update_dict_slow(dict_subset, components_range)
            else:
                _update_dict(self.components_,
                             dict_subset,
                             self.fit_intercept,
                             self.l1_ratio,
                             self.full_projection,
                             self.A_,
                             self.B_,
                             self.E_,
                             self.G_,
                             self.multiplier_,
                             self.var_red,
                             self.exact_E_,
                             R,
                             Q_subset,
                             norm,
                             buffer,
                             components_range)
            self.n_iter_ += len(row_batch)

            if self.verbose and (self.n_iter_ - old_n_iter) // ceil(
                    int(n_rows / self.verbose)) == n_verbose_call:
                print("Iteration %i" % self.n_iter_)
                n_verbose_call += 1
                if self.callback is not None:
                    self.callback(self)

    def _update_code_slow(self, X, subset,
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
        this_X = X[:, subset].copy()  # trigger copy

        batch_size, _ = this_X.shape
        _, n_cols = self.components_.shape

        Q_subset = self.components_[:, subset]

        self.counter_[0] += batch_size
        self.counter_[subset + 1] += batch_size

        if self.var_red == 'legacy':
            Qx = np.dot(Q_subset, this_X.T)
            this_alpha = self.alpha / self.reduction
            this_G = Q_subset.dot(Q_subset.T)
            this_G.flat[::self.n_components + 1] += this_alpha
            beta = Qx
            w = np.zeros(len(subset) + 1)
            _get_weights(w, subset, self.counter_, batch_size,
                         self.learning_rate, self.offset)
            w_A = w[0]
            w_B = w[1:]
            this_code = linalg.solve(this_G,
                                     beta,
                                     sym_pos=True, overwrite_a=True,
                                     check_finite=False)
            self.B_[:, subset] *= 1 - w_B
            self.B_[:, subset] += this_code.dot(this_X) * w_B / batch_size

            self.A_ *= 1 - w_A
            self.A_ += this_code.dot(this_code.T) * w_A / batch_size

            self.code_[sample_subset] = this_code.T
        else:
            this_X *= self.reduction
            Qx = np.dot(Q_subset, this_X.T)
            this_alpha = self.alpha
            w = _get_simple_weights(subset, self.counter_, batch_size,
                                    self.learning_rate, self.offset)
            # print(w, old_w)
            this_G = self.G_.copy()
            this_G.flat[::self.n_components + 1] += this_alpha

            if self.var_red:
                self.row_counter_[sample_subset] += 1
                w_beta = np.power(self.row_counter_[sample_subset]
                                  [:, np.newaxis], -self.learning_rate)
                self.beta_[sample_subset] *= 1 - w_beta
                self.beta_[sample_subset] += Qx.T * w_beta
                beta = self.beta_[sample_subset].T
            else:
                beta = Qx

            this_code = linalg.solve(this_G,
                                     beta,
                                     sym_pos=True, overwrite_a=True,
                                     check_finite=False)
            self.code_[sample_subset] = this_code.T

            if w != 1:
                self.multiplier_ *= 1 - w

            w_norm = w / self.multiplier_
            if self.var_red in ['past_based', 'two_epoch']:
                self.subsets_[sample_subset] = False
                these_subsets = np.zeros((len(sample_subset), n_cols),
                                         dtype=bool)
                these_subsets[:, subset] = True
                self.subsets_[sample_subset] = these_subsets
                last_subset = self.subsets_[sample_subset]

                if self.var_red == 'past_based':
                    self.B_[:, subset] += this_code.dot(
                        this_X) * w_norm / batch_size
                    for i in range(batch_size):
                        last_X = X[i][last_subset[i]] * self.reduction
                        last_code = self.code_[sample_subset[i]]
                        self.B_[:, last_subset[i]] -= np.outer(last_code,
                                                               last_X) * w_norm / batch_size
                else:
                    self.B_[:, subset] += this_code.dot(
                        this_X) * w_norm / batch_size / 2
                    for i in range(batch_size):
                        last_X = X[i][last_subset[i]] * self.reduction
                        last_code = self.code_[sample_subset[i]]
                        self.B_[:, last_subset[i]] += np.outer(last_code,
                                                               last_X) * w_norm / 2

            elif self.var_red == 'sample_based':
                self.B_ += this_code.dot(X) * w_norm / batch_size
            else:
                self.B_[:, subset] += this_code.dot(
                    this_X) * w_norm / batch_size

            self.A_ += this_code.dot(this_code.T) * w_norm / batch_size

        if self.debug:
            dict_loss = .5 * np.sum(
                self.components_.dot(self.components_.T) * self.A_) - np.sum(
                self.components_ * self.B_)
            self.loss_indep_ *= (1 - w)
            self.loss_indep_ += (.5 * np.sum(this_X ** 2) +
                                 self.alpha * np.sum(this_code ** 2)) * w
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
        Q_subset = self.components_[:, subset]

        if self.full_projection:
            norm = enet_norm(self.components_, self.l1_ratio)
        else:
            if self.var_red != 'legacy':
                self.G_ -= Q_subset.dot(Q_subset.T)
            norm = enet_norm(Q_subset, self.l1_ratio)

        if self.var_red == 'past_based':
            R = self.B_[:, subset] / self.multiplier_ - np.dot(Q_subset.T,
                                                               self.A_).T
        else:
            R = self.B_[:, subset] - np.dot(Q_subset.T, self.A_).T

        ger, = linalg.get_blas_funcs(('ger',), (self.A_, Q_subset))
        for j in components_range:
            ger(1.0, self.A_[j], Q_subset[j], a=R, overwrite_a=True)
            # R += np.dot(stat.A[:, j].reshape(n_components, 1),
            Q_subset[j] = R[j] / (self.A_[j, j])
            if self.full_projection:
                self.components_[j][subset] = Q_subset[j]
                self.components_[j] = enet_projection(self.components_[j],
                                                      norm[j],
                                                      self.l1_ratio)
                Q_subset[j] = self.components_[j][subset]
            else:
                Q_subset[j] = enet_projection(Q_subset[j], norm[j],
                                              self.l1_ratio)
            ger(-1.0, self.A_[j], Q_subset[j], a=R, overwrite_a=True)
            # R -= np.dot(stat.A[:, j].reshape(n_components, 1),
        if not self.full_projection:
            self.components_[:, subset] = Q_subset
            if self.var_red != 'legacy':
                self.G_ += Q_subset.dot(Q_subset.T)
        else:
            if self.var_red != 'legacy':
                self.G_ = self.components_.dot(self.components_.T).T

    def _callback(self):
        if self.callback is not None:
            self.callback(self)
