"""
Author: Arthur Mensch (2016)
Dictionary learning with masked data
"""
from math import floor

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches, check_array

from modl._utils.enet_proj import enet_projection, enet_scale, enet_norm
from .dict_fact_fast import dict_learning, _update_subset, \
    _get_simple_weights, enet_coordinate_descent_gram, sparse_coding


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

    _dummy_2d_float = np.zeros((1, 1), order='F')
    _dummy_3d_float = np.zeros((1, 1, 1))

    def __init__(self,
                 n_components=30,
                 alpha=1.0,
                 l1_ratio=0,
                 pen_l1_ratio=0,
                 tol=1e-3,
                 # Hyper-parameters
                 learning_rate=1.,
                 batch_size=1,
                 offset=0,
                 sample_learning_rate=None,
                 # Reduction parameter
                 reduction=1,
                 solver='gram',  # ['average', 'gram', 'masked']
                 weights='sync',  # ['sync', 'async']
                 subset_sampling='random',  # ['random', 'cyclic']
                 dict_subset_sampling='independent',
                 # ['independent', 'coupled']
                 # Dict parameter
                 dict_init=None,
                 # For variance reduction
                 n_samples=None,
                 # Generic parameters
                 max_n_iter=0,
                 n_epochs=1,
                 random_state=None,
                 verbose=0,
                 backend='c',
                 n_threads=1,
                 callback=None):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.sample_learning_rate = sample_learning_rate

        self.reduction = reduction
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pen_l1_ratio = pen_l1_ratio
        self.tol = tol

        self.dict_init = dict_init
        self.n_components = n_components

        self.solver = solver
        self.subset_sampling = subset_sampling
        self.dict_subset_sampling = dict_subset_sampling
        self.weights = weights

        self.max_n_iter = max_n_iter
        self.n_epochs = n_epochs

        self.n_samples = n_samples

        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend

        self.n_threads = n_threads

        self.callback = callback

    @property
    def components_(self):
        return enet_scale(self.D_, 1, self.l1_ratio)

    def _get_solver(self):
        solver = {
            'masked': 1,
            'gram': 2,
            'average': 3,
        }
        return solver[self.solver]

    def _get_weights(self):
        weights = {
            'sync': 1,
            'async_freq': 2,
            'async_prob': 3
        }
        return weights[self.weights]

    def _get_subset_sampling(self):
        subset_sampling = {
            'random': 1,
            'cyclic': 2,
        }
        return subset_sampling[self.subset_sampling]

    def _get_dict_subset_sampling(self):
        dict_subset_sampling = {
            'independent': 1,
            'coupled': 2,
        }
        return dict_subset_sampling[self.dict_subset_sampling]

    def _init(self, X):
        """Initialize statistic and dictionary"""
        X = check_array(X, dtype='float', order='F')

        n_rows, n_cols = X.shape

        # Magic
        if self.n_samples is not None:
            self.n_samples_ = self.n_samples
        else:
            self.n_samples_ = n_rows

        if self.sample_learning_rate is None:
            self.sample_learning_rate_ = 2.5 - 2 * self.learning_rate
        else:
            self.sample_learning_rate_ = self.sample_learning_rate
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
        else:
            self.D_ = np.empty((self.n_components, n_cols), order='C')
            self.D_[:] = self.random_state_.randn(self.n_components, n_cols)

        self.D_ = np.asfortranarray(
            enet_scale(self.D_, l1_ratio=self.l1_ratio, radius=1))

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        # We keep code even if it's not necessary
        self.code_ = np.zeros((self.n_samples_, self.n_components))

        self.counter_ = np.zeros(n_cols + 1, dtype='int')
        self.row_counter_ = np.zeros(self.n_samples_, dtype='int')

        if self.solver in ['gram', 'average']:
            self.Dx_average_ = np.zeros((self.n_samples_, self.n_components),
                                        order="F")
        if self.solver == 'gram':
            self.G_ = self.D_.dot(self.D_.T).T

        if self.solver == 'average':
            self.G_average_ = np.zeros((self.n_components,
                                        self.n_components, self.n_samples_),
                                       order="F")

        self.n_iter_ = np.zeros(1, dtype='long')

        # Temporary array allocation
        if self.backend == 'c':
            self._this_X = np.empty((self.batch_size, n_cols),
                                    order='F')
            self._full_X = np.empty((self.batch_size, n_cols),
                                    order='F')
            self._D_subset = np.empty((self.n_components, n_cols),
                                      order='F')
            self._Dx = np.empty((self.n_components, self.batch_size),
                                order='F')
            self._G_temp = np.empty((self.n_components, self.n_components),
                                    order='F')

            self._R = np.empty((self.n_components, n_cols), order='F')
            self._norm_temp = np.zeros(self.n_components)
            self._proj_temp = np.zeros(n_cols)

        self._H = np.empty((self.batch_size, self.n_components))
        self._XtA = np.empty((self.batch_size, self.n_components))

        self._this_sample_subset = np.empty(self.batch_size, dtype='long')

        self._subset_temp = np.empty(n_cols, dtype='long')

        self._subset_range = np.arange(n_cols)
        self.random_state_.shuffle(self._subset_range)
        self._subset_lim = np.zeros(2, dtype='long')

        self._dict_subset_range = np.arange(n_cols)
        self.random_state_.shuffle(self._dict_subset_range)
        self._dict_subset_lim = np.zeros(2, dtype='long')

        self._D_range = np.arange(self.n_components)

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
        return self

    def _prefit(self, X, reset=False, check_input=True):
        if reset or not self._is_initialized():
            self._init(X)
        if check_input:
            X = check_array(X, dtype='float', order='C')
        return X

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
        X = check_array(X, order='C', dtype='float64')
        n_rows, n_cols = X.shape
        D = self.components_
        Dx = (X.dot(D.T)).T
        G = D.dot(D.T).T
        if self.pen_l1_ratio == 0:
            G = G.copy()
            G.flat[::self.n_components + 1] += self.alpha
            code = linalg.solve(G, Dx, sym_pos=True,
                                overwrite_a=True, check_finite=False)
        else:
            code = np.ones((n_rows, self.n_components), order='C')
            if self.backend == 'python':
                for i in range(n_rows):
                    random_seed = self.random_state_.randint(
                        np.iinfo(np.uint32).max)
                    enet_coordinate_descent_gram(
                        code[i], self.alpha * self.pen_l1_ratio,
                                 self.alpha * (1 - self.pen_l1_ratio),
                        G, Dx[:, i], X[i],
                        self._H[0],
                        self._XtA[0],
                        100,
                        self.tol, random_seed, 0, 0)
            else:
                H = np.empty((n_rows, self.n_components), order='C')
                XtA = np.empty((n_rows, self.n_components), order='C')
                random_seed = self.random_state_.randint(
                    np.iinfo(np.uint32).max)
                sparse_coding(self.alpha,
                              self.pen_l1_ratio,
                              self.tol,
                              code,
                              H,
                              XtA,
                              random_seed,
                              G,
                              Dx,
                              X,
                              self.n_threads)
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
            dict_learning(X,
                          row_range,
                          sample_subset,
                          self.batch_size,
                          self.alpha,
                          self.learning_rate,
                          self.sample_learning_rate_,
                          self.offset,
                          self.l1_ratio,
                          self.pen_l1_ratio,
                          self.tol,
                          self.reduction,
                          self._get_solver(),
                          self._get_weights(),
                          self._get_subset_sampling(),
                          self._get_dict_subset_sampling(),
                          self.D_,
                          self.code_,
                          self.A_,
                          self.B_,
                          self.G_ if self.solver == 'gram'
                          else DictMF._dummy_2d_float,
                          self.Dx_average_ if self.solver in ['gram',
                                                              'average']
                          else DictMF._dummy_2d_float,
                          self.G_average_ if self.solver == 'average'
                          else DictMF._dummy_3d_float,
                          self.n_iter_,
                          self.counter_,
                          self.row_counter_,
                          self._D_subset,
                          self._Dx,
                          self._G_temp,
                          self._this_X,
                          self._full_X,
                          self._H,
                          self._XtA,
                          self._subset_range,
                          self._subset_temp,
                          self._subset_lim,
                          self._dict_subset_range,
                          self._dict_subset_lim,
                          self._this_sample_subset,
                          self._R,
                          self._D_range,
                          self._norm_temp,
                          self._proj_temp,
                          self.verbose,
                          self.n_threads,
                          random_seed,
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

                if 0 < self.max_n_iter <= self.n_iter_[0] + len_batch - 1:
                    return

                len_subset = int(floor(n_cols / self.reduction))
                random_seed = self.random_state_.randint(
                    np.iinfo(np.uint32).max)
                _update_subset(self.subset_sampling != 'cyclic',
                               len_subset,
                               self._subset_range,
                               self._subset_lim,
                               self._subset_temp,
                               random_seed)
                subset = self._subset_range[
                         self._subset_lim[0]:self._subset_lim[1]]

                self.counter_[0] += len_batch
                self.counter_[subset + 1] += len_batch
                self.row_counter_[sample_subset[row_batch]] += 1

                self._update_code_slow(X[row_batch],
                                       subset,
                                       sample_subset[row_batch],
                                       )

                if self.dict_subset_sampling == 'coupled':
                    dict_subset = subset
                else:
                    random_seed = self.random_state_.randint(
                        np.iinfo(np.uint32).max)
                    _update_subset(self.subset_sampling != 'cyclic',
                                   len_subset,
                                   self._dict_subset_range,
                                   self._dict_subset_lim,
                                   self._subset_temp,
                                   random_seed)
                    dict_subset = self._dict_subset_range[
                                  self._dict_subset_lim[0]:
                                  self._dict_subset_lim[1]]
                # End else
                self.random_state_.shuffle(self._D_range)
                # Dictionary update
                self._update_dict_slow(dict_subset, self._D_range)
                self.n_iter_[0] += len(row_batch)

    def _update_code_slow(self, full_X,
                          subset,
                          this_sample_subset,
                          ):
        """Compute code for a mini-batch and update algorithm statistics accordingly

        Parameters
        ----------
        this_X: ndarray, (batch_size, len_subset)
            Mini-batch of masked data to perform the update from
        this_subset: ndarray (len_subset),
            Mask used on X
        this_sample_subset: ndarray (batch_size),
            Sample indices of this_X within X
        """
        this_X = full_X[:, subset]
        len_batch = this_sample_subset.shape[0]
        len_subset = subset.shape[0]

        _, n_cols = self.D_.shape
        reduction = n_cols / len_subset

        D_subset = self.D_[:, subset]
        this_X *= reduction
        Dx = np.dot(this_X, D_subset.T).T

        if self.solver == 'masked':
            G_temp = D_subset.dot(D_subset.T).T * reduction
        else:  # ['full', 'gram']
            w_sample = np.power(self.row_counter_[this_sample_subset]
                                , -self.sample_learning_rate_)
            self.Dx_average_[this_sample_subset] *= 1 - w_sample[:, np.newaxis]
            self.Dx_average_[this_sample_subset] += Dx.T * w_sample[:,
                                                           np.newaxis]
            Dx = self.Dx_average_[this_sample_subset].T
            if self.solver == 'average':
                G_temp = D_subset.dot(D_subset.T) * reduction
                self.G_average_[:, :, this_sample_subset] *= 1 - w_sample[
                                                                 np.newaxis,
                                                                 np.newaxis, :]
                self.G_average_[:, :, this_sample_subset] += G_temp[:, :,
                                                             np.newaxis] * w_sample[
                                                                           np.newaxis,
                                                                           np.newaxis,
                                                                           :]
            else:  # ['gram']
                G_temp = self.G_
        if self.pen_l1_ratio == 0:
            if self.solver in ['gram', 'masked']:
                G_temp = G_temp.copy()
                G_temp.flat[::self.n_components + 1] += self.alpha
                self.code_[this_sample_subset] = linalg.solve(G_temp,
                                                              Dx,
                                                              sym_pos=True,
                                                              overwrite_a=False,
                                                              check_finite=False).T
            else:
                for ii in range(len_batch):
                    G_temp = self.G_average_[:, :,
                             this_sample_subset[ii]].copy()
                    G_temp.flat[::self.n_components + 1] += self.alpha
                    self.code_[
                        this_sample_subset[ii]] = linalg.solve(G_temp,
                                                               Dx[:, ii],
                                                               sym_pos=True,
                                                               overwrite_a=False,
                                                               check_finite=False)
        else:
            for ii in range(len_batch):
                if self.solver == 'average':
                    G_temp = self.G_average_[:, :, this_sample_subset[ii]]
                random_seed = self.random_state_.randint(
                    np.iinfo(np.uint32).max)
                enet_coordinate_descent_gram(
                    self.code_[this_sample_subset[ii]],
                    self.alpha * self.pen_l1_ratio,
                    self.alpha * (1 - self.pen_l1_ratio),
                    G_temp, Dx[:, ii],
                    this_X[ii],
                    self._H[ii],
                    self._XtA[ii],
                    1000,
                    self.tol, random_seed, 0, 0)
        this_X /= reduction
        w_A = _get_simple_weights(self.counter_[0], len_batch,
                                  self.learning_rate, self.offset)
        this_code = self.code_[this_sample_subset]

        self.A_ *= 1 - w_A
        self.A_ += this_code.T.dot(this_code) * w_A / len_batch

        if self.weights == 'sync':
            self.B_ *= 1 - w_A
            self.B_ += this_code.T.dot(full_X) * w_A / len_batch
        else:
            if self.weights == 'async_freq':
                w_B = w_A * self.counter_[0] / self.counter_[subset + 1]
                w_B = np.minimum(1, w_B)
            else:
                w_B = min(1., w_A * reduction)
            self.B_[:, subset] *= 1 - w_B
            self.B_[:, subset] += this_code.T.dot(this_X) * w_B / len_batch

    def _update_dict_slow(self, subset, D_range):
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
        n_cols = self.D_.shape[1]
        D_subset = self.D_[:, subset]

        norm = enet_norm(D_subset, self.l1_ratio)
        if self.solver == 'gram':
            self.G_ -= D_subset.dot(D_subset.T)

        # Cleaning D from unused atom
        # non_active = np.logical_or(norm < 1e-20, np.diag(self.A_) < 1e-20)
        # if np.sum(non_active) > 0:
        #     if self.solver == 'gram':
        #         self.G_[non_active, :] += D_subset.dot(
        #             D_subset[non_active].T).T
        #         self.G_[:, non_active] = self.G_[non_active, :].T
        #
        #     self.D_[non_active] = self.random_state_.randn(n_cols)
        #     self.D_[non_active] = enet_scale(self.D_[non_active],
        #                                      l1_ratio=self.l1_ratio)
        #     self.A_[non_active, :] = 0
        #     self.A_[:, non_active] = 0
        #     self.B_[non_active, :] = 0
        #
        #     if self.solver == 'gram':
        #         self.G_[non_active, :] = self.D_.dot(
        #             self.D_[non_active].T).T
        #         self.G_[:, non_active] = self.G_[non_active, :].T
        #
        #     D_subset[non_active] = self.D_[non_active][:, subset]
        #     norm[non_active] = enet_norm(D_subset[non_active],
        #                                  self.l1_ratio)
        #     if self.solver == 'gram':
        #         self.G_[non_active, :] -= D_subset.dot(D_subset[
        #                                                    non_active].T).T
        #         self.G_[:, non_active] = self.G_[non_active, :].T

        R = self.B_[:, subset] - np.dot(D_subset.T, self.A_).T

        ger, = linalg.get_blas_funcs(('ger',), (self.A_, D_subset))
        for k in D_range:
            # R{k] = self.B_[k][subset] - np.dot(D_subset.T, self.A_[k]).T
            ger(1.0, self.A_[k], D_subset[k], a=R, overwrite_a=True)
            if self.A_[k, k] > 1e-20:
                D_subset[k] = R[k] / self.A_[k, k]
            D_subset[k] = enet_projection(D_subset[k], norm[k],
                                          self.l1_ratio)
            ger(-1.0, self.A_[k], D_subset[k], a=R, overwrite_a=True)
        self.D_[:, subset] = D_subset
        if self.solver == 'gram':
            self.G_ += D_subset.dot(D_subset.T)

    def score(self, X):
        code = self.transform(X)
        loss = np.sum((X - code.dot(self.components_)) ** 2) / 2
        norm1_code = np.sum(np.abs(code))
        norm2_code = np.sum(code ** 2)
        regul = self.alpha * (norm1_code * self.pen_l1_ratio
                              + (1 - self.pen_l1_ratio) * norm2_code / 2)
        return (loss + regul) / X.shape[0]

    def _callback(self):
        if self.callback is not None:
            self.callback(self)
