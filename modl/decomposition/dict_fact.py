import atexit
from concurrent.futures import ThreadPoolExecutor
from math import log, ceil, sqrt
from tempfile import TemporaryFile
from copy import deepcopy

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils.validation import check_is_fitted

from modl.utils import get_sub_slice
from .dict_fact_fast import _enet_regression_multi_gram, \
    _enet_regression_single_gram, _update_G_average, _batch_weight
from ..utils.math.enet import enet_norm, enet_scale
from .proximal import _atomic_prox

MAX_INT = 2 ** 10 - 1  # np.iinfo(np.int64).max


class CodingMixin(TransformerMixin):
    def _set_coding_params(self,
                           n_components,
                           code_alpha=1,
                           code_l1_ratio=1,
                           tol=1e-2,
                           max_iter=100,
                           code_pos=False,
                           random_state=None,
                           n_threads=1
                           ):
        self.n_components = n_components
        self.code_l1_ratio = code_l1_ratio
        self.code_alpha = code_alpha
        self.code_pos = code_pos
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

        self.n_threads = n_threads

        if self.n_threads > 1:
            self._pool = ThreadPoolExecutor(n_threads)

    def transform(self, X):
        """
        Compute the codes associated to input matrix X, decomposing it onto
        the dictionary

        Parameters
        ----------
        X: ndarray, shape = (n_samples, n_features)

        Returns
        -------
        code: ndarray, shape = (n_samples, n_components)
        """
        check_is_fitted(self, 'components_')

        dtype = self.components_.dtype
        X = check_array(X, order='C', dtype=dtype.type)
        if X.flags['WRITEABLE'] is False:
            X = X.copy()
        n_samples, n_features = X.shape
        if not hasattr(self, 'G_agg') or self.G_agg != 'full':
            G = self.components_.dot(self.components_.T)
        else:
            G = self.G_
        Dx = X.dot(self.components_.T)
        code = np.ones((n_samples, self.n_components), dtype=dtype)
        sample_indices = np.arange(n_samples)
        size_job = ceil(n_samples / self.n_threads)
        batches = list(gen_batches(n_samples, size_job))

        par_func = lambda batch: _enet_regression_single_gram(
            G, Dx[batch], X[batch], code,
            get_sub_slice(sample_indices, batch),
            self.code_l1_ratio, self.code_alpha, self.code_pos,
            self.tol, self.max_iter)
        if self.n_threads > 1:
            res = self._pool.map(par_func, batches)
            _ = list(res)
        else:
            _enet_regression_single_gram(
                G, Dx, X, code,
                sample_indices,
                self.code_l1_ratio, self.code_alpha, self.code_pos,
                self.tol, self.max_iter)

        return code

    def score(self, X):
        """
        Objective function value on test data X

        Parameters
        ----------
        X: ndarray, shape=(n_samples, n_features)
            Input matrix
        Returns
        -------
        score: float, positive
        """
        check_is_fitted(self, 'components_')

        code = self.transform(X)
        loss = np.sum((X - code.dot(self.components_)) ** 2) / 2
        norm1_code = np.sum(np.abs(code))
        norm2_code = np.sum(code ** 2)
        regul = self.code_alpha * (norm1_code * self.code_l1_ratio
                                   + (1 - self.code_l1_ratio) * norm2_code / 2)
        return (loss + regul) / X.shape[0]

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_pool', None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.n_threads > 1:
            self._pool = ThreadPoolExecutor(self.n_threads)


class DictFact(CodingMixin, BaseEstimator):
    def __init__(self,
                 reduction=1,
                 learning_rate=1,
                 sample_learning_rate=0.76,
                 Dx_agg='masked',
                 G_agg='masked',
                 dict_init=None,
                 code_alpha=1,
                 code_l1_ratio=1,
                 comp_pos=False,
                 comp_l1_ratio=1.,
                 tol=1e-2,
                 max_iter=100,
                 code_pos=False,
                 random_state=None,
                 n_epochs=1,
                 n_components=10,
                 batch_size=10,
                 verbose=0,
                 callback=None,
                 n_threads=1,
                 rand_size=True,
                 replacement=True,
                 dict_structure="enet",
                 dict_structure_params={},
                 bcd_n_iter=1,
                 feature_sampling=True
                 ):
        """
        Estimator to perform matrix factorization by streaming samples and
        subsampling them randomly to increase speed. Solve for
                argmin_{comp_l1_ratio ||D^j ||_1
                 + (1 - comp_l1_ratio) || D^j ||_2^2 < 1, A}
                1 / 2 || X - D A ||_2
                + code_alpha ((1 - code_l1_ratio) || A ||_2 / 2
                + code_l1_ratio || A ||_1)

        References
        ----------
        'Massive Online Dictionary Learning'
        A. Mensch, J. Mairal, B. Thrion, G. Varoquaux, ICML '16
        'Subsampled Online Matrix Factorization with Convergence Guarantees
        A. Mensch, J. Mairal, G. Varoquaux, B. Thrion, OPT@NIPS '16

        Parameters
        ----------
        reduction: float
            Ratio of reduction in accessing the features of the data stream.
            The larger, the faster the algorithm will go over data.
             Too large reduction may lead to slower convergence.
        learning_rate: float in ]0.917, 1]
            Weights to use in learning the dictionary. 1 means no forgetting,
            lower means forgetting the past faster, 0.917 is the theoretical
            limit for convergence.
        sample_learning_rate: float in ]0.75, 3 * learning_rate - 2[
            Weights to use in reducing the variance due to the stochastic
            subsampling, when Dx_agg == 'average' or G_agg == 'average'.
            Lower means forgetting the past faster
        Dx_agg: str in ['full', 'average', 'masked']
            Estimator to use in estimating D^T x_t
        G_agg: str in ['full', 'average', 'masked']
            Estimator to use in estimating the Gram matrix D^T D
        code_alpha: float, positive
            Penalty applied to the code in the minimization problem
        code_l1_ratio: float in [0, 1]
            Ratio of l1 penalty for the code in the minimization problem
        dict_init: ndarray, shape = (n_components, n_features)
            Initial dictionary
        n_epochs: int
            Number of epochs to perform over data
        n_components: int
            Number of pipelining in the dictionary
        batch_size: int
            Size of mini-batches to use
        code_pos: boolean,
            Learn a positive code
        comp_pos: boolean,
            Learn a positive dictionary
        random_state: np.random.RandomState or int
            Seed randomness in the learning algorithm
        comp_l1_ratio: float in [0, 1]
            Ratio of l1 in the dictionary constraint
        verbose: int, positive
            Control the verbosity of the estimator
        callback: callable,
            Function called from time to time with local variables
        n_threads: int
            Number of processors to use in the algorithm
        tol: float, positive
            Tolerance for the elastic-net solver
        max_iter: int, positive
            Maximum iteration for the elastic-net solver
        rand_size: boolean
            Whether the masks should have fixed size
        replacement: boolean
            Whether to compute random or cycling masks

        Attributes
        ----------
        self.components_: ndarray, shape = (n_components, n_features)
            Current estimation of the dictionary
        self.code_: ndarray, shape = (n_samples, n_components)
            Current estimation of each sample code
        self.C_: ndarray, shape = (n_components, n_components)
            For computing D gradient
        self.B_: ndarray, shape = (n_components, n_features)
            For computing D gradient
        self.gradient_: ndarray, shape = (n_components, n_features)
            D gradient, to perform block coordinate descent
        self.G_: ndarray, shape = (n_components, n_components)
            Gram matrix
        self.Dx_average_: ndarray, shape = (n_samples, n_components)
            Current estimate of D^T X
        self.G_average_: ndarray, shape =
        (n_samples, n_components, n_components)
            Averaged previously seen subsampled Gram matrix. Memory-mapped
        self.n_iter_: int
            Number of seen samples
        self.sample_n_iter_: int
            Number of time each sample has been seen
        self.verbose_iter_: int
            List of verbose iteration
        self.feature_sampler_: Sampler
            Generator of masks
        """

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_learning_rate = sample_learning_rate
        self.Dx_agg = Dx_agg
        self.G_agg = G_agg
        self.reduction = reduction

        self.comp_pos = comp_pos
        self.comp_l1_ratio = comp_l1_ratio

        self.dict_init = dict_init

        self._set_coding_params(n_components,
                                code_l1_ratio=code_l1_ratio,
                                code_alpha=code_alpha,
                                code_pos=code_pos,
                                random_state=random_state,
                                tol=tol,
                                max_iter=max_iter,
                                n_threads=n_threads)

        self.n_epochs = n_epochs

        self.verbose = verbose
        self.callback = callback

        self.n_threads = n_threads

        self.rand_size = rand_size
        self.replacement = replacement

        self.dict_structure = dict_structure
        self.dict_structure_params = dict_structure_params
        self.bcd_n_iter = bcd_n_iter
        self.feature_sampling = feature_sampling

    def fit(self, X):
        """
        Compute the factorisation X ~ code_ x components_, solving for
        D, code_ = argmin_{r2 ||D^j ||_1 + (1 - r2) || D^j ||_2^2 < 1}
        1 / 2 || X - D A ||_2 + (1 - r) || A ||_2 / 2 + r || A ||_1
        Parameters
        ----------
        X:  ndarray, shape= (n_samples, n_features)

        Returns
        -------
        self
        """
        X = check_array(X, order='C', dtype=[np.float32, np.float64])
        if self.dict_init is None:
            dict_init = X
        else:
            dict_init = check_array(self.dict_init,
                                    dtype=X.dtype.type)
        self.prepare(n_samples=X.shape[0], X=dict_init)
        # Main loop
        for _ in range(self.n_epochs):
            self.partial_fit(X)
            permutation = self.random_state.permutation(len(X))
            X = X[permutation]
        return self

    def partial_fit(self, X, sample_indices=None):
        """
        Update the factorization using rows from X

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            Input data
        sample_indices:
            Indices for each row of X. If None, consider that row i index is i
            (useful when providing the whole data to the function)
        Returns
        -------
        self
        """
        X = check_array(X, dtype=[np.float32, np.float64], order='C')
        n_samples, n_features = X.shape
        batches = gen_batches(n_samples, self.batch_size)

        for batch in batches:
            this_X = X[batch]
            these_sample_indices = get_sub_slice(sample_indices, batch)
            self._single_batch_fit(this_X, these_sample_indices)
        return self

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """

        G_agg = params.pop('G_agg', None)
        if G_agg == 'full' and self.G_agg != 'full':
            if hasattr(self, 'components_'):
                self.G_ = self.components_.dot(self.components_.T)
            self.G_agg = 'full'
        BaseEstimator.set_params(self, **params)

    def prepare(self, n_samples=None, n_features=None,
                dtype=None, X=None):
        """
        Init estimator attributes based on input shape and type.

        Parameters
        ----------
        n_samples: int,

        n_features: int,

        dtype: dtype in np.float32, np.float64
             to use in the estimator. Override X.dtype if provided
        X: ndarray, shape (> n_components, n_features)
            Array to use to determine shape and types, and init dictionary if
            provided

        Returns
        -------
        self
        """
        if X is not None:
            X = check_array(X, order='C', dtype=[np.float32, np.float64])
            if dtype is None:
                dtype = X.dtype
            # Transpose to fit usual column streaming
            this_n_samples = X.shape[0]
            if n_samples is None:
                n_samples = this_n_samples
            if n_features is None:
                n_features = X.shape[1]
            else:
                if n_features != X.shape[1]:
                    raise ValueError('n_features and X does not match')
        else:
            if n_features is None or n_samples is None:
                raise ValueError('Either provide'
                                 'shape or data to function prepare.')
            if dtype is None:
                dtype = np.float64
            elif dtype not in [np.float32, np.float64]:
                return ValueError('dtype should be float32 or float64')

        # Regression statistics
        if self.G_agg == 'average':
            with TemporaryFile() as self.G_average_mmap_:
                self.G_average_mmap_ = TemporaryFile()
                self.G_average_ = np.memmap(self.G_average_mmap_, mode='w+',
                                            shape=(n_samples,
                                                   self.n_components,
                                                   self.n_components),
                                            dtype=dtype)
            atexit.register(self._exit)
        self.Dx_average_ = np.zeros((n_samples, self.n_components),
                                    dtype=dtype)
        # Dictionary statistics
        self.C_ = np.zeros((self.n_components, self.n_components), dtype=dtype)
        self.B_ = np.zeros((self.n_components, n_features), dtype=dtype)
        self.gradient_ = np.zeros((self.n_components, n_features), dtype=dtype,
                                  order='F')

        self.random_state = check_random_state(self.random_state)
        if X is None:
            self.components_ = np.empty((self.n_components,
                                         n_features),
                                        dtype=dtype)
            self.components_[:, :] = self.random_state.randn(self.n_components,
                                                             n_features)
        else:
            random_idx = self.random_state.permutation(this_n_samples)[
                         :self.n_components]
            self.components_ = check_array(X[random_idx], dtype=dtype.type,
                                           copy=True)
        if self.dict_structure == "enet":
            if self.comp_pos:
                neg = self.components_ <= 0
                self.components_[neg] = -self.components_[neg]
            for i in range(self.n_components):
                enet_scale(self.components_[i], l1_ratio=self.comp_l1_ratio,
                           radius=1)

        self.code_ = np.ones((n_samples, self.n_components), dtype=dtype)

        self.labels_ = np.arange(n_samples)

        self.comp_norm_ = np.zeros(self.n_components, dtype=dtype)

        if self.G_agg == 'full':
            self.G_ = self.components_.dot(self.components_.T)

        self.n_iter_ = 0
        self.sample_n_iter_ = np.zeros(n_samples, dtype='int')
        self.random_state = check_random_state(self.random_state)
        random_seed = self.random_state.randint(MAX_INT)
        if self.verbose:
            log_lim = log(n_samples * self.n_epochs / self.batch_size, 10)
            self.verbose_iter_ = (np.logspace(0, log_lim, self.verbose,
                                              base=10) - 1) * self.batch_size
            self.verbose_iter_ = self.verbose_iter_.tolist()
        return self

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def _single_batch_fit(self, X, sample_indices):
        """Fit a single batch X: compute code, update statistics, update the
        dictionary"""
        batch_size, n_features = X.shape
        if ( self.verbose and self.verbose_iter_
            and self.n_iter_ >= self.verbose_iter_[0]):
            print('Iteration %i' % self.n_iter_)
            self.verbose_iter_ = self.verbose_iter_[1:]
            self._callback()
        if X.flags['WRITEABLE'] is False:
            X = X.copy()

        if self.feature_sampling and self.reduction > 1.:
            if self.rand_size:
                len_subset = self.random_state.binomial(n_features,
                                                        1. / self.reduction)
            else:
                len_subset = len(n_features / self.reduction)
            subset = self.random_state.choice(n_features, len_subset)
        else:
            subset = np.arange(n_features)

        self.n_iter_ += batch_size
        self.sample_n_iter_[sample_indices] += 1
        this_sample_n_iter = self.sample_n_iter_[sample_indices]
        w_sample = np.power(this_sample_n_iter, -self.sample_learning_rate). \
            astype(self.components_.dtype)
        w = _batch_weight(self.n_iter_, batch_size,
                          self.learning_rate, 0)

        self._compute_code(X, sample_indices, w_sample, subset)

        this_code = self.code_[sample_indices]

        if self.n_threads == 1:
            self._update_stat_and_dict(subset, X, this_code, w)
        else:
            self._update_stat_and_dict_parallel(subset, X,
                                                this_code, w)

    def _update_stat_and_dict(self, subset, X, code, w):
        """For multi-threading"""
        self._update_C(code, w)
        self._update_B(X, code, w)
        self.gradient_[:, subset] = self.B_[:, subset]
        self._update_dict(subset)

    def _update_stat_and_dict_parallel(self, subset, X, this_code, w):
        """For multi-threading"""
        self.gradient_[:, subset] = self.B_[:, subset]
        dict_thread = self._pool.submit(self._update_stat_partial_and_dict,
                                        subset, X, this_code, w)
        B_thread = self._pool.submit(self._update_B, X,
                                     this_code, w)
        dict_thread.result()
        B_thread.result()

    def _update_stat_partial_and_dict(self, subset, X, code, w):
        """For multi-threading"""
        self._update_C(code, w)
        # Gradient update
        batch_size = X.shape[0]
        X_subset = X[:, subset]
        self.gradient_[:, subset] *= 1 - w
        self.gradient_[:, subset] += w * code.T.dot(X_subset) / batch_size

        self._update_dict(subset)

    def _update_B(self, X, code, w):
        """Update B statistics (for updating D)"""
        batch_size = X.shape[0]
        self.B_ *= 1 - w
        self.B_ += w * code.T.dot(X) / batch_size

    def _update_C(self, this_code, w):
        """Update C statistics (for updating D)"""
        batch_size = this_code.shape[0]
        self.C_ *= 1 - w
        self.C_ += w * this_code.T.dot(this_code) / batch_size

    def _compute_code(self, X, sample_indices,
                      w_sample, subset):
        """Update regression statistics if
        necessary and compute code from X[:, subset]"""
        batch_size, n_features = X.shape
        reduction = self.reduction

        if self.n_threads > 1:
            size_job = ceil(batch_size / self.n_threads)
            batches = list(gen_batches(batch_size, size_job))

        if self.Dx_agg != 'full' or self.G_agg != 'full':
            components_subset = self.components_[:, subset]

        if self.Dx_agg == 'full':
            Dx = X.dot(self.components_.T)
        else:
            X_subset = X[:, subset]
            Dx = X_subset.dot(components_subset.T) * reduction
            self.Dx_average_[sample_indices] \
                *= 1 - w_sample[:, np.newaxis]
            self.Dx_average_[sample_indices] \
                += Dx * w_sample[:, np.newaxis]
            if self.Dx_agg == 'average':
                Dx = self.Dx_average_[sample_indices]

        if self.G_agg != 'full':
            G = components_subset.dot(components_subset.T) * reduction
            if self.G_agg == 'average':
                G_average = np.array(self.G_average_[sample_indices],
                                     copy=True)
                if self.n_threads > 1:
                    par_func = lambda batch: _update_G_average(
                        G_average[batch],
                        G,
                        w_sample[batch],
                    )
                    res = list(self._pool.map(par_func, batches))
                else:
                    _update_G_average(G_average, G, w_sample)
                self.G_average_[sample_indices] = G_average
        else:
            G = self.G_
        if self.n_threads > 1:
            if self.G_agg == 'average':
                par_func = lambda batch: _enet_regression_multi_gram(
                    G_average[batch], Dx[batch], X[batch], self.code_,
                    get_sub_slice(sample_indices, batch),
                    self.code_l1_ratio, self.code_alpha, self.code_pos,
                    self.tol, self.max_iter)
            else:
                par_func = lambda batch: _enet_regression_single_gram(
                    G, Dx[batch], X[batch], self.code_,
                    get_sub_slice(sample_indices, batch),
                    self.code_l1_ratio, self.code_alpha, self.code_pos,
                    self.tol, self.max_iter)
            res = self._pool.map(par_func, batches)
            list(res)
        else:
            if self.G_agg == 'average':
                _enet_regression_multi_gram(
                    G_average, Dx, X, self.code_,
                    sample_indices,
                    self.code_l1_ratio, self.code_alpha, self.code_pos,
                    self.tol, self.max_iter)
            else:
                _enet_regression_single_gram(
                    G, Dx, X, self.code_,
                    sample_indices,
                    self.code_l1_ratio, self.code_alpha, self.code_pos,
                    self.tol, self.max_iter)

    def _update_dict(self, subset):
        """Dictionary update part

        Parameters
        ----------
        subset: ndarray,
            Subset of features to update.

        """
        ger, = scipy.linalg.get_blas_funcs(('ger',), (self.C_,
                                                      self.components_))
        len_subset = subset.shape[0]
        n_components, n_features = self.components_.shape
        components_subset = self.components_[:, subset]
        atom_temp = np.zeros(len_subset, dtype=self.components_.dtype)
        gradient_subset = self.gradient_[:, subset]

        if self.G_agg == 'full' and len_subset < n_features / 2.:
            self.G_ -= components_subset.dot(components_subset.T)

        gradient_subset -= self.C_.dot(components_subset)
        params = deepcopy(self.dict_structure_params)
        if "alpha" in params:
            weight = params.pop("alpha")
        else:
            weight = 1.
        delta_dict = np.inf
        for bcd_iter in range(self.bcd_n_iter):
            if self.verbose and self.bcd_n_iter > 1:
                print("[BCD] iter %02i/%02i: rel. change in dict = %g" % (
                    bcd_iter + 1, self.bcd_n_iter, delta_dict))
            old_dict = components_subset.copy()
            order = self.random_state.permutation(n_components)
            for idx, k in enumerate(order):
                if self.verbose:
                    print("  (%s) updating component %02i/%02i" % (
                        self.dict_structure, idx + 1, n_components))
                comp_norm = None
                if self.dict_structure == "enet":
                    subset_norm = enet_norm(components_subset[k],
                                            self.comp_l1_ratio)
                    self.comp_norm_[k] += subset_norm
                    comp_norm = self.comp_norm_[k]
                gradient_subset = ger(1.0, self.C_[k], components_subset[k],
                                      a=gradient_subset, overwrite_a=True)
                if self.C_[k, k] > 1e-20:
                    components_subset[k] = gradient_subset[k] / self.C_[k, k]
                # Else do not update
                this_weight = weight / self.C_[k, k]
                components_subset[k] = _atomic_prox(
                    components_subset[k], which=self.dict_structure,
                    weight=this_weight, norm=comp_norm,
                    output=atom_temp, pos=self.comp_pos,
                    l1_ratio=self.comp_l1_ratio, **params)
                if self.dict_structure == "enet":
                    subset_norm = enet_norm(components_subset[k],
                                            self.comp_l1_ratio)
                    self.comp_norm_[k] -= subset_norm
                gradient_subset = ger(-1.0, self.C_[k], components_subset[k],
                                      a=gradient_subset, overwrite_a=True)

            if self.bcd_n_iter > 1:
                delta_dict = np.sum((components_subset - old_dict) ** 2)
                delta_dict /= np.sum(old_dict ** 2)
                delta_dict = sqrt(delta_dict)
        self.components_[:, subset] = components_subset

        if self.G_agg == 'full':
            if len_subset < n_features / 2.:
                self.G_ += components_subset.dot(components_subset.T)
            else:
                self.G_[:] = self.components_.dot(self.components_.T)

    def _exit(self):
        """Useful to delete G_average_ memorymap when the algorithm is
         interrupted/completed"""
        if hasattr(self, 'G_average_mmap_'):
            self.G_average_mmap_.close()


class Coder(CodingMixin, BaseEstimator):
    def __init__(self, dictionary,
                 code_alpha=1,
                 code_l1_ratio=1,
                 tol=1e-2,
                 max_iter=100,
                 code_pos=False,
                 random_state=None,
                 n_threads=1
                 ):
        self._set_coding_params(dictionary.shape[0],
                                code_l1_ratio=code_l1_ratio,
                                code_alpha=code_alpha,
                                code_pos=code_pos,
                                random_state=random_state,
                                tol=tol,
                                max_iter=max_iter,
                                n_threads=n_threads)
        self.components_ = dictionary

    def fit(self, X=None):
        return self
