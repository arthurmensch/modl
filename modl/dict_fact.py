from concurrent.futures import ThreadPoolExecutor
from math import log, ceil
from tempfile import TemporaryFile

import atexit
import numpy as np
import scipy
from modl.utils.randomkit import RandomState
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state, gen_batches

from modl.utils import get_sub_slice
from .dict_fact_fast import _enet_regression_multi_gram, \
    _enet_regression_single_gram, _update_G_average, _batch_weight
from .utils.math.enet import enet_norm, enet_projection, enet_scale
from .utils.randomkit import Sampler

MAX_INT = np.iinfo(np.int64).max


class DictFact(BaseEstimator):
    def __init__(self,
                 reduction=1,
                 learning_rate=1,
                 sample_learning_rate=0.76,
                 Dx_agg='masked',
                 G_agg='masked',
                 code_alpha=1,
                 code_l1_ratio=1,
                 n_epochs=1,
                 n_components=10,
                 batch_size=10,
                 code_pos=False,
                 comp_pos=False,
                 random_state=None,
                 comp_l1_ratio=0,
                 verbose=0,
                 callback=None,
                 n_threads=1,
                 tol=1e-2,
                 max_iter=100,
                 rand_size=True,
                 replacement=True,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_learning_rate = sample_learning_rate
        self.Dx_agg = Dx_agg
        self.G_agg = G_agg
        self.code_l1_ratio = code_l1_ratio
        self.code_alpha = code_alpha
        self.reduction = reduction
        self.comp_l1_ratio = comp_l1_ratio

        self.comp_pos = comp_pos
        self.code_pos = code_pos

        self.n_components = n_components
        self.n_epochs = n_epochs

        self.random_state = random_state

        self.verbose = verbose
        self.callback = callback

        self.n_threads = n_threads

        self.tol = tol
        self.max_iter = max_iter

        self.rand_size = rand_size
        self.replacement = replacement

    def fit(self, X):
        """
        Compute the matrix factorisation X ~ components_ x code_,solving for
        D, code_ = argmin_{r2 ||D^j ||_1 + (1 - r2) || D^j ||_2^2 < 1}
        1 / 2 || X - D A ||_2 + (1 - r) || A ||_2 / 2 + r || A ||_1
        Parameters
        ----------
        X:  n_samples * n_features

        Returns
        -------

        """
        X = check_array(X, order='C', dtype=[np.float32, np.float64])
        self.prepare(X=X)
        # Main loop
        for _ in range(self.n_epochs):
            self.partial_fit(X)
            permutation = self.relabel()
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
        G_agg = params.pop('G_agg', None)
        if G_agg == 'full' and self.G_agg != 'full':
            if self.is_prepared():
                self.G_ = self.components_.dot(self.components_.T)
            self.G_agg = 'full'
        BaseEstimator.set_params(self, **params)

    def is_prepared(self):
        return hasattr(self, 'components_')

    def shuffle(self):
        random_seed = self.random_state.randint(MAX_INT)
        random_state = RandomState(random_seed)
        list = [self.code_]
        if self.G_agg == 'average':
            list.append(self.G_average_)
        list.append(self.Dx_average_)
        perm = random_state.shuffle_with_trace(list)
        self.labels_ = self.labels_[perm]
        return perm

    def prepare(self, n_samples=None, n_features=None,
                dtype=None, X=None):
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
        if self.comp_pos:
            self.components_[self.components_ <= 0] = \
                - self.components_[self.components_ <= 0]
        for i in range(self.n_components):
            enet_scale(self.components_[i],
                       l1_ratio=self.comp_l1_ratio,
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
        self.feature_sampler_ = Sampler(n_features, self.rand_size,
                                        self.replacement, random_seed)
        if self.verbose:
            log_lim = log(n_samples * self.n_epochs / self.batch_size, 10)
            self.verbose_iter_ = (np.logspace(0, log_lim, self.verbose,
                                              base=10) - 1) * self.batch_size
            self.verbose_iter_ = self.verbose_iter_.tolist()
        if self.n_threads > 1:
            self.pool_ = ThreadPoolExecutor(self.n_threads)

    def transform(self, X):
        dtype = self.components_.dtype
        X = check_array(X, order='C', dtype=dtype.type)
        n_samples, n_features = X.shape
        if self.G_agg != 'full':
            G = self.components_.dot(self.components_.T)
        else:
            G = self.G_
        Dx = X.dot(self.components_.T)
        code = np.ones((n_samples, self.n_components), dtype=dtype)
        sample_indices = np.arange(n_samples)
        if self.n_threads > 1:
            size_job = ceil(n_samples / self.n_threads)
            batches = list(gen_batches(n_samples, size_job))
            par_func = lambda batch: _enet_regression_single_gram(
                G, Dx[batch], X[batch], code,
                get_sub_slice(sample_indices, batch),
                self.code_l1_ratio, self.code_alpha, self.code_pos,
                self.tol, self.max_iter)
            res = self.pool_.map(par_func, batches)
            _ = list(res)
        else:
            _enet_regression_single_gram(
                G, Dx, X, code, sample_indices,
                self.code_l1_ratio, self.code_alpha,
                self.code_pos, self.tol, self.max_iter)
        return code

    def score(self, X):
        code = self.transform(X)
        loss = np.sum((X - code.dot(self.components_)) ** 2) / 2
        norm1_code = np.sum(np.abs(code))
        norm2_code = np.sum(code ** 2)
        regul = self.code_alpha * (norm1_code * self.code_l1_ratio
                                   + (1 - self.code_l1_ratio) * norm2_code / 2)
        return (loss + regul) / X.shape[0]

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def _single_batch_fit(self, X, sample_indices):
        if (self.verbose and self.verbose_iter_
                and self.n_iter_ >= self.verbose_iter_[0]):
            print('Iteration %i' % self.n_iter_)
            self.verbose_iter_ = self.verbose_iter_[1:]
            self._callback()

        subset = self.feature_sampler_.yield_subset(self.reduction)
        batch_size = X.shape[0]

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
        self._update_C(code, w)
        self._update_B(X, code, w)
        self.gradient_[:, subset] = self.B_[:, subset]
        self._update_dict(subset)

    def _update_stat_and_dict_parallel(self, subset, X, this_code, w):
        self.gradient_[:, subset] = self.B_[:, subset]
        dict_thread = self.pool_.submit(self._update_stat_partial_and_dict,
                                        subset, X, this_code, w)
        B_thread = self.pool_.submit(self._update_B, X,
                                     this_code, w)
        dict_thread.result()
        B_thread.result()

    def _update_stat_partial_and_dict(self, subset, X, code, w):
        self._update_C(code, w)
        # Gradient update
        batch_size = X.shape[0]
        X_subset = X[:, subset]
        self.gradient_[:, subset] *= 1 - w
        self.gradient_[:, subset] += w * code.T.dot(X_subset) / batch_size

        self._update_dict(subset)

    def _update_B(self, X, code, w):
        batch_size = X.shape[0]
        self.B_ *= 1 - w
        self.B_ += w * code.T.dot(X) / batch_size

    def _update_C(self, this_code, w):
        batch_size = this_code.shape[0]
        self.C_ *= 1 - w
        self.C_ += w * this_code.T.dot(this_code) / batch_size

    def _compute_code(self, X, sample_indices,
                      w_sample, subset):
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
                    res = self.pool_.map(par_func, batches)
                    _ = list(res)
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
            res = self.pool_.map(par_func, batches)
            _ = list(res)
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
        len_subset = subset.shape[0]
        n_components, n_features = self.components_.shape
        components_subset = self.components_[:, subset]
        atom_temp = np.zeros(len_subset, dtype=self.components_.dtype)
        gradient_subset = self.gradient_[:, subset]
        ger, = scipy.linalg.get_blas_funcs(('ger',), (self.C_,
                                                      components_subset))

        if self.G_agg == 'full' and len_subset < n_features / 2.:
            self.G_ -= components_subset.dot(components_subset.T)

        gradient_subset -= self.C_.dot(components_subset)

        order = self.random_state.permutation(n_components)
        for k in order:
            subset_norm = enet_norm(components_subset[k],
                                    self.comp_l1_ratio)
            self.comp_norm_[k] += subset_norm
            gradient_subset = ger(1.0, self.C_[k], components_subset[k],
                                  a=gradient_subset, overwrite_a=True)
            if self.C_[k, k] > 1e-20:
                components_subset[k] = gradient_subset[k] / self.C_[k, k]
            # Else do not update
            if self.comp_pos:
                components_subset[components_subset < 0] = 0
            enet_projection(components_subset[k],
                            atom_temp,
                            self.comp_norm_[k], self.comp_l1_ratio)
            components_subset[k] = atom_temp
            subset_norm = enet_norm(components_subset[k],
                                    self.comp_l1_ratio)
            self.comp_norm_[k] -= subset_norm
            gradient_subset = ger(-1.0, self.C_[k], components_subset[k],
                                  a=gradient_subset, overwrite_a=True)

        self.components_[:, subset] = components_subset

        if self.G_agg == 'full':
            if len_subset < n_features / 2.:
                self.G_ += components_subset.dot(components_subset.T)
            else:
                self.G_[:] = self.components_.dot(self.components_.T)

    def _exit(self):
        if hasattr(self, 'G_average_mmap_'):
            self.G_average_mmap_.close()
