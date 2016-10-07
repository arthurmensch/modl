import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state

from modl._utils.enet_proj import enet_scale
from .dict_fact_fast import DictFactImpl

from math import log

max_int = np.iinfo(np.uint32).max


class DictFact(BaseEstimator):
    def __init__(self,
                 n_components=30,
                 alpha=1.0,
                 l1_ratio=0,
                 pen_l1_ratio=0,
                 lasso_tol=1e-3,
                 purge_tol=0,
                 # Hyper-parameters
                 learning_rate=1.,
                 batch_size=1,
                 offset=0,
                 sample_learning_rate=None,
                 # Reduction parameter
                 reduction=1,
                 G_agg='full',
                 Dx_agg='average',
                 AB_agg='masked',
                 subset_sampling='random',  # ['random', 'cyclic']
                 dict_reduction='follow',
                 proj='partial',
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
                 n_threads=1,
                 temp_dir=None,
                 callback=None,
                 ):
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.sample_learning_rate = sample_learning_rate

        self.reduction = reduction
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pen_l1_ratio = pen_l1_ratio
        self.lasso_tol = lasso_tol
        self.purge_tol = purge_tol

        self.dict_init = dict_init
        self.n_components = n_components

        self.G_agg = G_agg
        self.Dx_agg = Dx_agg
        self.AB_agg = AB_agg
        self.proj = proj
        self.subset_sampling = subset_sampling
        self.dict_reduction = dict_reduction

        self.max_n_iter = max_n_iter
        self.n_epochs = n_epochs

        self.n_samples = n_samples

        self.random_state = random_state
        self.verbose = verbose

        self.n_threads = n_threads

        self.callback = callback

    @property
    def initialized(self):
        return hasattr(self, '_impl')

    @property
    def A_(self):
        return np.array(self._impl.A_)

    @property
    def B_(self):
        return np.array(self._impl.B_)

    @property
    def G_(self):
        return np.array(self._impl.G_)

    @property
    def G_average_(self):
        return np.array(self._impl.G_average_)

    @property
    def Dx_average_(self):
        return np.array(self._impl.Dx_average_)

    @property
    def D_(self):
        if self._impl.proj == 2 and self._impl.l1_ratio == 0:
            return np.array(self._impl.D_) * np.array(self._impl.D_mult)[:,
                                             np.newaxis]
        else:
            return np.array(self._impl.D_)

    @property
    def code_(self):
        return np.array(self._impl.code_)

    @property
    def total_counter_(self):
        return int(self._impl.total_counter_)

    @property
    def sample_counter_(self):
        return np.array(self._impl.sample_counter_)

    @property
    def feature_counter_(self):
        return np.array(self._impl.feature_counter_)

    @property
    def profiling_(self):
        return np.array(self._impl.profiling_, copy='True')

    @property
    def n_iter_(self):
        return self._impl.total_counter_

    def _initialize(self, X_shape, data_for_init=None):
        """Initialize statistic and dictionary"""
        if data_for_init is not None:
            assert data_for_init.shape == X_shape, ValueError
        n_samples, n_features = X_shape

        # Magic
        if self.n_samples is not None:
            self.n_samples_ = self.n_samples
        else:
            self.n_samples_ = n_samples

        random_state = check_random_state(self.random_state)
        if self.dict_init is not None:
            if self.dict_init.shape != (self.n_components, n_features):
                raise ValueError(
                    'Initial dictionary and X shape mismatch: %r != %r' % (
                        self.dict_init.shape,
                        (self.n_components, n_features)))
            D = check_array(self.dict_init, order='F',
                            dtype='float', copy=True)
        else:
            D = np.empty((self.n_components, n_features), order='F')
            if data_for_init is None:
                D[:] = random_state.randn(self.n_components, n_features)
            else:
                random_idx = random_state.permutation(n_samples)[
                             :self.n_components]
                D[:] = data_for_init[random_idx]

        D = enet_scale(D, l1_ratio=self.l1_ratio, radius=1)

        params = self._get_impl_params()
        random_seed = random_state.randint(max_int)

        self._impl = DictFactImpl(D, n_samples,
                                  n_threads=self.n_threads,
                                  random_seed=random_seed,
                                  temp_dir=self.temp_dir,
                                  **params)

    def _update_impl_params(self):
        self._impl.set_impl_params(**self._get_impl_params())

    def _get_impl_params(self):
        G_agg = {
            'masked': 1,
            'full': 2,
            'average': 3
        }
        Dx_agg = {
            'masked': 1,
            'full': 2,
            'average': 3
        }
        AB_agg = {
            'masked': 1,
            'full': 2,
            'async': 3
        }
        subset_sampling = {
            'random': 1,
            'cyclic': 2,
        }
        proj = {
            'partial': 1,
            'full': 2
        }
        if self.dict_reduction == 'follow':
            dict_reduction = 0
        elif self.dict_reduction == 'same':
            dict_reduction = self.reduction
        else:
            dict_reduction = self.dict_reduction

        if self.sample_learning_rate is None:
            self.sample_learning_rate_ = 2.5 - 2 * self.learning_rate
        else:
            self.sample_learning_rate_ = self.sample_learning_rate

        if self.verbose > 0:
            verbose_iter = np.unique((np.logspace(0, log(self.n_samples_ *
                                                         self.n_epochs // self.batch_size,
                                                         10),
                                                  self.verbose).astype(
                'i4') - 1) * self.batch_size)
            print(verbose_iter)
        else:
            verbose_iter = None

        res = {'alpha': self.alpha,
               "l1_ratio": self.l1_ratio,
               'pen_l1_ratio': self.pen_l1_ratio,
               'lasso_tol': self.lasso_tol,
               'purge_tol': self.purge_tol,
               'learning_rate': self.learning_rate,
               'sample_learning_rate': self.sample_learning_rate_,
               'offset': self.offset,
               'batch_size': self.batch_size,
               'G_agg': G_agg[self.G_agg],
               'Dx_agg': Dx_agg[self.Dx_agg],
               'AB_agg': AB_agg[self.AB_agg],
               'proj': proj[self.proj],
               'subset_sampling': subset_sampling[self.subset_sampling],
               'dict_reduction': dict_reduction,
               'reduction': self.reduction,
               'verbose_iter': verbose_iter,
               'callback': None if self.callback is None else lambda:
               self.callback(self)}
        return res

    def set_params(self, **params):
        if self.initialized:
            if 'n_samples' in params:
                raise ValueError('Cannot reset attribute n_samples after'
                                 'initialization')
            if 'n_threads' in params:
                raise ValueError('Cannot reset attribute n_threads after'
                                 'initialization')
            BaseEstimator.set_params(self, **params)
            self._update_impl_params()
        else:
            BaseEstimator.set_params(self, **params)

    def partial_fit(self, X, sample_indices=None, check_input=None):
        if sample_indices is None:
            sample_indices = np.arange(X.shape[0], dtype='i4')
        if not self.initialized or check_input is None:
            check_input = True
        if check_input:
            X = check_array(X, dtype='float', order='C')
        if not self.initialized:
            self._initialize(X.shape, data_for_init=X)
        if self.max_n_iter > 0:
            remaining_iter = self.max_n_iter - self._impl.total_counter_
            X = X[:remaining_iter]
        self._impl.partial_fit(X, sample_indices)
        return self

    def fit(self, X, y=None):
        """Use X to learn A_ dictionary Q_. The algorithm cycles on X
        until it reaches the max number of iteration

        Parameters
        ----------
        X: ndarray (n_samples, n_features)
            Dataset to learn the dictionary from
        """
        X = check_array(X, dtype='float', order='C')
        # if not X.flags['WRITEABLE']:
        #     X = np.array(X, copy=True)
        self._initialize(X.shape, data_for_init=X)
        sample_indices = np.arange(X.shape[0], dtype='i4')
        if self.max_n_iter > 0:
            while self._impl.total_counter_ < self.max_n_iter:
                self.partial_fit(X, sample_indices=sample_indices,
                                 check_input=False)
        else:
            for i in range(self.n_epochs):
                self.partial_fit(X, sample_indices=sample_indices,
                                 check_input=False)
        return self

    def transform(self, X, y=None, n_threads=None):
        if not self.initialized:
            raise ValueError()
        X = check_array(X, dtype='float64', order='C')
        code = self._impl.transform(X, n_threads=n_threads)
        return np.asarray(code)

    def score(self, X, n_threads=None):
        code = self.transform(X, n_threads=n_threads)
        loss = np.sum((X - code.dot(self.D_)) ** 2) / 2
        norm1_code = np.sum(np.abs(code))
        norm2_code = np.sum(code ** 2)
        regul = self.alpha * (norm1_code * self.pen_l1_ratio
                              + (1 - self.pen_l1_ratio) * norm2_code / 2)
        return (loss + regul) / X.shape[0]

    @property
    def components_(self):
        return np.array(self.D_, copy=True)
