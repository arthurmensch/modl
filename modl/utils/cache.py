from nilearn._utils import CacheMixin
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Memory
from sklearn.utils.metaestimators import if_delegate_has_method


def _cached_call(estimator, method, *args, **kwargs):
    func = getattr(estimator, method)
    return func(*args, **kwargs)


# memory_level = 1 -> fit is cached
# memory_level = 2 -> everything is cached
class CachedEstimator(BaseEstimator, CacheMixin):
    def __init__(self, estimator, memory=Memory(cachedir=None),
                 memory_level=1):
        self.estimator = estimator

        self.memory = memory
        self.memory_level = memory_level

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.estimator_.classes_

    def score(self, X, y=None):
        """Call score with cache.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'score',
                                                X, y)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def predict(self, X):
        """Call predict with cache.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'predict',
                                                X)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba with cache.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'predict_proba',
                                                X)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba with cache.
        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'predict_log_proba',
                                                X)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function with cache.
        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'decision_function',
                                                X)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def transform(self, X):
        """Call transform with cache.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'transform',
                                                X)

    @if_delegate_has_method(delegate=('estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform with cache.

        Parameters
        -----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self._cache(_cached_call,
                           func_memory_level=2)(self.estimator_,
                                                'inverse_transform',
                                                Xt)

    def fit(self, X, y):
        """Call fit with cache.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        # Strip attributes
        self.estimator_ = clone(self.estimator)
        self.estimator_ = self._cache(_cached_call,
                                      func_memory_level=1)(self.estimator_,
                                                           'fit',
                                                           X, y)
        return self

    def __setattr__(self, name, value):
        if name in ['memory', 'memory_level', 'estimator', 'estimator_']:
            super().__setattr__(name, value)
        else:
            if hasattr(self.estimator, name):
                setattr(self.estimator, name, value)
                if hasattr(self, 'estimator_'):
                    setattr(self.estimator_, name, value)
            else:
                super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in ['memory', 'memory_level', 'estimator', 'estimator_']:
            return super().__getattr__(name)
        else:
            if hasattr(self.estimator_, name):
                return getattr(self.estimator_, name)
            elif hasattr(self.estimator, name):
                return getattr(self.estimator, name)
            else:
                return super().__getattr__(name)

