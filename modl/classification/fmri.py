import numpy as np

from nilearn._utils import CacheMixin
import numpy.linalg as linalg
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def project_multi_bases(X, bases, identity=None):
    n_samples, n_features = X.shape
    S = np.sqrt((X ** 2).sum(axis=1))
    S[S == 0] = 1
    X /= S[:, np.newaxis]
    n_samples = X.shape[0]
    n_loadings = np.sum(np.array([basis.shape[0]
                                        for basis in bases]))
    if identity:
        n_loadings += n_features
    loadings = np.empty((n_samples, n_loadings), order='F')
    offset = 0
    for basis in bases:
        S = np.sqrt((basis ** 2).sum(axis=1))
        S[S == 0] = 1
        basis = basis / S[:, np.newaxis]
        loadings_length = basis.shape[0]
        these_loadings = linalg.lstsq(basis.T, X.T)[0].T
        S = np.sqrt((these_loadings ** 2).sum(axis=1))
        S[S == 0] = 1
        these_loadings /= S[:, np.newaxis]
        loadings[:, offset:offset + loadings_length] = these_loadings
    if identity:
        loadings[:, offset:] = X
    return loadings


class MultiProjectionTransformer(CacheMixin, TransformerMixin):
    def __init__(self, bases=None,
                 identity=True,
                 memory=Memory(cachedir=None),
                 memory_level=1):
        self.bases = bases
        self.identity = identity

        self.memory = memory
        self.memory_level = memory_level

    def fit(self, X=None, y=None):
        if not isinstance(self.bases, list):
            self.bases = [self.bases]
        n_features = np.array([basis.shape[1]
                             for basis in self.bases])
        assert(np.all(n_features == n_features[0]))
        return self

    def transform(self, X, y=None, confounds=None):
        loadings = self._cache(project_multi_bases)(X, self.bases,
                                                    identity=self.identity)
        return loadings


class L1BaggedLogisticRegression(CacheMixin):
    def __init__(self,
                 C=1,
                 standardize=False,
                 max_iter=100,
                 tol=1e-4,
                 random_state=None,
                 n_jobs=1,
                 memory=Memory(cachedir=None),
                 memory_level=1):
        self.memory = memory

        self.standardize = standardize
        self.C = C
        self.tol = tol
        self.max_iter = max_iter,
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.memory = memory
        self.memory_level = memory_level

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        self.lr_ = self._cache(_logistic_regression,
                               ignore=['n_jobs'])(X, y,
                                                  standardize=self.standardize,
                                                  C=self.C,
                                                  tol=self.tol,
                                                  max_iter=self.max_iter,
                                                  n_jobs=self.n_jobs,
                                                  random_state=self.random_state)

    def predict(self, imgs, confounds=None):
        X = self.transformer(imgs, confounds=confounds)
        y = self.lr_.predict(X)
        labels = self.le_.inverse_transform(y)
        return labels


def _logistic_regression(X_train, y_train,
                         standardize=False,
                         C=1,
                         tol=1e-7,
                         max_iter=1000,
                         early_tol=None,
                         early_max_iter=None,
                         test_size=0.1,
                         n_jobs=1,
                         verbose=0,
                         random_state=None):
    """Function to be cached"""
    if early_tol is None:
        early_tol = tol * 1e2
    if early_max_iter is None:
        early_max_iter = max_iter / 10

    lr = LogisticRegression(multi_class='multinomial', penalty='l2',
                            C=C,
                            solver='sag', tol=early_tol,
                            max_iter=early_max_iter, verbose=verbose,
                            random_state=random_state)
    if standardize:
        sc = StandardScaler()
        lr = Pipeline([('standard_scaler', sc), ('logistic_regression', lr)])
    if hasattr(C, '__iter__'):
        grid_lr = GridSearchCV(lr,
                               {'logistic_regression__C': C},
                               cv=ShuffleSplit(test_size=test_size),
                               refit=False,
                               n_jobs=n_jobs)
    grid_lr.fit(X_train, y_train)
    best_params = grid_lr.best_params_
    lr.set_params(**best_params)
    lr.set_params(logistic_regression__tol=tol,
                  logistic_regression__max_iter=max_iter)
    lr.fit(X_train, y_train)
    return lr
