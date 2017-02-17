import copy

from math import ceil

import numpy as np
from modl.model_selection import MemGridSearchCV
from nilearn._utils import CacheMixin
from numpy import linalg as linalg
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches

from lightning.classification import CDClassifier


def _project_multi_bases(X, bases, identity=None, n_jobs=1):
    n_samples = X.shape[0]
    batches = gen_batches(n_samples, int(ceil(n_samples / n_jobs)))
    loadings = Parallel(n_jobs=n_jobs)(
        delayed(_project_multi_bases_single)(X[batch], bases,
                                             identity=identity)
        for batch in batches)
    loadings = np.vstack(loadings)
    return loadings


def _project_multi_bases_single(X, bases, identity=None):
    n_samples, n_features = X.shape
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
        offset += loadings_length
    if identity:
        loadings[:, offset:] = X
    return loadings


class MultiProjectionTransformer(CacheMixin, TransformerMixin):
    def __init__(self,
                 bases=None,
                 identity=True,
                 memory=Memory(cachedir=None),
                 memory_level=1,
                 n_jobs=1):
        self.bases = bases
        self.identity = identity

        self.n_jobs = n_jobs

        self.memory = memory
        self.memory_level = memory_level

    def fit(self, X=None, y=None):
        if not isinstance(self.bases, list):
            self.bases = [self.bases]
        n_features = np.array([basis.shape[1]
                               for basis in self.bases])
        assert (np.all(n_features == n_features[0]))
        return self

    def transform(self, X, y=None):
        loadings = self._cache(_project_multi_bases,
                               ignore=['n_jobs'])(X, self.bases,
                                                  identity=self.identity,
                                                  n_jobs=self.n_jobs)
        return loadings


class OurLogisticRegressionCV(CacheMixin, BaseEstimator):
    def __init__(self,
                 Cs=1,
                 cv=10,
                 standardize=False,
                 max_iter=100,
                 tol=1e-4,
                 random_state=None,
                 loss='l2',
                 refit=False,
                 n_jobs=1,
                 memory=Memory(cachedir=None),
                 memory_level=1,
                 verbose=0):
        self.memory = memory

        self.standardize = standardize
        self.Cs = Cs
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.loss = loss

        self.refit = refit

        self.cv = cv

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, X, y):
        self.estimator_ = self._cache(_logistic_regression,
                                      ignore=['n_jobs', 'verbose'],
                                      func_memory_level=1)(
            X, y, standardize=self.standardize, Cs=self.Cs,
            tol=self.tol, max_iter=self.max_iter,
            refit=self.refit,
            loss=self.loss,
            cv=self.cv,
            n_jobs=self.n_jobs, random_state=self.random_state,
            verbose=self.verbose)
        return self

    def predict(self, X):
        y = self.estimator_.predict(X)
        return y

    def score(self, X, y):
        score = self.estimator_.score(X, y)
        return score


def _logistic_regression(X, y,
                         standardize=False,
                         Cs=1,
                         tol=1e-7,
                         max_iter=1000,
                         loss='l2',
                         early_tol=None,
                         early_max_iter=None,
                         n_jobs=1,
                         cv=10,
                         verbose=0,
                         refit=False,
                         random_state=None):
    """Function to be cached"""
    if early_tol is None:
        early_tol = tol * 10
    if early_max_iter is None:
        early_max_iter = max_iter / 2
    if loss == 'l2':
        lr = LogisticRegressionCV(multi_class='multinomial', penalty='l2',
                                  Cs=Cs,
                                  cv=cv,
                                  refit=refit,
                                  solver='sag',
                                  tol=tol,
                                  max_iter=max_iter,
                                  verbose=verbose,
                                  random_state=random_state)
    elif loss == 'l1':
        lr = CDClassifier(loss='log', penalty='l1',
                          verbose=verbose,
                          tol=early_tol if refit else tol,
                          max_iter=early_max_iter if refit else
                          max_iter,
                          random_state=random_state)
        cv = check_cv(cv, y=y, classifier=True)
        lr = MemGridSearchCV(lr,
                             {'C': Cs},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    else:
        raise NotImplementedError()
    if standardize:
        sc = StandardScaler()
        estimator = Pipeline([('standard_scaler', sc),
                              ('logistic_regression', lr)])
    else:
        estimator = lr
    estimator.fit(X, y)
    if refit:
        if loss == 'l2':
            pass
        else:
            lr = clone(lr.estimator).set_params(**lr.best_params_)
            lr.set_params(tol=tol, max_iter=max_iter)
        if standardize:
            estimator = Pipeline([('standard_scaler',
                                   sc),
                                  ('logistic_regression', lr)])
        else:
            estimator = lr
    else:
        if loss == 'l2':
            # Averaging has already been made within estimator
            pass
        else:
            coef = [best_estimator.coef_[..., np.newaxis]
                    for best_estimator in lr.best_estimators_]
            intercept = [best_estimator.intercept_[..., np.newaxis]
                         for best_estimator in lr.best_estimators_]
            label_binarizer = lr.best_estimators_[0].label_binarizer_
            lr = clone(lr.estimator).set_params(**lr.best_params_)
            # External fit
            lr.coef_ = np.mean(np.concatenate(coef, axis=2), axis=2)
            lr.intercept_ = np.mean(np.concatenate(intercept, axis=1),
                                    axis=1)
            lr.label_binarizer_ = label_binarizer
            # XXX There might be a bug when some labels disappear in the split
    if standardize:
        estimator = Pipeline([('standard_scaler',
                               sc),
                              ('logistic_regression', lr)])
    else:
        estimator = lr
    if refit:
        estimator.fit(X, y)
    return estimator
