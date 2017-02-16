import copy

from math import ceil

import numpy as np
from nilearn._utils import CacheMixin
from numpy import linalg as linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import gen_batches


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


class L2LogisticRegressionCV(CacheMixin, BaseEstimator):
    def __init__(self,
                 C=1,
                 standardize=False,
                 max_iter=100,
                 tol=1e-4,
                 random_state=None,
                 ensemble=False,
                 n_jobs=1,
                 memory=Memory(cachedir=None),
                 memory_level=1,
                 verbose=0):
        self.memory = memory

        self.standardize = standardize
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.ensemble = ensemble

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, X, y):
        self.lr_ = self._cache(_logistic_regression,
                               ignore=['n_jobs', 'verbose']) \
            (X, y, standardize=self.standardize, C=self.C,
             tol=self.tol, max_iter=self.max_iter,
             ensemble=self.ensemble,
             n_jobs=self.n_jobs, random_state=self.random_state,
             verbose=self.verbose)
        return self

    def predict(self, X):
        y = self.lr_.predict(X)
        return y


def _logistic_regression(X, y,
                         standardize=False,
                         C=1,
                         tol=1e-7,
                         max_iter=1000,
                         early_tol=None,
                         early_max_iter=None,
                         test_size=0.1,
                         n_jobs=1,
                         verbose=0,
                         ensemble=False,
                         random_state=None):
    """Function to be cached"""
    lr = LogisticRegression(multi_class='multinomial', penalty='l2',
                            C=C,
                            solver='sag', tol=tol,
                            max_iter=max_iter, verbose=verbose,
                            random_state=random_state)
    if standardize:
        sc = StandardScaler()
        lr = Pipeline([('standard_scaler', sc), ('logistic_regression', lr)])
    if hasattr(C, '__iter__'):
        # TODO Blast this pipeline shit
        if ensemble:
            cv = ShuffleSplit(test_size=test_size)
            best_lr_list = []
            for train, test in cv.split(X, y):
                res = Parallel(n_jobs=n_jobs)(delayed(_single_fit_lr)
                                            (X, lr, test, this_C, train, y)
                                        for this_C in C)
                scores, lr_list = zip(*res)
                scores = np.array(scores)
                i = np.argmax(scores)
                best_lr = lr_list[i]
                best_lr_list.append(best_lr)
            coef = np.concatenate([this_lr.steps[1][1].coef_[..., np.newaxis]
                                   for this_lr in best_lr_list], axis=2)
            coef = np.mean(coef, axis=2)
            intercept = np.concatenate([this_lr.steps[1][1].intercept_[..., np.newaxis]
                                        for this_lr in best_lr_list], axis=1)
            intercept = np.mean(intercept, axis=1)
            n_iter = np.array([this_lr.steps[1][1].n_iter_ for this_lr in best_lr_list])
            n_iter = np.mean(n_iter)
            lr.steps[0][1].fit(X)
            lr.steps[1][1].n_iter_ = n_iter
            lr.steps[1][1].intercept_ = intercept
            lr.steps[1][1].coef_ = coef
            lr.steps[1][1].classes_ = best_lr_list[0].steps[1][1].classes_
        else:
            if early_tol is None:
                early_tol = tol * 1e2
            if early_max_iter is None:
                early_max_iter = max_iter / 10
            lr.set_params(logistic_regression__tol=early_tol,
                          logistic_regression__max_iter=early_max_iter)
            grid_lr = GridSearchCV(lr,
                                   {'logistic_regression__C': C},
                                   cv=ShuffleSplit(test_size=test_size),
                                   refit=False,
                                   verbose=verbose,
                                   n_jobs=n_jobs)
            grid_lr.fit(X, y)
            best_params = grid_lr.best_params_
            lr.set_params(**best_params)
            lr.set_params(logistic_regression__tol=tol,
                          logistic_regression__max_iter=max_iter)
            lr.fit(X, y)
    return lr


def _single_fit_lr(X, lr, test, this_C, train, y):
    this_lr = copy.deepcopy(lr)
    this_lr.set_params(logistic_regression__C=this_C)
    this_lr.fit(X[train], y[train])
    this_score = this_lr.score(X[test], y[test])
    return this_score, this_lr


class fMRITaskClassifier(CacheMixin):
    def __init__(self,
                 transformer,
                 C=1,
                 standardize=False,
                 max_iter=100,
                 tol=1e-4,
                 random_state=None,
                 n_jobs=1,
                 memory=Memory(cachedir=None),
                 memory_level=1):
        self.transformer = transformer
        self.memory = memory

        self.standardize = standardize
        self.C = C
        self.tol = tol
        self.max_iter = max_iter,
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.memory = memory
        self.memory_level = memory_level

    def fit(self, imgs=None, labels=None, confounds=None):
        X = self.transformer.transform(imgs, confounds=confounds)
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(labels)
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
