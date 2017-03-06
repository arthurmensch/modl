import copy
import warnings

from math import ceil, sqrt
from numpy import linalg

import numpy as np
from lightning.impl.sag import SAGAClassifier
from lightning.impl.sgd import SGDClassifier
from sklearn.linear_model import LogisticRegression

from modl.model_selection import MemGridSearchCV
from nilearn._utils import CacheMixin
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches, check_array

from lightning.classification import CDClassifier

from sklearn.linear_model import SGDClassifier as sklearn_SGDClassifier

_INTERCEPT_SCALER = 100


class Projector(CacheMixin, TransformerMixin, BaseEstimator):
    def __init__(self, basis,
                 n_jobs=1,
                 identity=False,
                 memory=Memory(cachedir=None),
                 memory_level=1):
        self.basis = basis
        self.n_jobs = n_jobs

        self.identity = identity

        self.memory = memory
        self.memory_level = memory_level

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        loadings = self._cache(_project)(X, self.basis,
                                         n_jobs=self.n_jobs)
        if self.identity:
            loadings = np.concatenate([loadings, X], axis=1)
        return loadings

    def inverse_transform(self, Xt):
        rec = Xt.dot(self.basis)
        if self.identity:
            rec += Xt
        return rec


def _project(X, basis, n_jobs=1):
    n_samples = X.shape[0]
    batch_size = int(ceil(n_samples / n_jobs))
    batches = gen_batches(n_samples, batch_size)
    loadings = Parallel(n_jobs=n_jobs)(
        delayed(_lstsq)(basis.T, X[batch].T) for batch in batches)
    loadings = np.hstack(loadings).T
    return loadings


def _lstsq(a, b):
    out, _, _, _ = linalg.lstsq(a, b)
    return out


class FeatureImportanceTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_importance):
        self.feature_importance = feature_importance

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X = X * self.feature_importance[np.newaxis, :]
        return X

    def inverse_transform(self, Xt):
        Xt = Xt / self.feature_importance[np.newaxis, :]
        return Xt


def make_loadings_extractor(bases, scale_bases=True,
                            identity=False,
                            standardize=True, scale_importance=True,
                            memory=Memory(cachedir=None),
                            n_jobs=1,
                            memory_level=1):
    if not isinstance(bases, list):
        bases = [bases]
    sizes = []
    for basis in bases:
        sizes.append(basis.shape[0])
    if identity:
        sizes.append(bases[0].shape[1])
    sizes = np.array(sizes)
    if scale_bases:
        for i, basis in enumerate(bases):
            S = np.std(basis, axis=1)
            S[S == 0] = 0
            basis = basis / S[:, np.newaxis]
            bases[i] = basis
    bases = np.vstack(bases)
    pipeline = [('projector', Projector(bases, n_jobs=n_jobs, memory=memory,
                                        identity=identity,
                                        memory_level=memory_level)),
                ('standard_scaler', StandardScaler(with_std=standardize))]
    if scale_importance in ['linear', 'sqrt']:
        if scale_importance == 'sqrt':
            scales = np.sqrt(sizes)
        else:
            scales = sizes
        const = 1. / np.sum(1. / scales)
        feature_importance = np.concatenate([np.ones(size) *
                                             const / scale
                                             for size, scale in
                                             zip(sizes, scales)])
        pipeline.append(('feature_importance', FeatureImportanceTransformer(
            feature_importance=feature_importance)))
    return pipeline


class OurLogisticRegressionCV(CacheMixin, BaseEstimator,
                              LinearClassifierMixin):
    def __init__(self,
                 alphas=[1],
                 cv=10,
                 max_iter=100,
                 tol=1e-4,
                 random_state=None,
                 multi_class='ovr',
                 solver='cd',
                 penalty='l2',
                 refit=False,
                 fit_intercept=False,
                 n_jobs=1,
                 memory=Memory(cachedir=None),
                 memory_level=1,
                 verbose=0):
        self.memory = memory

        self.alphas = alphas
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.penalty = penalty

        self.refit = refit
        self.multi_class = multi_class
        self.solver = solver
        self.fit_intercept = fit_intercept

        self.cv = cv

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, X, y):
        self.coef_, self.intercept_, self.classes_ = \
            self._cache(_logistic_regression,
                        ignore=['n_jobs', 'verbose'],
                        func_memory_level=1)(
                X, y, alphas=self.alphas,
                solver=self.solver,
                multi_class=self.multi_class,
                tol=self.tol, max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
                refit=self.refit,
                penalty=self.penalty,
                cv=self.cv,
                n_jobs=self.n_jobs, random_state=self.random_state,
                verbose=self.verbose)
        return self


def _logistic_regression(X, y,
                         solver='cd',
                         alphas=[1],
                         fit_intercept=False,
                         tol=1e-7,
                         max_iter=1000,
                         multi_class='ovr',
                         penalty='l2',
                         n_jobs=1,
                         cv=10,
                         verbose=0,
                         refit=False,
                         random_state=None):
    """Function to be cached"""
    early_tol = tol * 10 if refit else tol
    early_max_iter = max_iter / 2 if refit else max_iter
    n_samples = X.shape[0]
    cv = check_cv(cv, y=y, classifier=True)

    if fit_intercept:
        if solver not in ['sgd', 'sag_sklearn', 'sgd_sklearn']:
            X = np.concatenate(
                [X, np.ones((n_samples, 1)) * _INTERCEPT_SCALER], axis=1)

    if solver == 'cd':
        X = check_array(X, order='F', dtype=np.float64)
    else:
        X = check_array(X, order='C', dtype=np.float64)

    if solver == 'cd':
        if multi_class != 'ovr':
            raise ValueError('Unsupported multiclass for solver `cd`.')
        lr = CDClassifier(loss='log', penalty=penalty,
                          C=1 / n_samples,
                          multiclass=False,
                          verbose=verbose,
                          tol=early_tol,
                          max_iter=early_max_iter,
                          random_state=random_state)
        lr = MemGridSearchCV(lr,
                             {'alpha': alphas},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    elif solver == 'sag_sklearn':
        Cs = 1. / (np.array(alphas) * n_samples)
        lr = LogisticRegression(penalty=penalty, solver='sag',
                                fit_intercept=fit_intercept,
                                multi_class=multi_class,
                                tol=early_tol,
                                max_iter=early_max_iter, )
        lr = MemGridSearchCV(lr,
                             {'C': Cs},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    elif solver == 'saga_sklearn':
        Cs = 1. / (np.array(alphas) * n_samples)
        lr = LogisticRegression(penalty=penalty, solver='saga',
                                fit_intercept=fit_intercept,
                                multi_class=multi_class,
                                tol=early_tol,
                                max_iter=early_max_iter, )
        lr = MemGridSearchCV(lr,
                             {'C': Cs},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    elif solver == 'liblinear':
        if multi_class != 'ovr':
            raise ValueError('Unsupported multiclass for solver `liblinear`.')
        Cs = 1. / (np.array(alphas) * n_samples)
        lr = LogisticRegression(penalty=penalty,
                                solver='liblinear',
                                fit_intercept=fit_intercept,
                                multi_class=multi_class,
                                tol=early_tol,
                                max_iter=early_max_iter, )
        lr = MemGridSearchCV(lr,
                             {'C': Cs},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    elif solver == 'saga':
        if multi_class != 'ovr':
            raise ValueError("Unsupported multiclass != 'ovr'"
                             "for solver `saga`.")
        if penalty == 'l1':
            lr = SAGAClassifier(eta='auto',
                                loss='log',
                                alpha=0,
                                penalty='l1',
                                verbose=0,
                                tol=early_tol,
                                max_iter=early_max_iter,
                                random_state=random_state)
            lr = MemGridSearchCV(lr,
                                 {'beta': alphas},
                                 cv=cv,
                                 refit=False,
                                 keep_best=not refit,
                                 verbose=verbose,
                                 n_jobs=n_jobs)
        elif penalty == 'l2':
            lr = SAGAClassifier(eta='auto',
                                loss='log',
                                beta=0,
                                penalty=None,
                                tol=early_tol,
                                max_iter=early_max_iter,
                                verbose=0,
                                random_state=random_state)
            lr = MemGridSearchCV(lr,
                                 {'alpha': alphas},
                                 cv=cv,
                                 refit=False,
                                 keep_best=not refit,
                                 verbose=verbose,
                                 n_jobs=n_jobs)
        else:
            raise ValueError('Non valid penalty %s' % penalty)
    elif solver == 'sgd':
        lr = SGDClassifier(loss='log', multiclass=multi_class,
                           fit_intercept=fit_intercept,
                           penalty=penalty,
                           verbose=0,
                           max_iter=early_max_iter,
                           alpha=0)
        lr = MemGridSearchCV(lr,
                             {'alpha': alphas},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    elif solver == 'sgd_sklearn':
        if multi_class != 'ovr':
            raise ValueError("Unsupported multiclass != 'ovr'"
                             "for solver `sklearn_sgd`.")
        lr = sklearn_SGDClassifier(loss='log',
                                   fit_intercept=fit_intercept,
                                   penalty=penalty,
                                   verbose=0,
                                   n_iter=early_max_iter)
        lr = MemGridSearchCV(lr,
                             {'alpha': alphas},
                             cv=cv,
                             refit=False,
                             keep_best=not refit,
                             verbose=verbose,
                             n_jobs=n_jobs)
    else:
        raise ValueError('Wrong solver %s' % solver)
    lr.fit(X, y)
    if refit:
        lr = clone(lr.estimator).set_params(**lr.best_params_)
        if isinstance(lr, SGDClassifier):
            lr.set_params(max_iter=max_iter)
        elif isinstance(lr, sklearn_SGDClassifier):
            lr.set_params(n_iter=max_iter)
        else:
            lr.set_params(tol=tol, max_iter=max_iter)
        lr.fit(X, y)
        coef = lr.coef_
        classes = lr.classes_
        n_classes = classes.shape[0]
        if fit_intercept:
            if solver in ['sgd', 'sag_sklearn', 'saga_sklearn', 'sgd_sklearn']:
                intercept = lr.intercept_
            else:
                intercept = coef[:, -1] * _INTERCEPT_SCALER
                coef = coef[:, :-1]
        else:
            intercept = np.zeros(n_classes)
    else:
        classes = lr.best_estimators_[0].classes_
        n_classes = classes.shape[0]
        coef = [best_estimator.coef_[..., np.newaxis]
                for best_estimator in lr.best_estimators_]
        coef = np.mean(np.concatenate(coef, axis=2), axis=2)
        if fit_intercept:
            if solver in ['sgd', 'sag_sklearn', 'saga_sklearn', 'sgd_sklearn']:
                intercept = [best_estimator.intercept_[..., np.newaxis]
                             for best_estimator in lr.best_estimators_]
                intercept = np.mean(np.concatenate(intercept, axis=1), axis=1)
            else:
                intercept = coef[:, -1] * _INTERCEPT_SCALER
                coef = coef[:, :-1]
        else:
            intercept = np.zeros(n_classes)
    return coef, intercept, classes
