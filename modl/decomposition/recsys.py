import numpy as np
import scipy
import scipy.sparse as sp
from numpy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_batches

from .recsys_fast import _predict
from .dict_fact_fast import _batch_weight

from math import log, sqrt, ceil


class RecsysDictFact(BaseEstimator):
    """Matrix factorization estimator based on masked online dictionary
     learning.

    Parameters
    ----------
    alpha: float,
        Regularization of the code (ridge penalty)
    n_components: int,
        Number of pipelining for the dictionary
    n_epochs:
        Number of epochs to perform over data
    learning_rate: float in [0.5, 1],
        Controls the sequence of weights in
         the update of the surrogate function
    batch_size: int,
        Number of samples to consider between each dictionary update
    dict_init: ndarray (n_components, n_cols),
        Initial dictionary
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.
    verbose: boolean,
        Degree of output the procedure will print.
    callback: callable,
        Function to be called when printing information
    detrend: boolean,
        Perform matrix pipelining on centered data, and predict data
         accordingly
    crop: 2-uple or None,
        Bounds of matrix values, useful at prediction time


    Attributes
    -------
        self.Q_: ndarray (n_components, n_cols):
            Learned dictionary
    """

    def __init__(self,
                 alpha=1.0, beta=.0,
                 n_components=30,
                 learning_rate=1.,
                 batch_size=1,
                 dict_init=None,
                 l1_ratio=0,
                 n_epochs=1,
                 random_state=None,
                 verbose=0,
                 detrend=False,
                 crop=None,
                 callback=None):
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.l1_ratio = l1_ratio
        self.dict_init = dict_init
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.detrend = detrend
        self.crop = crop

    def fit(self, X, y=None):
        """Learns a dictionary from sparse matrix X

        Parameters
        ----------
        X: csr-matrix (n_samples, n_features)
            Datset to learn the dictionary from

        """
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = check_array(X, accept_sparse='csr',
                        dtype=[np.float32, np.float64], copy=True)
        dtype = X.dtype
        n_samples, n_features = X.shape

        self.random_state = check_random_state(self.random_state)

        if self.detrend:
            self.row_mean_, self.col_mean_ = compute_biases(X,
                                                            beta=self.beta,
                                                            inplace=False)
            for i in range(X.shape[0]):
                X.data[X.indptr[i]:X.indptr[i + 1]] -= self.row_mean_[i]
            X.data -= self.col_mean_.take(X.indices, mode='clip')

        self.components_ = self.random_state.randn(self.n_components,
                                                      n_features).astype(dtype)
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        self.components_ /= S[:, np.newaxis]
        self.code_ = np.zeros((n_samples, self.n_components), dtype=dtype)
        self._refit(X)

        self.feature_freq_ = np.bincount(X.indices) / n_samples
        self.feature_n_iter_ = np.zeros(n_features, dtype=int)

        sparsity = X.nnz / n_samples / n_features
        if self.batch_size is None:
            batch_size = int(ceil(1. / sparsity))
        else:
            batch_size = self.batch_size

        self.comp_norm_ = np.zeros(self.n_components, dtype=dtype)
        self.C_ = np.zeros((self.n_components, self.n_components), dtype=dtype)
        self.B_ = np.zeros((self.n_components, n_features), dtype=dtype)

        self.n_iter_ = 0

        if self.verbose:
            log_lim = log(n_samples * self.n_epochs / batch_size, 10)
            self.verbose_iter_ = (np.logspace(0, log_lim, self.verbose,
                                              base=10) - 1) * batch_size
            self.verbose_iter_ = self.verbose_iter_.tolist()

        for i in range(self.n_epochs):
            permutation = self.random_state.permutation(n_samples)
            batches = gen_batches(n_samples, batch_size)
            for batch in batches:
                self._single_batch_fit(X, permutation[batch])
        self._refit(X)
        return self

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def _single_batch_fit(self, X, batch):
        if (self.verbose and self.verbose_iter_
                and self.n_iter_ >= self.verbose_iter_[0]):
            print('Iteration %i' % self.n_iter_)
            self.verbose_iter_ = self.verbose_iter_[1:]
            self._callback()

        batch_size = batch.shape[0]
        self.n_iter_ += batch_size
        w = _batch_weight(self.n_iter_, batch_size, self.learning_rate, 0)
        for i in batch:
            self._single_sample_update(X, i, w)
        self.C_ *= 1 - w
        self.C_ += w / batch_size * self.code_[batch].T.dot(self.code_[batch])

        subset = np.concatenate([X.indices[X.indptr[i]:X.indptr[i+1]]
                                 for i in batch])
        subset = np.unique(subset)
        self._update_dict(subset)

    # Could be made into Cython
    def _single_sample_update(self, X, i, w):
        n_features = X.shape[1]
        if X.indptr[i + 1] - X.indptr[i] != 0:
            X_subset = X.data[X.indptr[i]:X.indptr[i + 1]]
            subset = X.indices[X.indptr[i]:X.indptr[i + 1]]
            len_subset = subset.shape[0]
            reduction = n_features / len_subset
            self.feature_n_iter_[subset] += 1
            components_subset = self.components_[:, subset]
            Dx = components_subset.dot(X_subset)
            G = components_subset.dot(components_subset.T)
            G.flat[::self.n_components + 1] += self.alpha / reduction
            self.code_[i] = linalg.solve(G, Dx)
            code = self.code_[i]
            w_B = np.minimum(1,
                             w * self.n_iter_ / self.feature_n_iter_[subset])
            self.B_[:, subset] *= 1 - w_B
            self.B_[:, subset] += np.outer(code, X_subset * w_B)

    def _update_dict(self, subset):
        ger, = scipy.linalg.get_blas_funcs(('ger',), (self.C_,
                                                      self.components_))

        n_components, n_features = self.components_.shape
        components_subset = self.components_[:, subset]
        gradient_subset = self.B_[:, subset]
        gradient_subset -= self.C_.dot(components_subset)

        order = self.random_state.permutation(n_components)
        subset_norm = np.sum(components_subset ** 2, axis=1)
        self.comp_norm_ += subset_norm
        for k in order:
            gradient_subset = ger(1.0, self.C_[k], components_subset[k],
                                  a=gradient_subset, overwrite_a=True)
            if self.C_[k, k] > 1e-20:
                components_subset[k] = gradient_subset[k] / self.C_[k, k]
            # Else do not update
            norm = sqrt(np.sum(components_subset[k] ** 2))
            lim_norm = sqrt(self.comp_norm_[k])
            if norm > lim_norm:
                components_subset[k] /= norm / lim_norm
            gradient_subset = ger(-1.0, self.C_[k], components_subset[k],
                                  a=gradient_subset, overwrite_a=True)
        subset_norm = np.sum(components_subset ** 2, axis=1)
        self.comp_norm_ -= subset_norm
        self.components_[:, subset] = components_subset

    def predict(self, X):
        """ Predict values of X from internal dictionary and intercepts

        Parameters
        ----------
        X: csr-matrix (n_samples, n_features)
            Matrix holding the loci of prediction

        Returns
        -------
        X_pred: csr-matrix (n_samples, n_features)
            Matrix with the same sparsity structure as X, with predicted values
        """
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = check_array(X, accept_sparse='csr')
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.code_,
                 self.components_)

        if self.detrend:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')

        if self.crop is not None:
            out[out > self.crop[1]] = self.crop[1]
            out[out < self.crop[0]] = self.crop[0]

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        """Score prediction based on root mean squared error"""
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = check_array(X, accept_sparse='csr')
        X_pred = self.predict(X)
        return rmse(X, X_pred)

    def _refit(self, X):
        n_samples, n_features = X.shape
        for i in range(n_samples):
            X_subset = X.data[X.indptr[i]:X.indptr[i + 1]]
            subset = X.indices[X.indptr[i]:X.indptr[i + 1]]
            len_subset = subset.shape[0]
            reduction = n_features / len_subset
            components_subset = self.components_[:, subset]
            Dx = components_subset.dot(X_subset)
            G = components_subset.dot(components_subset.T)
            G.flat[::self.n_components + 1] += self.alpha / reduction
            self.code_[i] = linalg.solve(G, Dx)


def compute_biases(X, beta=0, inplace=False):
    """Row and column centering from csr matrices

    Parameters
    ----------
    X: csr-matrix (n_samples, n_features)
        Data matrix

    inplace: boolean,
        Perform centering on the input matrix

    Returns
    ---------
    X: csr-matrix (n_samples, n_features)
        Centered data
    """
    if not inplace:
        X = X.copy()
    X = sp.csr_matrix(X)

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    print('Centering data')
    average_rating = np.mean(X.data)
    for _ in range(2):
        w_u = (X.sum(axis=1).A[:, 0] + average_rating * beta) / (n_u + beta)
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / (n_m + beta)
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return acc_u, acc_m


def rmse(X_true, X_pred):
    """Root mean squared error for two sparse matrices"""
    X_true = check_array(X_true, accept_sparse='csr')
    X_pred = check_array(X_pred, accept_sparse='csr')
    mse = np.mean((X_true.data - X_pred.data) ** 2)
    return np.sqrt(mse)
