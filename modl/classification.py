import numbers
from math import ceil
from tempfile import NamedTemporaryFile

import numpy as np
from keras.engine import Input, Model
from keras.layers import Dense
from keras.models import load_model
from keras.regularizers import l2
from lightning.impl.base import BaseEstimator
from nilearn._utils import CacheMixin
from numpy import linalg
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import gen_batches
from sklearn.utils.multiclass import check_classification_targets


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
        loadings = self._cache(_project, ignore=['n_jobs'])(X, self.basis,
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
    loadings = Parallel(n_jobs=n_jobs, backend='threading')(
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


class FactoredLogistic(BaseEstimator, LinearClassifierMixin,
                       SparseCoefMixin):
    """
    Parameters
    ----------

    latent_dim: int,

    activation: str, Keras activation

    optimizer: str, Keras optimizer
    """

    def __init__(self, latent_dim=10, activation='linear',
                 optimizer='adam', max_iter=100, batch_size=256,
                 alpha=0.01,
                 ):
        self.latent_dim = latent_dim
        self.activation = activation
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = True
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        self.label_binarizer_ = LabelBinarizer()
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        y_bin = self.label_binarizer_.fit_transform(y)

        n_samples, n_features = X.shape

        self.model_ = make_model(n_features, self.classes_.shape[0],
                                 self.alpha,
                                 self.latent_dim,
                                 self.activation, self.optimizer)
        self.model_.fit(X, y_bin, nb_epoch=self.max_iter,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __setstate__(self, state):
        if 'model_' in state:
            model = state.pop('model_')
            with NamedTemporaryFile(dir='/tmp') as f:
                f.write(model)
                state['model_'] = load_model(f.name)
        super().__setstate__(state)

    def __getstate__(self):
        state = super().__getstate__()
        if hasattr(self, 'model_'):
            with NamedTemporaryFile(dir='/tmp') as f:
                self.model_.save(f.name)
                data = f.read()
            state['model_'] = data
        return state


def make_model(n_features, n_classes, alpha=0.01,
               latent_dim=10, activation='linear', optimizer='adam'):
    input = Input(shape=(n_features,), name='input')
    encoded = Dense(latent_dim, activation=activation,
                    bias=False, W_regularizer=l2(alpha),
                    name='encoded')(input)
    supervised = Dense(n_classes, activation='softmax',
                       W_regularizer=l2(alpha),
                       name='classifier')(encoded)
    model = Model(input=input, output=supervised, name='factored_lr')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
