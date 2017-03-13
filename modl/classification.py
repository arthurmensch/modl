import numbers
from os.path import expanduser, join

import mkl
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.callbacks import EarlyStopping, TensorBoard
from keras.engine import Input, Model
from keras.layers import Dense
from keras.models import load_model
from keras.regularizers import l2, l1, activity_l1
from nilearn._utils import CacheMixin
from scipy.linalg import lstsq
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from tempfile import NamedTemporaryFile


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
        current_n_threads = mkl.get_max_threads()
        mkl.set_num_threads(self.n_jobs)
        loadings, _, _, _ = self._cache(lstsq)(self.basis.T, X.T)
        mkl.set_num_threads(current_n_threads)
        loadings = loadings.T
        if self.identity:
            loadings = np.concatenate([loadings, X], axis=1)
        return loadings

    def inverse_transform(self, Xt):
        rec = Xt.dot(self.basis)
        if self.identity:
            rec += Xt
        return rec


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


class FactoredLogistic(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """
    Parameters
    ----------

    latent_dim: int,

    activation: str, Keras activation

    optimizer: str, Keras optimizer
    """

    def __init__(self, latent_dim=10, activation='linear', optimizer='adam',
                 max_iter=100, batch_size=256, n_jobs=1, alpha=0.01,
                 penalty='l2', log_dir=None
                 ):
        self.latent_dim = latent_dim
        self.activation = activation
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = True
        self.alpha = alpha
        self.penalty = penalty
        self.log_dir = log_dir
        self.n_jobs = n_jobs

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

        init_tensorflow(n_jobs=self.n_jobs)

        self.model_ = make_model(n_features, self.classes_.shape[0],
                                 self.alpha,
                                 self.latent_dim,
                                 self.activation, self.optimizer,
                                 penalty=self.penalty)
        callbacks = [EarlyStopping(monitor='val_acc', min_delta=1e-5,
                                   patience=20)]
        if self.log_dir is not None:
            tensorboard = TensorBoard(log_dir=join(self.log_dir,
                                                   str(self.latent_dim),
                                                   str(self.penalty), str(self.alpha))
                                      )
            callbacks.append(tensorboard)
        self.model_.fit(X, y_bin, nb_epoch=self.max_iter,
                        validation_split=0.1,
                        callbacks=callbacks,
                        verbose=False,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        sample_weight=sample_weight)

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
               latent_dim=10, activation='linear', optimizer='adam',
               penalty='l2'):
    input = Input(shape=(n_features,), name='input')
    if penalty == 'l2':
        W_regularizer_enc = l2(alpha)
        W_regularizer_sup = l2(alpha)
        activity_regularizer_enc = None
    elif penalty == 'l1':
        W_regularizer_enc = l1(alpha)
        W_regularizer_sup = None
        activity_regularizer_enc = None # activity_l1(alpha)
    else:
        raise ValueError()
    encoded = Dense(latent_dim, activation=activation,
                    bias=False, W_regularizer=W_regularizer_enc,
                    activity_regularizer=activity_regularizer_enc,
                    name='encoded')(input)
    supervised = Dense(n_classes, activation='softmax',
                       W_regularizer=W_regularizer_sup,
                       name='classifier')(encoded)
    model = Model(input=input, output=supervised, name='factored_lr')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def init_tensorflow(n_jobs=1):
    sess = tf.Session(
        config=tf.ConfigProto(
            device_count={'CPU': n_jobs},
            inter_op_parallelism_threads=n_jobs,
            intra_op_parallelism_threads=n_jobs,
            use_per_session_threads=True)
    )
    set_session(sess)
