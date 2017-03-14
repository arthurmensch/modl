import numbers

import mkl
import numpy as np
import tensorflow as tf

from .fixes import Model

from keras.backend import set_session
from keras.engine import Input, Merge, merge
from keras.layers import Dense, Activation
from keras.regularizers import l2, l1
from nilearn._utils import CacheMixin
from scipy.linalg import lstsq
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Memory
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, gen_batches, \
    check_random_state


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
                 label_to_dataset=None,
                 penalty='l2', log_dir=None,
                 random_state=None,
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
        self.label_to_dataset = label_to_dataset
        self.random_state = random_state

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

        X = check_array(X, accept_sparse='csr', dtype=np.float32,
                        order="C")

        n_samples, n_features = X.shape
        y = check_array(y, ensure_2d=True)
        assert (y.shape[1] == 2)

        self.datasets_ = np.unique(y[:, 0])
        n_datasets = self.datasets_.shape[0]

        X_list = []
        y_bin_list = []
        self.classes_list_ = []

        for dataset in self.datasets_:
            indices = y[:, 0] == dataset
            this_X = X[indices]
            this_y = y[indices, 1]
            label_binarizer = LabelBinarizer()
            classes = np.unique(this_y)
            self.classes_list_.append(classes)
            this_y_bin = label_binarizer.fit_transform(this_y)
            y_bin_list.append(this_y_bin)
            X_list.append(this_X)

        self.classes_ = np.concatenate(self.classes_list_)

        self.random_state = check_random_state(self.random_state)

        # Model construction
        init_tensorflow(n_jobs=self.n_jobs)

        self.models_, self.stacked_model_ =\
            make_model(n_features, self.classes_list_, self.alpha,
                       self.latent_dim, self.activation,
                       self.optimizer, penalty=self.penalty)

        # Optimization loop
        n_epochs = np.zeros(n_datasets)
        batches_list = []
        for i, this_X in enumerate(X_list):
            self.random_state.shuffle(this_X)
            batches = gen_batches(len(this_X), self.batch_size)
            batches_list.append(batches)
        while np.min(n_epochs) < self.max_iter:
            for i, model in enumerate(self.models_):
                this_X = X_list[i]
                this_y_bin = y_bin_list[i]
                try:
                    batch = next(batches_list[i])
                except StopIteration:
                    this_X = X_list[i]
                    self.random_state.shuffle(this_X)
                    batches_list[i] = gen_batches(len(this_X), self.batch_size)
                    n_epochs[i] += 1
                    print('Epoch %s' % n_epochs)
                    batch = next(batches_list[i])
                model.train_on_batch(this_X[batch], this_y_bin[batch])

    def predict_proba(self, X, dataset=None):
        if dataset is None:
            return self.stacked_model_.predict(X)
        else:
            idx = np.where(self.datasets_ == dataset)[0][0]
            return self.models_[idx].predict(X)

    def predict(self, X, dataset=None):
        if dataset is None:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        else:
            pred = np.zeros(X.shape[0], dtype='int')
            for this_dataset in self.datasets_:
                indices = dataset == this_dataset
                this_X = X[indices]
                these_classes = self.classes_list_[this_dataset]
                pred[indices] = these_classes[np.argmax(self.predict_proba(
                    this_X, dataset=this_dataset), axis=1)]
            return pred


def make_model(n_features, classes_list, alpha=0.01,
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
        activity_regularizer_enc = None
    else:
        raise ValueError()
    encoded = Dense(latent_dim, activation=activation,
                    bias=False, W_regularizer=W_regularizer_enc,
                    activity_regularizer=activity_regularizer_enc,
                    name='encoded')(input)
    models = []
    supervised_list = []
    for classes in classes_list:
        n_classes = len(classes)
        supervised = Dense(n_classes, activation='linear',
                           W_regularizer=W_regularizer_sup)(encoded)
        supervised_list.append(supervised)
        softmax = Activation('softmax')(supervised)
        model = Model(input=input, output=softmax)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        models.append(model)
    if len(supervised_list) > 1:
        stacked = merge(supervised_list, mode='concat', concat_axis=1)
    else:
        stacked = supervised_list[0]
    softmax = Activation('softmax')(stacked)
    stacked_model = Model(input=input, output=softmax)
    return models, stacked_model


def init_tensorflow(n_jobs=1):
    sess = tf.Session(
        config=tf.ConfigProto(
            device_count={'CPU': n_jobs},
            inter_op_parallelism_threads=n_jobs,
            intra_op_parallelism_threads=n_jobs,
            use_per_session_threads=True)
    )
    set_session(sess)
