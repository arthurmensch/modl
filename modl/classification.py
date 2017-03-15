import numbers

import keras
import mkl
import numpy as np
import tensorflow as tf
from keras.callbacks import CallbackList
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .fixes import Model

from keras.backend import set_session
from keras.engine import Input
from keras.layers import Dense, Activation, Concatenate, Dropout
from keras.regularizers import l2, l1
from nilearn._utils import CacheMixin
from scipy.linalg import lstsq
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Memory
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, gen_batches, \
    check_random_state, check_X_y


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
        X *= self.feature_importance[np.newaxis, :]
        return X

    def inverse_transform(self, Xt):
        Xt /= self.feature_importance[np.newaxis, :]
        return Xt


class LabelDropper(TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop('dataset', axis=1)
        return X


class LabelGetter(TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X[['dataset']]


def make_loadings_extractor(bases, scale_bases=True,
                            identity=False,
                            factored=False,
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
    if factored:
        pipeline = [('label_dropper', LabelDropper())]
    else:
        pipeline = []
    pipeline += [('projector', Projector(bases, n_jobs=n_jobs, memory=memory,
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
    pipeline = Pipeline(pipeline)
    if factored:
        pipeline = FeatureUnion([('data', pipeline),
                                 ('label', LabelGetter())])
    return pipeline


def make_classifier(alphas,
                    latent_dims,
                    factored=True,
                    fit_intercept=True,
                    max_iter=10,
                    activation='linear',
                    n_jobs=1,
                    penalty='l2',
                    dropout=False,
                    # Useful for non factored LR
                    multi_class='multinomial',
                    tol=1e-4,
                    train_samples=1,
                    random_state=None):
    if factored:
        classifier = FactoredLogistic(optimizer='adam',
                                      latent_dim=latent_dims[0],
                                      max_iter=max_iter,
                                      activation=activation,
                                      penalty=penalty,
                                      dropout=dropout,
                                      alpha=alphas[0],
                                      batch_size=200,
                                      validation_split=0.1,
                                      n_jobs=n_jobs)
        if len(alphas) > 1 or len(latent_dims) > 1:
            classifier.set_params(n_jobs=1)
            classifier = GridSearchCV(classifier,
                                      {'alpha': alphas,
                                       'latent_dim': latent_dims},
                                      cv=10,
                                      refit=True,
                                      verbose=1,
                                      n_jobs=n_jobs)
    else:
        if len(alphas) > 1:
            classifier = LogisticRegressionCV(solver='saga',
                                              multi_class=multi_class,
                                              fit_intercept=fit_intercept,
                                              random_state=random_state,
                                              refit=True,
                                              tol=tol,
                                              max_iter=max_iter,
                                              n_jobs=n_jobs,
                                              penalty=penalty,
                                              cv=10,
                                              verbose=True,
                                              Cs=1. / train_samples / np.array(
                                                  alphas))
        else:
            classifier = LogisticRegression(solver='saga',
                                            multi_class=multi_class,
                                            fit_intercept=fit_intercept,
                                            random_state=random_state,
                                            tol=tol,
                                            max_iter=max_iter, n_jobs=n_jobs,
                                            penalty=penalty,
                                            verbose=True,
                                            C=1. / train_samples / alphas[0])
    return classifier


class FactoredLogistic(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """
    Parameters
    ----------

    latent_dim: int,

    activation: str, Keras activation

    optimizer: str, Keras optimizer
    """

    def __init__(self, latent_dim=10, activation='linear', optimizer='adam',
                 fit_intercept=True,
                 max_iter=100, batch_size=256, n_jobs=1, alpha=0.01,
                 penalty='l2',
                 validation_split=0,
                 dropout=False,
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
        self.n_jobs = n_jobs
        self.dropout = dropout
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.validation_split = validation_split

    def fit(self, X, y):
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

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float32,
                         order="C")
        datasets = X[:, -1].astype('int')
        X = X[:, :-1]

        self.random_state = check_random_state(self.random_state)

        if 0 < self.validation_split < 1:
            (X, X_val, y, y_val, datasets,
             datasets_val) = train_test_split(X, y, datasets,
                                              stratify=y,
                                              test_size=self.validation_split,
                                              random_state=self.random_state)
            do_validation = True
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=1e-2,
                                                           patience=2,
                                                           verbose=1,
                                                           mode='auto')
            early_stopping.set_model(self)

        else:
            do_validation = False

        n_samples, n_features = X.shape

        self.datasets_ = np.unique(datasets)
        n_datasets = self.datasets_.shape[0]

        X_list = []
        y_bin_list = []
        self.classes_list_ = []

        if do_validation:
            X_val_list = []
            y_bin_val_list = []

        for this_dataset in self.datasets_:
            indices = datasets == this_dataset
            this_X = X[indices]
            this_y = y[indices]
            label_binarizer = LabelBinarizer()
            classes = np.unique(this_y)
            self.classes_list_.append(classes)
            this_y_bin = label_binarizer.fit_transform(this_y)
            y_bin_list.append(this_y_bin)
            X_list.append(this_X)

            if do_validation:
                indices = datasets_val == this_dataset
                this_X_val = X_val[indices]
                this_y_val = y_val[indices]
                this_y_bin_val = label_binarizer.fit_transform(this_y_val)
                y_bin_val_list.append(this_y_bin_val)
                X_val_list.append(this_X_val)

        self.classes_ = np.concatenate(self.classes_list_)

        # Model construction
        init_tensorflow(n_jobs=self.n_jobs)

        self.models_, self.stacked_model_ = \
            make_model(n_features,
                       classes_list=self.classes_list_,
                       alpha=self.alpha,
                       latent_dim=self.latent_dim,
                       activation=self.activation,
                       dropout=self.dropout,
                       optimizer=self.optimizer,
                       fit_intercept=self.fit_intercept,
                       penalty=self.penalty)

        # Optimization loop
        n_epochs = np.zeros(n_datasets)
        batches_list = []
        for i, (this_X, this_y_bin) in enumerate(zip(X_list, y_bin_list)):
            permutation = self.random_state.permutation(len(this_X))
            this_X[:] = this_X[permutation]
            this_y_bin[:] = this_y_bin[permutation]
            batches = gen_batches(len(this_X), self.batch_size)
            batches_list.append(batches)
        epoch_logs = {}
        if do_validation:
            early_stopping.on_train_begin()
        stop_training = False
        for model in self.models_:
            model.stop_training = False
        while not stop_training and np.min(n_epochs) < self.max_iter:
            for i, model in enumerate(self.models_):
                this_X = X_list[i]
                this_y_bin = y_bin_list[i]
                this_X_val = X_val_list[i]
                this_y_bin_val = y_bin_val_list[i]
                try:
                    batch = next(batches_list[i])
                except StopIteration:
                    this_X = X_list[i]
                    permutation = self.random_state.permutation(len(this_X))
                    this_X[:] = this_X[permutation]
                    this_y_bin[:] = this_y_bin[permutation]
                    batches_list[i] = gen_batches(len(this_X), self.batch_size)
                    n_epochs[i] += 1
                    loss, acc = model.evaluate(this_X, this_y_bin, verbose=0)
                    epoch_logs['loss'] = loss
                    epoch_logs['acc'] = acc
                    if do_validation:
                        val_loss, val_acc = model.evaluate(this_X_val,
                                                           this_y_bin_val,
                                                           verbose=0)
                        epoch_logs['val_loss'] = val_loss
                        epoch_logs['val_acc'] = val_acc
                        early_stopping.on_epoch_end(n_epochs[i], epoch_logs)
                    else:
                        val_acc = 0
                        val_loss = 0
                    print('Epoch %i, dataset %i, loss: %.4f, acc: %.4f, '
                          'val_acc: %.4f, val_loss: %.4f' %
                          (n_epochs[i], self.datasets_[i],
                           loss, acc, val_acc, val_loss))
                    batch = next(batches_list[i])
                model.train_on_batch(this_X[batch], this_y_bin[batch])
                stop_training = np.all(np.array([model.stop_training
                                                 for model in self.models_]))

    def predict_proba(self, X, dataset=None):
        if dataset is None:
            return self.stacked_model_.predict(X)
        else:
            idx = np.where(self.datasets_ == dataset)[0][0]
            return self.models_[idx].predict(X)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', order='C', dtype=np.float32)
        datasets = X[:, -1].astype('int')
        X = X[:, :-1]
        pred = np.zeros(X.shape[0], dtype='int')
        indices = np.logical_not(np.any(self.datasets_[:, np.newaxis]
                                        == datasets, axis=1))
        if np.sum(indices) > 0:
            this_X = X[indices]
            pred[indices] = self.classes_[np.argmax(self.predict_proba(this_X),
                                                    axis=1)]
        for this_dataset in self.datasets_:
            indices = datasets == this_dataset
            if np.sum(indices) > 0:
                this_X = X[indices]
                these_classes = self.classes_list_[this_dataset]
                pred[indices] = these_classes[np.argmax(self.predict_proba(
                    this_X, dataset=this_dataset), axis=1)]
        return pred


def make_model(n_features, classes_list, alpha=0.01,
               latent_dim=10, activation='linear', optimizer='adam',
               fit_intercept=True,
               dropout=False,
               penalty='l2'):
    input = Input(shape=(n_features,), name='input')

    if penalty == 'l2':
        kernel_regularizer_enc = l2(alpha)
        kernel_regularizer_sup = l2(alpha)
        activity_regularizer_enc = None
    elif penalty == 'l1':
        kernel_regularizer_enc = l1(alpha)
        kernel_regularizer_sup = None
        activity_regularizer_enc = None
    else:
        raise ValueError()
    encoded = Dense(latent_dim, activation=activation,
                    use_bias=False, kernel_regularizer=kernel_regularizer_enc,
                    activity_regularizer=activity_regularizer_enc,
                    name='encoded')(input)
    if dropout:
        encoded = Dropout(rate=0.5)(encoded)
    models = []
    supervised_list = []
    for classes in classes_list:
        n_classes = len(classes)
        supervised = Dense(n_classes, activation='linear',
                           use_bias=fit_intercept,
                           kernel_regularizer=kernel_regularizer_sup)(encoded)
        supervised_list.append(supervised)
        softmax = Activation('softmax')(supervised)
        model = Model(inputs=input, outputs=softmax)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        models.append(model)
    if len(supervised_list) > 1:
        stacked = Concatenate(axis=1)(supervised_list)
    else:
        stacked = supervised_list[0]
    softmax = Activation('softmax')(stacked)
    stacked_model = Model(inputs=input, outputs=softmax)
    stacked_model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
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
