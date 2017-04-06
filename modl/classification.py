import numbers
from copy import copy

import pandas as pd

import mkl
import numpy as np
from lightning.impl.fista import FistaClassifier
from scipy.linalg import lstsq
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, gen_batches, \
    check_random_state, check_X_y


class Projector(TransformerMixin, BaseEstimator):
    def __init__(self, basis,
                 n_jobs=1,
                 identity=False):
        self.basis = basis
        self.n_jobs = n_jobs

        self.identity = identity

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        current_n_threads = mkl.get_max_threads()
        mkl.set_num_threads(self.n_jobs)
        loadings, _, _, _ = lstsq(self.basis.T, X.T)
        mkl.set_num_threads(current_n_threads)
        loadings = loadings.T
        return loadings

    def inverse_transform(self, Xt):
        rec = Xt.dot(self.basis)
        return rec


class LabelDropper(TransformerMixin, BaseEstimator):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X = X.drop('dataset', axis=1)
        return X


class LabelGetter(TransformerMixin, BaseEstimator):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X[['dataset']]


def make_loadings_extractor(bases, scale_bases=True,
                            identity=False,
                            standardize=True, scale_importance=True,
                            memory=Memory(cachedir=None),
                            n_jobs=1, ):
    if not isinstance(bases, list):
        bases = [bases]
    if scale_bases:
        for i, basis in enumerate(bases):
            S = np.std(basis, axis=1)
            S[S == 0] = 0
            basis = basis / S[:, np.newaxis]
            bases[i] = basis

    feature_union = []
    for i, basis in enumerate(bases):
        projection_pipeline = Pipeline([('projector',
                                         Projector(basis, n_jobs=n_jobs)),
                                        ('standard_scaler',
                                         StandardScaler(
                                             with_std=standardize))],
                                       memory=memory)
        feature_union.append(('scaled_projector_%i' % i, projection_pipeline))
    if identity:
        feature_union.append(('scaled_identity',
                              StandardScaler(with_std=standardize)))

    # Weighting
    transformer_weights = {}
    sizes = np.array([basis.shape[0] for basis in bases])
    if scale_importance is None:
        scales = np.ones(len(bases))
    elif scale_importance == 'sqrt':
        scales = np.sqrt(sizes)
    elif scale_importance == 'linear':
        scales = sizes
    else:
        raise ValueError
    if identity:
        n_features = bases[0].shape[1]
        scales = np.concatenate([scales, np.array(1. / n_features)])
    for i in range(len(bases)):
        transformer_weights[
            'scaled_projector_%i' % i] = 1. / scales[i] / np.sum(1. / scales)
    if identity:
        transformer_weights[
            'scaled_identity'] = 1. / scales[-1] / np.sum(1. / scales)

    concatenated_projector = FeatureUnion(feature_union,
                                          transformer_weights=
                                          transformer_weights)

    projector = Pipeline([('label_dropper', LabelDropper()),
                          ('concatenated_projector',
                           concatenated_projector)])
    transformer = FeatureUnion([('projector', projector),
                                ('label_getter', LabelGetter())])
    return transformer


class FactoredLogistic(BaseEstimator, LinearClassifierMixin):
    """
    Parameters
    ----------

    latent_dim: int,

    activation: str, Keras activation

    optimizer: str, Keras optimizer
    """

    def __init__(self, latent_dim=10, activation='linear',
                 optimizer='adam',
                 fit_intercept=True,
                 fine_tune=True,
                 max_samples=100, batch_size=256, n_jobs=1, alpha=0.01,
                 beta=0.01,
                 dropout=0,
                 random_state=None,
                 early_stop=False,
                 verbose=0
                 ):
        self.latent_dim = latent_dim
        self.activation = activation
        self.fine_tune = fine_tune
        self.optimizer = optimizer
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.shuffle = True
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.dropout = dropout
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.beta = beta
        self.early_stop = early_stop

    def fit(self, X, y, validation_data=None, dataset_weight=None):
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
        if not isinstance(self.max_samples,
                          numbers.Number) or self.max_samples < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_samples=%r)" % self.max_iter)

        datasets = X[:, -1]
        X, y = check_X_y(X[:, :-1], y, accept_sparse='csr',
                         dtype=np.float32,
                         order="C")

        self.random_state = check_random_state(self.random_state)

        if validation_data is not None:
            X_val, y_val = validation_data
            datasets_val = X_val[:, -1]
            X_val, y_val = check_X_y(X_val[:, :-1], y_val,
                                     accept_sparse='csr', dtype=np.float32,
                                     order="C")
            do_validation = True
        else:
            do_validation = False

        n_samples, n_features = X.shape

        self.datasets_ = np.unique(datasets)

        X_dict = {}
        y_bin_dict = {}
        self.classes_dict_ = {}

        if do_validation:
            X_val_dict = {}
            y_bin_val_dict = {}

        for dataset in self.datasets_:
            indices = datasets == dataset
            this_X = X[indices]
            this_y = y[indices]
            label_binarizer = LabelBinarizer()
            classes = np.unique(this_y)
            self.classes_dict_[dataset] = classes
            this_y_bin = label_binarizer.fit_transform(this_y)
            y_bin_dict[dataset] = this_y_bin
            X_dict[dataset] = this_X

            if do_validation:
                indices = datasets_val == dataset
                this_X_val = X_val[indices]
                this_y_val = y_val[indices]
                this_y_bin_val = label_binarizer.fit_transform(this_y_val)
                y_bin_val_dict[dataset] = this_y_bin_val
                X_val_dict[dataset] = this_X_val

        # Model construction
        from keras.callbacks import EarlyStopping, History
        init_tensorflow(n_jobs=self.n_jobs)

        self.models_, self.stacked_model_ = \
            make_factored_model(n_features,
                                datasets=self.datasets_,
                                classes_dict=self.classes_dict_,
                                beta=self.beta,
                                alpha=self.alpha,
                                latent_dim=self.latent_dim,
                                activation=self.activation,
                                dropout=self.dropout,
                                optimizer=self.optimizer,
                                fit_intercept=self.fit_intercept, )
        self.n_samples_ = [len(this_X) for this_X in X_dict]

        if do_validation and self.early_stop:
            early_stoppings = {}
            for dataset in datasets:
                model = self.models_[dataset]
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-3,
                    patience=2,
                    verbose=1,
                    mode='auto')
                early_stopping.set_model(model)
                early_stoppings[dataset] = early_stopping

        self.histories_ = {}
        self.n_epochs_ = {dataset: 0 for dataset in self.datasets_}
        batches_dict = {}
        epoch_logs = {}

        for dataset in self.datasets_:
            model = self.models_[dataset]
            history = History()
            history.set_model(model)
            self.histories_[dataset] = history
        for dataset in self.datasets_:
            len_X = len(X_dict[dataset])
            permutation = self.random_state.permutation(len_X)
            X_dict[dataset] = X_dict[dataset][permutation]
            y_bin_dict[dataset] = y_bin_dict[dataset][permutation]
            batches = gen_batches(len_X, int(self.batch_size))
            batches_dict[dataset] = batches
        stop_training = False
        if do_validation:
            for history in self.histories_.values():
                history.on_train_begin()
            if self.early_stop:
                for early_stopping in early_stoppings:
                    early_stopping.on_train_begin()
        # Optimization loop
        self.n_seen_samples_ = 0

        dataset_weight = copy(dataset_weight)
        min_weight = 1000
        for weight in dataset_weight.values():
            if weight != 0:
                min_weight = min(min_weight, weight)
        for dataset in self.datasets_:
            dataset_weight[dataset] = int(dataset_weight[dataset] / min_weight)
        print(dataset_weight)

        fine_tune_n_samples = (1 - self.fine_tune) * self.max_samples
        fine_tune = False
        while not stop_training:
            if do_validation:
                for model in self.models_.values():
                    model.stop_training = stop_training
            for dataset in self.datasets_:
                model = self.models_[dataset]
                this_X = X_dict[dataset]
                this_y_bin = y_bin_dict[dataset]
                if do_validation:
                    this_X_val = X_val_dict[dataset]
                    this_y_bin_val = y_bin_val_dict[dataset]
                i = 0
                while not stop_training and i < dataset_weight[dataset]:
                    i += 1
                    try:
                        batch = next(batches_dict[dataset])
                    except StopIteration:
                        this_X = X_dict[dataset]
                        permutation = self.random_state.permutation(
                            len(this_X))
                        this_X[:] = this_X[permutation]
                        this_y_bin[:] = this_y_bin[permutation]
                        batches_dict[dataset] = gen_batches(len(this_X),
                                                            self.batch_size)
                        self.n_epochs_[dataset] += 1
                        loss, acc = model.evaluate(this_X, this_y_bin,
                                                   verbose=0)
                        epoch_logs['loss'] = loss
                        epoch_logs['acc'] = acc
                        if do_validation:
                            val_loss, val_acc = model.evaluate(this_X_val,
                                                               this_y_bin_val,
                                                               verbose=0)
                            epoch_logs['val_loss'] = val_loss
                            epoch_logs['val_acc'] = val_acc
                            if self.early_stop:
                                early_stoppings[dataset].on_epoch_end(
                                    self.n_epochs_[dataset], epoch_logs)
                            self.histories_[dataset].on_epoch_end(
                                self.n_epochs_[dataset], epoch_logs)
                        else:
                            val_acc = 0
                            val_loss = 0
                        if self.verbose:
                            print(
                                'Epoch %.5i, n_samples %i, dataset %s, loss: %.4f, acc: %.4f, '
                                'val_acc: %.4f, val_loss: %.4f' %
                                (self.n_epochs_[dataset],
                                 self.n_seen_samples_,
                                 dataset,
                                 loss, acc, val_acc, val_loss))
                        batch = next(batches_dict[dataset])
                    model.train_on_batch(this_X[batch], this_y_bin[batch])
                    self.n_seen_samples_ += batch.stop - batch.start
                    if not fine_tune and (self.latent_dim is not None and
                                    self.max_samples > self.n_seen_samples_ > fine_tune_n_samples):
                        print('Fine tuning')
                        fine_tune = True
                        for dataset in self.datasets_:
                            model.layers_by_depth[3][0].trainable = False
                            dataset_weight[dataset] = 1
                    if do_validation and self.early_stop:
                        stop_training = np.all(np.array([model.stop_training
                                                         for model in
                                                         self.models_]))
                    stop_training = (stop_training or
                                     self.n_seen_samples_ > self.max_samples)
        if do_validation and self.early_stop:
            for early_stopping in early_stoppings:
                early_stopping.on_train_end()

    def predict_proba(self, X, dataset=None):
        if dataset is None:
            return self.stacked_model_.predict(X)
        else:
            return self.models_[dataset].predict(X)

    def predict(self, X):
        datasets = X[:, -1]
        X = check_array(X[:, :-1], accept_sparse='csr',
                        order='C', dtype=np.float32)
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
                these_classes = self.classes_dict_[this_dataset]
                pred[indices] = these_classes[np.argmax(self.predict_proba(
                    this_X, dataset=this_dataset), axis=1)]
        return pred

    def score(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float32,
                         order="C")
        datasets = X[:, -1]
        X = X[:, :-1]
        accs = []
        total_weight = 0
        for dataset in self.datasets_:
            indices = datasets == dataset
            this_X = X[indices]
            this_y = y[indices]
            label_binarizer = LabelBinarizer().fit(self.classes_dict_[dataset])
            model = self.models_[dataset]
            this_y_bin = label_binarizer.fit_transform(this_y)
            loss, acc = model.evaluate(this_X, this_y_bin, verbose=0)
            weight = self.n_epochs_[dataset]
            total_weight += weight
            accs.append(acc * weight)
        return np.sum(np.array(accs)) / total_weight


def make_factored_model(n_features,
                        datasets,
                        classes_dict,
                        alpha=0.01, beta=0.01,
                        latent_dim=10, activation='linear', optimizer='adam',
                        fit_intercept=True,
                        dropout=0, ):
    from keras.engine import Input
    from keras.layers import Dense, Activation, Concatenate, Dropout
    from keras.regularizers import l2, l1_l2
    from .fixes import Model

    input = Input(shape=(n_features,), name='input')

    kernel_regularizer_enc = l1_l2(l1=beta, l2=alpha)
    kernel_regularizer_sup = l2(alpha)
    if latent_dim is not None:
        encoded = Dense(latent_dim, activation=activation,
                        use_bias=False,
                        kernel_regularizer=kernel_regularizer_enc,
                        name='latent')(input)
    else:
        encoded = input
    if dropout > 0:
        encoded = Dropout(rate=dropout, name='dropout')(encoded)
    models = {}
    supervised_dict = {}
    for dataset in datasets:
        classes = classes_dict[dataset]
        n_classes = len(classes)
        supervised = Dense(n_classes, activation='linear',
                           use_bias=fit_intercept,
                           kernel_regularizer=kernel_regularizer_sup,
                           name='supervised_%s' % dataset)(encoded)
        supervised_dict[dataset] = supervised
        softmax = Activation('softmax', name='softmax_%s' % dataset)(supervised)
        model = Model(inputs=input, outputs=softmax)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        models[dataset] = model
    if len(supervised_dict) > 1:
        stacked = Concatenate(axis=1,
                              name='concatenate')([supervised_dict[dataset] for
                                       dataset in datasets])
    else:
        stacked = supervised_dict[0]
    softmax = Activation('softmax')(stacked)
    stacked_model = Model(inputs=input, outputs=softmax)
    stacked_model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    return models, stacked_model


def init_tensorflow(n_jobs=1):
    import tensorflow as tf
    from keras.backend import set_session
    sess = tf.Session(
        config=tf.ConfigProto(
            device_count={'CPU': n_jobs},
            inter_op_parallelism_threads=n_jobs,
            intra_op_parallelism_threads=n_jobs,
            use_per_session_threads=True)
    )
    set_session(sess)
