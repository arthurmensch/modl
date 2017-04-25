import numbers
from copy import copy

import numpy as np
from scipy.linalg import lstsq
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Memory
from sklearn.linear_model.base import LinearClassifierMixin
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
        loadings, _, _, _ = lstsq(self.basis.T, X.T)
        loadings = loadings.T
        return loadings

    def inverse_transform(self, Xt):
        rec = Xt.dot(self.basis)
        return rec


class LabelDropper(TransformerMixin, BaseEstimator):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X = X.drop('model_indices', axis=1)
        return X


class LabelGetter(TransformerMixin, BaseEstimator):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X[['model_indices']]


def make_loadings_extractor(bases, scale_bases=True,
                            identity=False,
                            standardize=True,
                            scale_importance=True,
                            memory=Memory(cachedir=None),
                            handle_indices=False,
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
                                         Projector(basis, n_jobs=1)),
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
    sizes = [basis.shape[0] for basis in bases]
    if identity:
        n_features = bases[0].shape[1]
        sizes += n_features
    sizes = np.array(sizes)
    if scale_importance is None:
        scales = np.ones(len(sizes))
    elif scale_importance == 'sqrt':
        scales = np.sqrt(sizes)
    elif scale_importance == 'linear':
        scales = sizes
    else:
        raise ValueError
    for i in range(len(bases)):
        transformer_weights['scaled_projector_%i' % i] = 1. / scales[i]
    if identity:
        transformer_weights['scaled_identity'] = 1. / scales[-1]

    concatenated_projector = FeatureUnion(feature_union,
                                          transformer_weights=
                                          transformer_weights, n_jobs=n_jobs)

    if handle_indices:
        projector = Pipeline([('label_dropper', LabelDropper()),
                              ('concatenated_projector',
                               concatenated_projector)])
        transformer = FeatureUnion([('projector', projector),
                                    ('label_getter', LabelGetter())])
    else:
        return concatenated_projector
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
                 dropout_input=0,
                 dropout_latent=0,
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
        self.dropout_input = dropout_input
        self.dropout_latent = dropout_latent
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.beta = beta
        self.early_stop = early_stop

    def fit(self, X, y, validation_data=None, model_weight=None,
            model_indices=None, latent_weights=None):
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

        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float32,
                         order="C")
        model_indices = check_array(model_indices, dtype=None, ensure_2d=False)

        self.random_state = check_random_state(self.random_state)

        if validation_data is not None:
            X_val, y_val, model_indices_val = validation_data
            X_val, y_val = check_X_y(X_val, y_val,
                                     accept_sparse='csr', dtype=np.float32,
                                     order="C")
            model_indices_val = check_array(model_indices_val, dtype=None,
                                            ensure_2d=False)

            do_validation = True
        else:
            do_validation = False

        n_samples, n_features = X.shape

        self.model_indices_ = np.unique(model_indices)

        X_dict = {}
        y_bin_dict = {}
        self.classes_dict_ = {}

        if do_validation:
            X_val_dict = {}
            y_bin_val_dict = {}

        for model_index in self.model_indices_:
            these_indices = model_indices == model_index
            this_X = X[these_indices]
            this_y = y[these_indices]
            label_binarizer = LabelBinarizer()
            classes = np.unique(this_y)
            self.classes_dict_[model_index] = classes
            this_y_bin = label_binarizer.fit_transform(this_y)
            y_bin_dict[model_index] = this_y_bin
            X_dict[model_index] = this_X

            if do_validation:
                these_indices = model_indices_val == model_index
                this_X_val = X_val[these_indices]
                this_y_val = y_val[these_indices]
                this_y_bin_val = label_binarizer.fit_transform(this_y_val)
                y_bin_val_dict[model_index] = this_y_bin_val
                X_val_dict[model_index] = this_X_val

        # Model construction
        from keras.callbacks import EarlyStopping, History
        init_tensorflow(n_jobs=self.n_jobs)

        self.models_, self.encoder_ = \
            make_factored_model(n_features,
                                model_indices=self.model_indices_,
                                classes_dict=self.classes_dict_,
                                beta=self.beta,
                                alpha=self.alpha,
                                latent_dim=self.latent_dim,
                                activation=self.activation,
                                dropout_latent=self.dropout_latent,
                                dropout_input=self.dropout_input,
                                optimizer=self.optimizer,
                                fit_intercept=self.fit_intercept, )
        if latent_weights is not None:
            self.encoder_.get_layer('latent').set_weights(latent_weights)
            for model in self.models_.values():
                model.get_layer('latent').set_weights(latent_weights)
                model.get_layer('latent').trainable = False
                model.compile(optimizer=model.optimizer,
                              loss=model.loss,
                              metrics=model.metrics)
        self.n_samples_ = [len(this_X) for this_X in X_dict]

        if do_validation and self.early_stop:
            early_stoppings = {}
            for model_index in self.model_indices_:
                model = self.models_[model_index]
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-3,
                    patience=10,
                    verbose=1,
                    mode='auto')
                early_stopping.set_model(model)
                early_stoppings[model_index] = early_stopping

        self.histories_ = {}
        self.n_epochs_ = {model_index: 0 for model_index in
                          self.model_indices_}
        batches_dict = {}
        epoch_logs = {}

        for model_index in self.model_indices_:
            model = self.models_[model_index]
            history = History()
            history.set_model(model)
            self.histories_[model_index] = history
        for model_index in self.model_indices_:
            len_X = len(X_dict[model_index])
            permutation = self.random_state.permutation(len_X)
            X_dict[model_index] = X_dict[model_index][permutation]
            y_bin_dict[model_index] = y_bin_dict[model_index][permutation]
            batches = gen_batches(len_X, int(self.batch_size))
            batches_dict[model_index] = batches
        stop_training = False
        for history in self.histories_.values():
            history.on_train_begin()
        if do_validation:
            if self.early_stop:
                for early_stopping in early_stoppings.values():
                    early_stopping.on_train_begin()
        # Optimization loop
        self.n_seen_samples_ = 0

        model_weight = copy(model_weight)
        min_weight = 1000
        for weight in model_weight.values():
            if weight != 0:
                min_weight = min(min_weight, weight)
        for model_index in self.model_indices_:
            model_weight[model_index] = int(model_weight[model_index]
                                            / min_weight)

        fine_tune_n_samples = (1 - self.fine_tune) * self.max_samples
        fine_tune = False
        while not stop_training:
            if do_validation:
                for model in self.models_.values():
                    model.stop_training = stop_training
            self.random_state.shuffle(self.model_indices_)
            for model_index in self.model_indices_:
                model = self.models_[model_index]
                this_X = X_dict[model_index]
                this_y_bin = y_bin_dict[model_index]
                if do_validation:
                    this_X_val = X_val_dict[model_index]
                    this_y_bin_val = y_bin_val_dict[model_index]
                i = 0
                while not stop_training and i < model_weight[model_index]:
                    i += 1
                    try:
                        batch = next(batches_dict[model_index])
                    except StopIteration:
                        this_X = X_dict[model_index]
                        permutation = self.random_state.permutation(
                            len(this_X))
                        this_X[:] = this_X[permutation]
                        this_y_bin[:] = this_y_bin[permutation]
                        batches_dict[model_index] = gen_batches(len(this_X),
                                                                self.batch_size)
                        self.n_epochs_[model_index] += 1
                        loss, acc = model.evaluate(this_X, this_y_bin,
                                                   verbose=0)
                        epoch_logs['loss'] = loss
                        epoch_logs['acc'] = acc
                        if self.encoder_ is not None:
                            latent = self.encoder_.get_layer(
                                'latent').get_weights()[0][:10]
                            epoch_logs['latent'] = latent
                        if do_validation:
                            val_loss, val_acc = model.evaluate(this_X_val,
                                                               this_y_bin_val,
                                                               verbose=0)
                            epoch_logs['val_loss'] = val_loss
                            epoch_logs['val_acc'] = val_acc
                            if self.early_stop:
                                early_stoppings[model_index].on_epoch_end(
                                    self.n_epochs_[model_index], epoch_logs)
                        else:
                            val_acc = 0
                            val_loss = 0
                        if self.verbose and self.n_epochs_[model_index] % 5 == 0:
                            print(
                                'Epoch %.5i, n_samples %i, model_index %s, '
                                'loss: %.4f, acc: %.4f, '
                                'val_acc: %.4f, val_loss: %.4f' %
                                (self.n_epochs_[model_index],
                                 self.n_seen_samples_,
                                 model_index,
                                 loss, acc, val_acc, val_loss))
                        self.histories_[model_index].on_epoch_end(
                            self.n_epochs_[model_index], epoch_logs)
                        batch = next(batches_dict[model_index])
                    model.train_on_batch(this_X[batch], this_y_bin[batch])
                    self.n_seen_samples_ += batch.stop - batch.start
                    if not fine_tune and (self.latent_dim is not None and
                                                      self.max_samples > self.n_seen_samples_ >
                                                  fine_tune_n_samples):
                        print('Fine tuning')
                        fine_tune = True
                        for model_index, model in self.models_.items():
                            model.get_layer('latent').trainable = False
                            model.compile(optimizer=model.optimizer,
                                          loss=model.loss,
                                          metrics=model.metrics)
                            model_weight[model_index] = 1
                    if do_validation and self.early_stop:
                        stop_training = np.all(np.array([model.stop_training
                                                         for model in
                                                         self.models_.values()]))
                    stop_training = (stop_training or
                                     self.n_seen_samples_ > self.max_samples)
        if do_validation and self.early_stop:
            for early_stopping in early_stoppings.values():
                early_stopping.on_train_end()

    def transform(self, X):
        '''Project data onto latent space'''
        if self.encoder_ is not None:
            coefs = self.encoder_.get_layer('latent').get_weights()[0]
            Xt = X.dot(coefs)
            return Xt
        else:
            return X

    def predict_proba(self, X, model_index):
        return self.models_[model_index].predict(X)

    def predict(self, X, model_indices=None):
        X = check_array(X, accept_sparse='csr',
                        order='C', dtype=np.float32)
        model_indices = check_array(model_indices, dtype=None, ensure_2d=False)
        pred = np.zeros(X.shape[0], dtype='int')
        for model_index in self.model_indices_:
            these_indices = model_indices == model_index
            if np.sum(these_indices) > 0:
                this_X = X[these_indices]
                these_classes = self.classes_dict_[model_index]
                pred[these_indices] = these_classes[
                    np.argmax(self.predict_proba(
                        this_X, model_index=model_index), axis=1)]
        return pred

    def score(self, X, y, model_indices=None, **kwargs):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float32,
                         order="C")
        accs = []
        total_weight = 0
        for model_index in self.model_indices_:
            these_indices = model_indices == model_index
            this_X = X[these_indices]
            this_y = y[these_indices]
            label_binarizer = LabelBinarizer().fit(
                self.classes_dict_[model_index])
            model = self.models_[model_index]
            this_y_bin = label_binarizer.fit_transform(this_y)
            loss, acc = model.evaluate(this_X, this_y_bin, verbose=0)
            weight = self.n_epochs_[model_index]
            total_weight += weight
            accs.append(acc * weight)
        return np.sum(np.array(accs)) / total_weight


def make_factored_model(n_features,
                        model_indices,
                        classes_dict,
                        alpha=0.01, beta=0.01,
                        latent_dim=10, activation='linear', optimizer='adam',
                        fit_intercept=True,
                        dropout_latent=0, dropout_input=0):
    from keras.engine import Input
    from keras.layers import Dense, Dropout
    from keras.regularizers import l2, l1_l2
    from .fixes import Model

    input = Input(shape=(n_features,), name='input')
    if dropout_input > 0:
        input_dropout = Dropout(rate=dropout_input, name='dropout_input')(
            input)
    else:
        input_dropout = input
    kernel_regularizer_enc = l1_l2(l1=beta, l2=alpha)
    kernel_regularizer_sup = l2(alpha)
    if latent_dim is not None:
        encoded = Dense(latent_dim, activation=activation,
                        use_bias=False,
                        kernel_regularizer=kernel_regularizer_enc,
                        name='latent')(input_dropout)
    else:
        encoded = input_dropout
    if dropout_latent > 0:
        encoded = Dropout(rate=dropout_latent, name='dropout_latent')(encoded)
    models = {}
    for model_index in model_indices:
        classes = classes_dict[model_index]
        n_classes = len(classes)
        if n_classes == 2:
            n_classes = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = 'softmax'
            loss = 'categorical_crossentropy'
        supervised = Dense(n_classes, activation=activation,
                           use_bias=fit_intercept,
                           kernel_regularizer=kernel_regularizer_sup,
                           name='supervised_%s' % model_index)(encoded)
        model = Model(inputs=input, outputs=supervised)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        models[model_index] = model

    if latent_dim is not None:
        encoder = Model(inputs=input, outputs=encoded)
    else:
        encoder = None
    return models, encoder


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
