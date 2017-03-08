import numbers
from tempfile import NamedTemporaryFile

import numpy as np
from keras.engine import Input, Model
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from lightning.impl.base import BaseClassifier, BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from keras.regularizers import l2


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
