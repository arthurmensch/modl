import functools

import numpy as np
import keras.backend as K
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs, indices):
        logits = inputs
        logits_max = K.reduce_max(logits, axis=1)
        logits -= logits_max[:, np.newaxis]
        exp_logits = K.where(indices, K.exp(logits),
                             K.zeros(K.shape(logits)))
        sum_exp_logits = K.reduce_sum(exp_logits)
        return exp_logits / sum_exp_logits

    def compute_output_shape(self, input_shape):
        return input_shape


def partial_sparse_categorical_crossentropy_with_logits(y_true,
                                                        y_pred, indices):
    logits = y_pred
    logits_max = K.reduce_max(logits, axis=1)
    logits = y_pred - logits_max[:, np.newaxis]
    exp_logits = K.where(indices, K.exp(logits), K.zeros(K.shape(logits)))

    log_prob = logits - K.log(K.reduce_sum(exp_logits))

    cross_entropy = K.reduce_sum(K.gather(log_prob, y_true))

    return cross_entropy


class HierachicalLabelMasking(Layer):
    def __init__(self, labels,
                 seed=None,
                 **kwargs):
        self.seed = seed
        self.labels = labels
        super(HierachicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        n_labels, max_depth = self.labels.shape
        adversaries = np.zeros((max_depth, n_labels, n_labels))
        for depth in range(max_depth):
            adversaries[depth] = labels[:, [depth]] == labels[:, depth]
        adversaries = K.constant(adversaries, dtype=np.bool)
        self.add_weight(K.shape(adversaries),
                        initializer=lambda: adversaries,
                        name='adversaries', trainable=False)
        super(HierachicalLabelMasking, self).build(input_shape)

    def call(self, y):
        adversaries = self.get_weights()[0]
        max_depth = K.shape(adversaries)[0]
        y_leaf = y[:, -1]
        n_samples = K.shape(y_leaf)[0]
        depths = K.random_uniform(shape=[n_samples],
                                  maxval=max_depth, dtype=np.int32)
        y_at_depth = K.gather_nd(y, indices=K.concat(
            [K.range(n_samples)[:, np.newaxis], depths[:, np.newaxis]],
            axis=1))
        indices = K.gather_nd(adversaries, indices=K.concat(
            [depths[:, np.newaxis], y_at_depth[:, np.newaxis]], axis=1))
        return indices

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.labels.shape[0]


def create_dataset():
    labels = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 2],
                       [0, 1, 3],
                       [0, 1, 4],
                       [1, 2, 5],
                       [1, 2, 6],
                       [1, 2, 7],
                       [1, 3, 8],
                       [1, 3, 9],
                       [1, 3, 10]], dtype=np.int32)
    n_samples = 10000
    y_indices = np.random.randint(11, size=n_samples)
    y = labels[y_indices]
    x = np.random.randn(n_samples, 100)
    x = x.astype(np.float32)
    return x, y, labels

X, y, labels = create_dataset()

n_features = X.shape[1]
n_classes = labels.shape[0]

latent_dim = 10
dropout_rate = 0.5

input = Input(shape=(n_features, ), name='input')
input_labels = Input((3, ), name='labels')
# Dropout
latent = Dense(latent_dim, activation='linear',
               use_bias=False, name='latent')(input)
latent_dropout = Dropout(rate=0.5, name='dropout')(latent)
logits = Dense(n_classes, activation='linear',
               use_bias=True, name='supervised')(latent_dropout)

indices = HierachicalLabelMasking(labels=labels, seed=None)(input_labels)
prob = PartialSoftmax()(logits, indices=indices)
loss = functools.partial(partial_sparse_categorical_crossentropy_with_logits,
                         indice=indices)
training_model = Model(input=[input, input_labels], output=logits)
training_model.compile('adam', loss=loss)

test_model = Model(input=input, output=prob)