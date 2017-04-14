import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout, Lambda
from keras.regularizers import l2
from keras.utils import to_categorical
from tensorflow.contrib.distributions import Categorical
from tensorflow.python import debug as tf_debug


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs, indices):
        logits = inputs
        logits_max = tf.reduce_max(logits, axis=1, keep_dims=True)
        logits -= logits_max
        exp_logits = tf.where(indices, tf.exp(logits),
                              tf.zeros(tf.shape(logits)))
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=1, keep_dims=True)
        return exp_logits / sum_exp_logits

    def compute_output_shape(self, input_shape):
        return input_shape


def partial_sparse_categorical_crossentropy_with_logits(y_true,
                                                        y_pred, indices):
    logits = y_pred
    logits_max = tf.reduce_max(logits, axis=1)
    logits = y_pred - logits_max[:, np.newaxis]
    exp_logits = tf.where(indices, tf.exp(logits), tf.zeros(tf.shape(logits)))

    log_prob = logits - tf.log(tf.reduce_sum(exp_logits))

    cross_entropy = tf.reduce_sum(tf.gather(log_prob, indices=y_true))

    return cross_entropy


class HierachicalLabelMasking(Layer):
    def __init__(self, label_pool,
                 depth_probs=None,
                 seed=None,
                 shared=True,
                 **kwargs):
        self.label_pool = label_pool
        self.depth_probs = depth_probs
        self.shared = shared
        self.seed = seed
        super(HierachicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_labels, self.max_depth = self.label_pool.shape
        adversaries = np.ones((self.max_depth, self.n_labels, self.n_labels))
        for depth in range(1, self.max_depth):
            adversaries[depth] = (self.label_pool[:, [depth - 1]]
                                  == self.label_pool[:, depth - 1])
            self.adversaries = tf.constant(adversaries, dtype=np.bool)
        super(HierachicalLabelMasking, self).build(input_shape)

    def call(self, y, depths=None, training=None):
        # Depths is simply ignored during training time
        y_leaf = y[:, -1]
        n_samples = tf.shape(y_leaf)[0]

        if self.shared:
            def random_depths():
                dist = Categorical(probs=self.depth_probs, dtype=np.int32,
                                   name='random_depths')
                return dist.sample(sample_shape=[n_samples, 1], seed=self.seed)

            depths = K.in_train_phase(random_depths, depths, training=training)

            indices = tf.gather_nd(self.adversaries, indices=tf.concat(
                [depths, y_leaf[:, np.newaxis]], axis=1))
        else:
            indices = []
            for i in range(self.max_depth):
                indices.append(tf.gather_nd(self.adversaries,
                                            indices=tf.concat(
                                                [np.tile(i, n_samples),
                                                 y_leaf[:, np.newaxis]],
                                                axis=1)))
            indices = tf.concat(indices[:, :, np.newaxis], axis=2)
        return indices

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.label_pool.shape[0]


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
    n_samples = 1000
    y_indices = np.random.randint(11, size=n_samples)
    y = labels[y_indices]
    x = np.random.randn(n_samples, 1000)
    x = x.astype(np.float32)
    return x, y, labels


def make_model(n_features, alpha,
               latent_dim, dropout_input,
               dropout_latent,
               optimizer,
               activation,
               seed,
               depth_probs, label_pool,
               shared_supervised):
    n_classes, max_depth = label_pool.shape
    data = Input(shape=(n_features,), name='data')
    labels = Input((3,), name='labels', dtype=np.int32)
    # Unused at test time
    depths = Input((1,), name='depths', dtype=np.int32)
    # Dropout
    dropout_data = Dropout(rate=dropout_input, seed=seed)(data)
    latent = Dense(latent_dim, activation=activation,
                   use_bias=False, name='latent',
                   kernel_regularizer=l2(alpha))(dropout_data)
    latent_dropout = Dropout(rate=dropout_latent, name='dropout',
                             seed=seed)(latent)
    if shared_supervised:
        indices = HierachicalLabelMasking(
            label_pool=label_pool,
            depth_probs=depth_probs,
            shared=True,
            seed=seed,
            name='label_masking')(
            labels, depths=depths)
        logits = Dense(n_classes, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        prob = PartialSoftmax()(logits, indices=indices)
    else:
        indices = HierachicalLabelMasking(
            label_pool=label_pool,
            depth_probs=depth_probs,
            shared=False,
            seed=seed,
            name='label_masking')(labels)
        logits = Dense((n_classes, max_depth), activation='linear',
                       use_bias=True, kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        prob_per_depth = PartialSoftmax()(logits, indices=indices)
        prob = Lambda(lambda tensor: tf.matmul(tensor, depth_probs))(
            prob_per_depth)
    model = Model(inputs=[data, labels, depths], outputs=prob)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def init_tensorflow(n_jobs=1, debug=False):
    sess = tf.Session(
        config=tf.ConfigProto(
            device_count={'CPU': n_jobs},
            inter_op_parallelism_threads=n_jobs,
            intra_op_parallelism_threads=n_jobs,
            use_per_session_threads=True)
    )
    if debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    set_session(sess)


def run():
    dropout_rate = 0.5
    latent_dim = 50

    x, y, label_pool = create_dataset()
    d = np.ones(y.shape[0])
    y_oh = to_categorical(y[:, -1])

    n_features = x.shape[1]

    model = make_model(n_features, latent_dim, dropout_rate,
                       label_pool)
    model.fit(x=[x, y, d], y=y_oh, epochs=50)
    loss = model.evaluate(x=[x, y, d], y=y_oh)
    print()
    print(loss)


if __name__ == '__main__':
    run()
