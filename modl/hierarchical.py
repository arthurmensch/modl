import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout
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
        logits_max = tf.reduce_max(logits, axis=1)
        logits -= logits_max[:, np.newaxis]
        exp_logits = tf.where(indices, tf.exp(logits),
                              tf.zeros(tf.shape(logits)))
        sum_exp_logits = tf.reduce_sum(exp_logits)
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
                 depth_probs,
                 random_at_training,
                 seed,
                 **kwargs):
        self.labels = label_pool
        self.depth_probs = depth_probs
        self.random_at_training = random_at_training
        self.seed = seed
        super(HierachicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        n_labels, max_depth = self.labels.shape
        adversaries = np.ones((max_depth, n_labels, n_labels))
        for depth in range(1, max_depth):
            adversaries[depth] = self.labels[:, [depth - 1]] == self.labels[:,
                                                                depth - 1]
        self.adversaries = tf.constant(adversaries, dtype=np.bool)
        self.max_depth = adversaries.shape[0]
        super(HierachicalLabelMasking, self).build(input_shape)

    def call(self, y, depths, training=None):
        # Depths is simply ignored during training time
        y_leaf = y[:, -1][:, np.newaxis]
        n_samples = tf.shape(y_leaf)[0]

        def random_depths():
            dist = Categorical(probs=self.depth_probs, dtype=np.int32,
                               name='random_depths')
            return dist.sample(sample_shape=[n_samples, 1], seed=self.seed)

        depths = K.in_train_phase(random_depths
                                  if self.random_at_training
                                  else depths, depths, training=training)

        # y_at_depth = tf.gather_nd(y, indices=tf.concat(
        #     [tf.range(n_samples)[:, np.newaxis], depths], axis=1))

        indices = tf.gather_nd(self.adversaries, indices=tf.concat(
            [depths, y_leaf], axis=1))
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
    n_classes = label_pool.shape[0]
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
            random_at_training=True,
            seed=seed,
            name='label_masking')(
            labels, depths=depths)
        logits = Dense(n_classes, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        prob = PartialSoftmax()(logits, indices=indices)
    else:
        dist = Categorical(probs=depth_probs, dtype=np.int32,
                           name='random_depths')
        depth = dist.sample(sample_shape=[1], seed=seed)
        logits = []
        prob_list = []
        for i in range(3):
            logits[i] = Dense(n_classes, activation='linear',
                              use_bias=True,
                              kernel_regularizer=l2(alpha),
                              name='supervised')(latent_dropout)
            prob_list[i] = PartialSoftmax()(logits, indices=indices)
        prob = tf.case({tf.equal(depth, i): lambda: prob_list[i]
                        for i in range(3)})
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
