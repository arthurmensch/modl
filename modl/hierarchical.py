import numpy as np
import tensorflow as tf
from keras.backend import set_session, in_train_phase
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout, RepeatVector, Lambda, Merge, Reshape
from keras.regularizers import l2
from sklearn.utils import check_array
from tensorflow.python import debug as tf_debug


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs):
        logits, mask = inputs
        logits_max = tf.reduce_max(logits, axis=2, keep_dims=True)
        logits -= logits_max
        if mask is not None:
            exp_logits = tf.where(mask, tf.exp(logits),
                                  tf.zeros(tf.shape(logits)))
        else:
            exp_logits = tf.exp(logits)
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=2, keep_dims=True)
        return exp_logits / sum_exp_logits

    def compute_output_shape(self, input_shape):
        return input_shape


class HierachicalLabelMasking(Layer):
    def __init__(self,
                 adversaries,
                 **kwargs):
        adversaries = check_array(adversaries, dtype=np.bool, allow_nd=True)
        self.n_depths, self.n_labels, _ = adversaries.shape
        self.adversaries = tf.convert_to_tensor(adversaries, dtype=np.bool)
        super(HierachicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HierachicalLabelMasking, self).build(input_shape)

    def call(self, labels):
        # Depths is simply ignored during training time
        label_leaf = labels[:, -1]
        n_samples = tf.shape(label_leaf)[0]
        masks = []
        for depth in range(self.n_depths):
            indices = tf.concat([(tf.ones((n_samples, 1),
                                          dtype=np.int32) * depth),
                                 label_leaf[:, np.newaxis]], axis=1)
            masks.append(tf.gather_nd(self.adversaries, indices=indices))
        masks = tf.concat([mask[:, np.newaxis, :] for mask in masks], axis=1)
        return masks

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_depths, self.n_labels


class Pool(Layer):
    def __init__(self, depth_weight, output_shape, **kwargs):
        self.depth_weight = tf.convert_to_tensor(depth_weight, dtype=np.float32)
        # Keras can't infer output_shape
        self._output_shape = output_shape

        super(Pool, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Pool, self).build(input_shape)

    def call(self, inputs, training=None):
        prob, depths = inputs
        n_samples = tf.shape(prob)[0]

        def train_pool_prob():
            res = tf.transpose(prob, perm=[0, 2, 1])
            depth_weight = tf.tile(self.depth_weight[np.newaxis,
                                   :, np.newaxis],
                                   multiples=(n_samples, 1, 1))
            return tf.matmul(res, depth_weight)[:, :, 0]

        def test_pool_prob():
            res = tf.gather_nd(prob, indices=tf.concat(
                [tf.range(n_samples)[:, np.newaxis], depths], axis=1))
            return res

        return in_train_phase(train_pool_prob, test_pool_prob,
                              training=training)

    def compute_output_shape(self, input_shape):
        return self._output_shape


def make_aversaries(label_pool):
    n_labels, max_depth = label_pool.shape
    adversaries = np.ones((max_depth, n_labels, n_labels))
    for depth in range(1, max_depth):
        adversaries[depth] = (label_pool[:, [depth - 1]]
                              == label_pool[:, depth - 1])
    return adversaries


def make_model(n_features, alpha,
               latent_dim, dropout_input,
               depth_weight,
               dropout_latent,
               activation,
               seed, label_pool,
               shared_supervised):
    depth_weight = check_array(depth_weight, dtype=np.float32, ensure_2d=False)
    n_classes, n_depths = label_pool.shape
    assert(n_depths == depth_weight.shape[0])
    assert(np.all(depth_weight >= 0))
    depth_weight /= np.sum(depth_weight)
    adversaries = make_aversaries(label_pool)

    data = Input(shape=(n_features,), name='data')
    labels = Input((3,), name='labels', dtype=np.int32)
    # Unused at test time
    depths = Input((1,), name='depths', dtype=np.int32)
    # Dropout
    dropout_data = Dropout(rate=dropout_input, seed=seed)(data)
    if latent_dim is not None:
        latent = Dense(latent_dim, activation=activation,
                       use_bias=False, name='latent',
                       kernel_regularizer=l2(alpha))(dropout_data)
        latent_dropout = Dropout(rate=dropout_latent, name='dropout',
                                 seed=seed)(latent)
    else:
        latent_dropout = dropout_data
    mask = HierachicalLabelMasking(
        adversaries,
        name='label_masking')(labels)
    if shared_supervised:
        logits = Dense(n_classes, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        logits = RepeatVector(n=n_depths, name='repeat')(logits)
    else:
        logits = Dense(n_classes * n_depths,
                       activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        logits = Reshape((n_depths,
                          n_classes), name='reshape')(logits)
    prob = PartialSoftmax(name='partial_softmax')([logits, mask])

    pooled_prob = Pool(depth_weight=depth_weight, name='pool',
                       output_shape=(None, n_classes))([prob, depths])

    model = Model(inputs=[data, labels, depths], outputs=pooled_prob)
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