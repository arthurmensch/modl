import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2
import keras.backend as K
from scipy.linalg import pinv
from tensorflow.python import debug as tf_debug
from keras.initializers import Orthogonal

MIN_FLOAT32 = np.finfo(np.float32).min


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, mask = inputs
        # Put logits to -inf for constraining probability to some support
        logits_min = tf.ones_like(logits) * MIN_FLOAT32
        logits = tf.where(mask, logits, logits_min)
        logits_max = K.max(logits, axis=1, keepdims=True)
        logits -= logits_max
        exp_logits = tf.exp(logits)
        sum_exp_logits = K.sum(exp_logits, axis=1, keepdims=True)
        return exp_logits / sum_exp_logits

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `PartialSoftmax` layer should be called '
                             'on a list of inputs.')
        input_shape = input_shape[0]
        return input_shape


class HierarchicalLabelMasking(Layer):
    def __init__(self,
                 n_labels,
                 adversaries,
                 **kwargs):
        self.n_labels = n_labels
        self.adversaries = adversaries
        super(HierarchicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.n_depths = input_shape
        # self.adversaries = tf.Variable(
        #     tf.constant_initializer(True, dtype=tf.bool)((self.n_depths,
        #                                                   self.n_labels,
        #                                                   self.n_labels),
        #                                                  ),
        #     name='adversaries', dtype=tf.bool)
        self.adversaries = tf.constant(self.adversaries)
        # self._non_trainable_weights = [self.adversaries]
        super(HierarchicalLabelMasking, self).build(input_shape)

    def call(self, labels, **kwargs):
        # Depths is simply ignored during training time
        label_leaf = labels[:, -1]
        n_samples = tf.shape(label_leaf)[0]
        masks = []
        for depth in range(self.n_depths):
            indices = tf.concat([(tf.ones((n_samples, 1),
                                          dtype=tf.int32) * depth),
                                 label_leaf[:, np.newaxis]], axis=1)
            masks.append(tf.gather_nd(self.adversaries, indices=indices))
        return masks

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.n_labels)] * self.n_depths

    def compute_mask(self, inputs, mask=None):
        return [None] * self.n_depths

    def get_config(self):
        config = {
            'n_labels': self.n_labels,
        }
        base_config = super(HierarchicalLabelMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_aversaries(label_pool):
    n_labels, max_depth = label_pool.shape
    adversaries = np.ones((max_depth, n_labels, n_labels), dtype=np.bool)
    for depth in range(1, max_depth):
        adversaries[depth] = (label_pool[:, [depth - 1]]
                              == label_pool[:, depth - 1])
    return adversaries


def make_model(n_features, alpha,
               latent_dim, dropout_input,
               dropout_latent,
               activation,
               seed, label_pool,
               shared_supervised):
    n_labels, n_depths = label_pool.shape
    adversaries = make_aversaries(label_pool)

    data = Input(shape=(n_features,), name='data', dtype='float32')
    labels = Input((3,), name='labels', dtype='int32')
    # if dropout_input > 0:
    #     latent = Dropout(rate=dropout_input, seed=seed)(data)
    # else:
    latent = data
    if latent_dim is not None:
        for i in range(1):
            latent = Dense(latent_dim, activation='linear',
                           use_bias=False, name='latent_%i' % i,
                           kernel_regularizer=l2(alpha)
                           )(latent)
            # latent = BatchNormalization(name='batch_norm_%i' % i)(latent)
            latent = Activation(activation=activation,
                                name='activation_%i' % i)(latent)
            # if dropout_latent > 0:
            #     latent = Dropout(rate=dropout_latent, name='dropout_%i' % i,
            #                      seed=seed)(latent)
    masks = HierarchicalLabelMasking(n_labels,
                                     adversaries=adversaries,
                                     # weights=[adversaries],
                                     name='label_masking')(labels)
    outputs = []
    if shared_supervised:
        logits = Dense(n_labels, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       kernel_initializer=Orthogonal(seed=seed),
                       # kernel_constraint=non_neg(),
                       # bias_constraint=non_neg(),
                       name='supervised')(latent)
        for i, mask in enumerate(masks):
            prob = PartialSoftmax(name='softmax_depth_%i' % i)([logits, mask])
            outputs.append(prob)
    else:
        for i, mask in enumerate(masks):
            logits = Dense(n_labels,
                           activation='linear',
                           use_bias=True,
                           kernel_regularizer=l2(alpha),
                           # kernel_constraint=non_neg(),
                           # bias_constraint=non_neg(),
                           name='supervised_depth_%i' % i)(latent_dropout)
            prob = PartialSoftmax(name='softmax_depth_%i' % i)([logits, mask])
            outputs.append(prob)
    model = Model(inputs=[data, labels], outputs=outputs)
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
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    set_session(sess)
    return sess


def make_projection_matrix(bases, scale_bases=True, inverse=True):
    if not isinstance(bases, list):
        bases = [bases]
    res = []
    for i, basis in enumerate(bases):
        if scale_bases:
            S = np.std(basis, axis=1)
            S[S == 0] = 1
            basis = basis / S[:, np.newaxis]
            if inverse:
                res.append(pinv(basis))
            else:
                res.append(basis)
    if inverse:
        res = np.concatenate(res, axis=1)
    else:
        res = np.concatenate(res, axis=0)

    return res
