import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.constraints import non_neg
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout, Lambda
from keras.regularizers import l2
import keras.backend as K
from sklearn.utils import check_array
from tensorflow.python import debug as tf_debug


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, mask = inputs
        logits_max = tf.reduce_max(logits, axis=1, keep_dims=True)
        logits -= logits_max
        if mask is not None:
            exp_logits = tf.where(mask, tf.exp(logits),
                                  tf.zeros(tf.shape(logits)))
        else:
            exp_logits = tf.exp(logits)
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=1, keep_dims=True)
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
                 **kwargs):
        self.n_labels = n_labels
        super(HierarchicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.n_depths = input_shape
        self.adversaries = tf.Variable(
            tf.constant_initializer(True, dtype=tf.bool)((self.n_depths,
                                                          self.n_labels,
                                                          self.n_labels),
                                                         ),
            name='adversaries', dtype=tf.bool)
        self._non_trainable_weights = [self.adversaries]
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
    if dropout_input > 0:
        dropout_data = Dropout(rate=dropout_input, seed=seed)(data)
    else:
        dropout_data = data
    if latent_dim is not None:
        latent = Dense(latent_dim, activation=activation,
                       use_bias=False, name='latent',
                       # kernel_constraint=non_neg(),
                       kernel_regularizer=l2(alpha))(dropout_data)
        if dropout_latent > 0:
            latent_dropout = Dropout(rate=dropout_latent, name='dropout',
                                     seed=seed)(latent)
        else:
            latent_dropout = latent
    else:
        latent_dropout = dropout_data
    masks = HierarchicalLabelMasking(n_labels,
                                     weights=[adversaries],
                                     name='label_masking')(labels)
    outputs = []
    if shared_supervised:
        logits = Dense(n_labels, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent_dropout)
        for i, mask in enumerate(masks):
            prob = PartialSoftmax(name='softmax_depth_%i' % i)([logits, mask])
            outputs.append(prob)
    else:
        for i, mask in enumerate(masks):
            logits = Dense(n_labels,
                           activation='linear',
                           use_bias=True,
                           kernel_regularizer=l2(alpha),
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
    set_session(sess)
