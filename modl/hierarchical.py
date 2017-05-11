import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.engine import Layer, Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2
import keras.backend as K
from scipy.linalg import pinv
from tensorflow.python import debug as tf_debug

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
                 n_depths,
                 **kwargs):
        self.n_labels = n_labels
        self.n_depths = n_depths
        super(HierarchicalLabelMasking, self).__init__(**kwargs)

    def build(self, input_shape):
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
        n_samples = tf.shape(labels)[0]
        masks = []
        for depth in range(self.n_depths):
            indices = tf.concat([(tf.ones((n_samples, 1),
                                          dtype=tf.int32) * depth),
                                 labels], axis=1)
            masks.append(tf.gather_nd(self.adversaries, indices=indices))
        return masks

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.n_labels)] * self.n_depths

    def compute_mask(self, inputs, mask=None):
        return [None] * self.n_depths

    def get_config(self):
        config = {
            'n_labels': self.n_labels,
            'n_depths': self.n_depths,
        }
        base_config = super(HierarchicalLabelMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_adversaries(label_pool):
    n_labels, max_depth = label_pool.shape
    adversaries = np.ones((max_depth, n_labels, n_labels), dtype=np.bool)
    for depth in range(1, max_depth):
        for i in range(0, depth):
            adversaries[depth] *= label_pool[:, [i]] == label_pool[:, i]
    return adversaries


def make_multi_model(n_features, lbins,
                     alpha, latent_dim, dropout_input,
                     dropout_latent, activation, use_task_specific,
                     seed):
    data = Input(shape=(n_features,), name='data', dtype='float32')

    if dropout_input > 0:
        dropout_data = Dropout(rate=dropout_input, seed=seed,
                               name='dropout_input')(data)
    else:
        dropout_data = data
    if latent_dim is not None:
        latent = Dense(latent_dim, activation='linear',
                       use_bias=False, name='latent',
                       kernel_regularizer=l2(alpha))(dropout_data)
        if dropout_latent > 0:
            latent = Dropout(rate=dropout_latent, name='dropout',
                             seed=seed)(latent)
    else:
        latent = dropout_data

    models = {'dataset': {}, 'task': {}}

    for dataset in lbins['dataset']:
        len_output = len(lbins['dataset'][dataset].classes_)
        output = Dense(len_output, activation='softmax',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised_%s' % dataset)(latent)
        model = Model(inputs=[data], outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam')
        models['dataset'][dataset] = model
        if use_task_specific:
            models['task'][dataset] = {}
            for task in lbins['task'][dataset]:
                len_output = len(lbins['task'][dataset][task].classes_)
                if len_output == 2:
                    output = Dense(1, activation='sigmoid',
                                   use_bias=True,
                                   kernel_regularizer=l2(alpha),
                                   name='supervised_%s_%s' % (dataset, task))(
                        latent)
                    loss = 'binary_crossentropy'
                else:
                    output = Dense(len_output, activation='softmax',
                                   use_bias=True,
                                   kernel_regularizer=l2(alpha),
                                   name='supervised_%s_%s' % (dataset, task))(
                        latent)
                    loss = 'categorical_crossentropy'
                model = Model(inputs=[data], outputs=output)
                model.compile(loss=loss,
                              optimizer='adam')
                models['task'][dataset][task] = model
    return models


def make_model(n_features, alpha,
               latent_dim, dropout_input,
               dropout_latent,
               activation,
               seed, adversaries,
               shared_supervised):
    n_depths, n_labels, _ = adversaries.shape

    data = Input(shape=(n_features,), name='data', dtype='float32')
    labels = Input((1,), name='labels', dtype='int32')
    if dropout_input > 0:
        dropout_data = Dropout(rate=dropout_input, seed=seed,
                               name='dropout_input')(data)
    else:
        dropout_data = data
    if latent_dim is not None:
        latent = Dense(latent_dim, activation=activation,
                       use_bias=False, name='latent',
                       kernel_regularizer=l2(alpha))(dropout_data)
        if dropout_latent > 0:
            latent = Dropout(rate=dropout_latent, name='dropout',
                             seed=seed)(latent)
    else:
        latent = dropout_data
    masks = HierarchicalLabelMasking(n_labels, n_depths,
                                     weights=[adversaries],
                                     name='label_masking')(labels)
    outputs = []
    if shared_supervised:
        logits = Dense(n_labels, activation='linear',
                       use_bias=True,
                       kernel_regularizer=l2(alpha),
                       name='supervised')(latent)
        for i, mask in enumerate(masks):
            prob = PartialSoftmax(name='softmax_depth_%i' % i)([logits, mask])
            outputs.append(prob)
    else:
        for i, mask in enumerate(masks):
            this_latent = latent
            logits = Dense(n_labels,
                           activation='linear',
                           use_bias=True,
                           kernel_regularizer=l2(alpha),
                           name='supervised_depth_%i' % i)(this_latent)
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


def make_projection_matrix(bases, scale_bases=True):
    if not isinstance(bases, list):
        bases = [bases]
    proj = []
    rec = []
    for i, basis in enumerate(bases):
        if scale_bases:
            S = np.std(basis, axis=1)
            S[S == 0] = 1
            basis = basis / S[:, np.newaxis]
            proj.append(pinv(basis))
            rec.append(basis)
    proj = np.concatenate(proj, axis=1)
    rec = np.concatenate(rec, axis=0)
    proj_inv = np.linalg.inv(proj.T.dot(rec.T)).T.dot(rec)
    return proj, proj_inv, rec
