from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from modl.datasets import fetch_adhd
from modl.utils.system import get_cache_dirs
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory

from keras.models import Model
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.callbacks import TensorBoard

from os.path import expanduser

from numpy.linalg import qr



data = fetch_adhd()

imgs = data.rest

memory = Memory(cachedir=get_cache_dirs()[0])

masker = MultiNiftiMasker(mask_img=data.mask,
                          smoothing_fwhm=6,
                          standardize=True,
                          detrend=True,
                          memory=memory,
                         memory_level=1).fit()

n_voxels = masker.mask_img_.get_data().sum()

data = []
for i, (subject, img) in enumerate(imgs.iterrows()):
    data.append(masker.transform(img['filename'], img['confounds']))


train_data, test_data = train_test_split(data, test_size=.1)

train_data = np.concatenate(train_data)
test_data = np.concatenate(test_data)

sess = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=3,
    intra_op_parallelism_threads=3,
    use_per_session_threads=True))


set_session(sess)
encoding_dim = 32


def fit_autoencoder(train_data, test_data,
                    encoding_dim=32):
    n_voxels = train_data.shape[1]
    # this is our input placeholder
    input_img = Input(shape=(n_voxels,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='linear', bias=False)(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(n_voxels, activation='linear', bias=False)(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(loss='mse', optimizer='adagrad')

    # this model maps an input to its encoded representation
    encoder = Model(input=input_img, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    callback = TensorBoard(log_dir=expanduser('~/output/logs'))

    autoencoder.fit(train_data, train_data,
                    validation_data=(test_data, test_data),
                    callbacks=[callback], nb_epoch=10,
                    batch_size=32)

    components = autoencoder.layers[1].get_weights()[0]
    return components