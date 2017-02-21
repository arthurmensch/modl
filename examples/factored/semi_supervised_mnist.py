import os
from os.path import expanduser
from os.path import join

import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sacred import Experiment

import tensorflow as tf
from keras.backend import set_session

from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

autoencoder = Experiment('autoencoder')


@autoencoder.config
def config():
    encoding_dim = 32
    supervision = True
    n_epochs = 1


def init_tensorflow(n_jobs=1):
    sess = tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=n_jobs,
        intra_op_parallelism_threads=n_jobs,
        use_per_session_threads=True))
    set_session(sess)


def plot_images(original, decoded, supervision):
    n = 10  # how many digits we will display
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('supervised' if supervision else 'unsupervised')
    plt.show()


@autoencoder.command
def test(supervision):
    init_tensorflow(n_jobs=2)

    output_dir = join(expanduser('~/models'))

    autoencoder = keras.models.load(join(output_dir, 'autoencoder.keras'))

    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    # this model maps an input to its encoded representation
    encoder = Model(input=input_img, output=encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers_by_depth[0][0]
    # create the decoder model
    decoder = Model(input=encoded_input,
                          output=decoder_layer(encoded_input))

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    if supervision:
        classifier = Model(input=input_img, output=supervised)


    decoded_test = autoencoder.predict(X_test)
    predicted_labels = autoencoder.predict_label(X_test)
    accuracy = np.sum(predicted_labels == y_test) / predicted_labels.shape[0]
    print(accuracy)
    plot_images(X_test, decoded_test, supervision)

@autoencoder.automain
def train(supervision, encoding_dim, n_epochs, _run):
    init_tensorflow(n_jobs=2)

    (X_train, y_train), (_, _) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

    le = LabelEncoder()
    labels = le.fit_transform(y_train)
    n_labels = np.unique(labels).shape[0]
    n_features = X_train.shape[1]
    input_img = Input(shape=(n_features,), name='input')
    encoded = Dense(encoding_dim, activation='relu',
                    name='encoder')(input_img)
    decoded = Dense(n_features, activation='sigmoid',
                    name='decoder')(encoded)
    supervised = Dense(n_labels, activation='softmax',
                       name='classifier')(encoded)

    if supervision:
        output = [decoded, supervised]
        loss = ['binary_crossentropy',
                'sparse_categorical_crossentropy']
    else:
        output = decoded
        loss = 'binary_crossentropy'

    autoencoder = Model(input=input_img,
                        output=output)
    autoencoder.compile(optimizer='adadelta',
                        loss=loss)

    autoencoder.fit(X_train, [X_train, y_train] if supervision else X_train,
                    batch_size=256,
                    nb_epoch=n_epochs)

    output_dir = join(expanduser('~/models'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    autoencoder.save(join(output_dir, 'autoencoder.keras'))