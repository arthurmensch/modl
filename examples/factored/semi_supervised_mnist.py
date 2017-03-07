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
    n_epochs = 3
    supervision_ratio = 1


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

    autoencoder = keras.models.load_model(
        join(output_dir, 'autoencoder.keras'))

    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    input_imgs = autoencoder.input
    encoded = autoencoder.layers_by_depth[1][0](input_imgs)
    encoder = Model(input=input_imgs, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=encoder.output_shape)
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.output_layers[0]
    # create the decoder model
    # decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    if supervision:
        supervised = autoencoder.output_layers[1](encoded)
        classifier = Model(input=input_imgs, output=supervised)
        predicted_labels = classifier.predict(X_test)
        predicted_labels = np.argmax(predicted_labels, axis=1)
        accuracy = np.sum(predicted_labels == y_test) / predicted_labels.shape[0]
        print('Accuracy', accuracy)

    rec_X_test = autoencoder.predict(X_test)[0]
    rec_X_test = rec_X_test.reshape(len(X_test), 28, 28)
    plot_images(X_test, rec_X_test, supervision)
    return 0


@autoencoder.automain
def train(supervision, encoding_dim, n_epochs, supervision_ratio,
          _run):
    init_tensorflow(n_jobs=3)

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
        loss_weights = [1 - supervision_ratio, supervision_ratio]
    else:
        output = decoded
        loss = 'binary_crossentropy'
        loss_weights = 1
    autoencoder = Model(input=input_img,
                        output=output, name='autoencoder')
    autoencoder.compile(optimizer='adadelta',
                        loss=loss, metrics=['accuracy'], loss_weights=loss_weights)

    autoencoder.fit(X_train, [X_train, y_train] if supervision else X_train,
                    batch_size=256,
                    nb_epoch=n_epochs)

    output_dir = join(expanduser('~/models'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    autoencoder.save(join(output_dir, 'autoencoder.keras'))
