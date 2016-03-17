import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces

from modl.dict_fact import DictMF

n_row, n_col = 3, 6
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
faces_centered /= np.sqrt(np.sum(faces_centered ** 2, axis=1))[:, np.newaxis]

print("Dataset consists of %d faces" % n_samples)


def sqnorm(X):
    return sqrt(np.sum(X ** 2))


class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr):
        self.X_tr = X_tr
        # self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.rmse_tr = []
        self.times = []
        self.sparsity = []
        self.iter = []
        self.q = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        P = mf.transform(self.X_tr)
        loss = np.sum((self.X_tr - P.T.dot(mf.components_)) ** 2)
        regul = mf.alpha * np.sum(P ** 2)
        self.obj.append(loss + regul)

        self.q.append(mf.Q_[1, np.linspace(0, 4095, 20, dtype='int')].copy())
        self.sparsity.append(np.sum(mf.Q_ != 0) / mf.Q_.size)
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)
        self.iter.append(mf.n_iter_)


def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


###############################################################################
# Do the estimation and plot it

name = 'MODL'
print("Extracting the top %d %s..." % (n_components, name))
t0 = time.time()
data = faces_centered
cb = Callback(data)

estimator = DictMF(n_components=n_components, batch_size=10,
                   reduction=3, l1_ratio=1, alpha=0.1, max_n_iter=15000,
                   full_projection=False,
                   impute=False,
                   persist_P=True,
                   backend='c',
                   verbose=3,
                   learning_rate=0.75,
                   offset=1000,
                   random_state=0,
                   callback=cb)
estimator.fit(data)
train_time = (time.time() - t0)
print("done in %0.3fs" % train_time)
components_ = estimator.components_
plot_gallery('%s - Train time %.1fs' % (name, train_time),
             components_[:n_components])

P = estimator.transform(data)
plot_gallery('Original faces',
             data[:n_components])
plot_gallery('Reconstruction',
             P.T.dot(estimator.components_)[:n_components])
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(cb.iter, cb.obj, label='Function value')
axes[1].plot(cb.iter, cb.sparsity, label='sparsity')
plt.legend()

plt.show()
