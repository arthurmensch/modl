import time
from math import sqrt

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
        self.obj_tr = []
        self.rmse = []
        self.rmse_tr = []
        self.times = []
        self.sparsity = []
        self.iter = []
        self.regul = []
        self.regul_tr = []
        self.q = []
        self.diff = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        P = mf.code_.T
        loss = np.sum((self.X_tr - P.T.dot(mf.components_)) ** 2) / 2
        regul = mf.alpha * np.sum(P ** 2)
        self.obj.append(loss + regul)
        self.regul.append(regul)

        P = mf.transform(self.X_tr)
        loss = np.sum((self.X_tr - P.T.dot(mf.components_)) ** 2) / 2
        regul = mf.alpha * np.sum(P ** 2)
        self.obj_tr.append(loss + regul)
        self.regul_tr.append(regul)

        beta = self.X_tr.dot(mf.components_.T)
        if hasattr(mf, 'beta_'):
            self.diff.append(np.sum((beta - mf.beta_) ** 2))
        else:
            self.diff.append(0.)
        self.q.append(mf.components_[1, np.linspace(0, 4095, 20, dtype='int')].copy())
        self.sparsity.append(np.sum(mf.components_ != 0) / mf.components_.size)
        # self.sparsity.append(np.sum(np.abs(mf.Q_)) / np.sum(mf.Q_ ** 2))
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

estimator = DictMF(n_components=n_components, batch_size=1,
                   reduction=10,
                   l1_ratio=1,
                   alpha=0.001,
                   max_n_iter=400,
                   projection='partial',
                   var_red='weight_based',
                   backend='python',
                   verbose=1,
                   learning_rate=.8,
                   offset=0,
                   random_state=2,
                   callback=cb)
estimator.fit(data)
train_time = (time.time() - t0)
print("done in %0.3fs" % train_time)

import matplotlib.pyplot as plt
components_ = estimator.components_
plot_gallery('%s - Train time %.1fs' % (name, train_time),
             components_[:n_components])

P = estimator.transform(data)
# plot_gallery('Original faces',
#              data[:n_components])
plot_gallery('Residual',
             data[:n_components] - P.T.dot(estimator.components_)[:n_components])
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(cb.iter, cb.obj, label='P_')
axes[0].plot(cb.iter, cb.regul, label='regul_')
axes[0].plot(cb.iter, cb.regul_tr, label='regul')
axes[0].plot(cb.iter, cb.obj_tr, label='P')
axes[0].legend()

axes[2].plot(cb.iter, cb.diff, label='beta_ variance')
axes[2].set_xlabel('beta_ variance')

axes[0].set_xlabel('Function value')
axes[1].plot(cb.iter, cb.sparsity, label='sparsity')
axes[1].set_xlabel('Sparsity')
axes[0].set_xscale('log')

plt.show()
