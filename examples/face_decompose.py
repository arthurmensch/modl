import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces

from modl.dict_fact_remake import DictMFRemake

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
        self.iter = []
        self.q = []
        self.e = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        P = mf.transform(self.X_tr)
        loss = np.sum((data - mf.transform(self.X_tr).T.dot(mf.components_)) ** 2)
        regul = mf.alpha * np.sum(P ** 2)
        self.obj.append(loss + regul)

        self.q.append(mf.Q_[1, np.linspace(0, 4095, 3, dtype='int')].copy())
        self.e.append(mf._stat.E[1, np.linspace(0, 4095, 3, dtype='int')].copy())

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)
        self.iter.append(mf._stat.n_iter)


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

estimator = DictMFRemake(n_components=n_components, batch_size=10,
                         reduction=3, l1_ratio=1, alpha=1, max_n_iter=3000,
                         full_projection=True,
                         impute=True,
                         backend='python',
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

fig = plt.figure()
plt.plot(cb.iter, cb.obj, label='Function value')
plt.legend()

fig = plt.figure()
plt.plot(cb.iter, cb.q, label='q')
plt.plot(cb.iter, cb.e, label='e')
plt.legend()

plt.show()
