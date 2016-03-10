import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

from modl.dict_fact import DictMF


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
        self.n_iter = []
        self.q = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        P = mf.transform(self.X_tr)
        loss = sqnorm(self.X_tr - mf.transform(self.X_tr).T.dot(mf.components_)) / 2
        regul = mf.alpha * sqnorm(P)
        self.obj.append(loss + regul)

        self.q.append(mf.Q_[1, np.linspace(0, 4095, 50, dtype='int')].copy())
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)
        self.n_iter.append(mf._stat.n_iter)


def plot_gallery(title, images, n_col, n_row, image_shape):
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


def main():
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

    print("Dataset consists of %d faces" % n_samples)
    data = faces_centered

    lrs = np.empty(6)
    lrs[0] = -1
    lrs[1:] = np.linspace(0.1, 1, 5)

    cbs = Parallel(n_jobs=3)(delayed(single_run)(n_components, impute_lr, data)
                       for impute_lr in lrs)

    fig, axes = plt.subplots(2, 1)
    for impute_lr, this_cb in zip(lrs, cbs):
        axes[0].plot(this_cb.n_iter, this_cb.obj, label='Function value %.2f' % impute_lr)
        axes[1].plot(this_cb.n_iter, np.array(this_cb.q)[:, 0], label='Dict value %.2f' % impute_lr)
    axes[0].legend()
    axes[1].legend()

    plt.show()


def single_run(n_components, impute_lr, data):
    cb = Callback(data)
    estimator = DictMF(n_components=n_components, batch_size=10,
                       reduction=3, l1_ratio=1, alpha=1e-3, max_n_iter=30000,
                       full_projection=True,
                       impute=True,
                       impute_lr=impute_lr,
                       backend='c',
                       verbose=3,
                       random_state=0,
                       callback=cb)
    estimator.fit(data)
    return cb

if __name__ == '__main__':
    main()