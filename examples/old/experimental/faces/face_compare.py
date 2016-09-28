import json
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

from modl.dict_fact import DictFact


def sqnorm(X):
    return sqrt(np.sum(X ** 2))


class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr):
        self.X_tr = X_tr
        self.obj = []
        self.rmse = []
        self.rmse_tr = []
        self.times = []
        self.sparsity = []
        self.iter = []
        self.components = []
        self.e = []
        self.f = []
        self.b = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        # code = mf.code_.T
        code = mf.transform(self.X_tr)
        loss = np.sum((self.X_tr - code.T.dot(mf.components_)) ** 2) / 2
        regul = mf.alpha * np.sum(code ** 2)
        self.obj.append(loss.flat[0] + regul)
        sum_B = sqrt(np.sum(mf.B_ ** 2))
        if not mf.var_red == 'weight_based':
            sum_B *= mf.multiplier_
        self.b.append(sum_B)
        self.components.append(
            mf.components_[1, np.linspace(0, 4095, 20, dtype='int')].tolist())
        self.sparsity.append(np.sum(mf.components_ != 0) / mf.components_.size)
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)
        self.iter.append(int(mf.n_iter_[0]))


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
    faces_centered /= np.sqrt(np.sum(faces_centered ** 2, axis=1))[:,
                      np.newaxis]

    print("Dataset consists of %d faces" % n_samples)
    data = faces_centered

    res = Parallel(n_jobs=3, verbose=10)(
        delayed(single_run)(n_components, var_red, projection, coupled_subset,
                            offset,
                            learning_rate, reduction,
                            alpha,
                            data)
        for projection in ['partial']
        for offset in [0]
        for learning_rate in [.9]
        for var_red, reduction, coupled_subset in [#('weight_based', 10, True),
                                                   ('weight_based', 10, True),
                                                   # ('combo', 10, False),
                                                   ('combo', 10, True),
                                                   # ('sample_based', 10, True),
                                                   ('sample_based', 10, False),
                                                   # ('none', 1, True)
                                                   ]
        for alpha in [0.001])

    full_res_dict = []
    for cb, estimator in res:
        res_dict = {'var_red': estimator.var_red,
                    'projection': estimator.projection,
                    'learning_rate': estimator.learning_rate,
                    'offset': estimator.offset,
                    'reduction': estimator.reduction, 'alpha': estimator.alpha,
                    'iter': cb.iter, 'times': cb.times,
                    'obj': cb.obj,
                    'sparsity': cb.sparsity, 'components': cb.components}
        full_res_dict.append(res_dict)
    json.dump(full_res_dict, open('results.json', 'w+'))

    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.subplots_adjust(left=0.15, right=0.6)
    for cb, estimator in res:
        axes[0].plot(cb.times, cb.obj,
                     label='%s, %s, %s' % (
                         estimator.reduction, estimator.var_red,
                         estimator.coupled_subset))
        axes[1].plot(cb.times, cb.sparsity)
        axes[2].plot(cb.times, np.array(cb.components)[:, 2])
        axes[3].plot(cb.times, cb.b)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[0].set_ylabel('Function value')
    axes[1].set_ylabel('Sparsity')

    axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    axes[3].set_xlabel('Iter')
    axes[3].set_ylabel('|B|')
    # axes[2].legend()
    axes[2].set_ylabel('Dictionary value')
    plt.show()
    # plt.savefig('face_compare.pdf')


def single_run(n_components, var_red, projection, coupled_subset, offset,
               learning_rate,
               reduction,
               alpha,
               data):
    cb = Callback(data)
    estimator = DictFact(n_components=n_components, batch_size=10,
                       reduction=reduction, l1_ratio=1, alpha=alpha,
                       coupled_subset=coupled_subset,
                       max_n_iter=20000,
                       projection=projection,
                       var_red=var_red,
                       backend='python',
                       verbose=10,
                       learning_rate=learning_rate,
                       offset=offset,
                       random_state=0,
                       callback=cb)
    estimator.fit(data)
    return cb, estimator


if __name__ == '__main__':
    main()
