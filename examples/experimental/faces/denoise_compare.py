import json
from os.path import expanduser
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from joblib import Parallel
from joblib import delayed
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d

from modl.dict_fact import DictMF


class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr):
        self.X_tr = X_tr
        self.obj = []
        self.times = []
        self.iter = []
        self.start_time = time()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time()
        self.obj.append(mf.score(self.X_tr))
        self.test_time += time() - test_time
        self.times.append(time() - self.start_time - self.test_time)
        self.iter.append(mf.n_iter_[0])


def single_run(solver, weights,
               reduction, subset_sampling, dict_subset_sampling,
               learning_rate, data_tr, data_te):
    t0 = time()
    cb = Callback(data_te)
    estimator = DictMF(n_components=100, alpha=1,
                       l1_ratio=0,
                       pen_l1_ratio=.9,
                       batch_size=10,
                       learning_rate=learning_rate,
                       reduction=reduction,
                       subset_sampling=subset_sampling,
                       dict_subset_sampling=dict_subset_sampling,
                       verbose=5,
                       solver=solver,
                       weights=weights,
                       n_epochs=2, backend='c',
                       callback=cb)
    V = estimator.fit(data_tr).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return cb, estimator


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def run(redundency=1):
    face = misc.face(gray=True)

    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    face = face / 255

    # downsample for higher speed
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2,
                                                                1::2]
    face /= 4.0
    height, width = face.shape

    # Distort the right half of the image
    print('Distorting image...')
    distorted = face.copy()
    # distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    patch_size = (8, 8)
    tile = redundency
    data = extract_patches_2d(distorted[:, :width // 2], patch_size,
                              max_patches=4000)
    tiled_data = np.empty(
        (data.shape[0], data.shape[1] * tile, data.shape[2] * tile))
    for i in range(tile):
        for j in range(tile):
            tiled_data[:, i::tile, j::tile] = data
    data = tiled_data
    patch_size = (8 * tile, 8 * tile)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    data_tr = data[:2000]
    data_te = data[:1000]
    print('done in %.2fs.' % (time() - t0))

    res = Parallel(n_jobs=4, verbose=10, max_nbytes=None)(
        delayed(single_run)(solver, weights,
                            reduction,
                            subset_sampling,
                            dict_subset_sampling,
                            learning_rate,
                            data_tr, data_tr)
        for weights in ['async_freq', 'async_prob']
        for reduction in [6]
        for solver in ['masked']
        for subset_sampling in ['random', 'cyclic']
        for dict_subset_sampling in ['independent', 'coupled']
        for learning_rate in [1.])

    return res


def main():
    mem = Memory(cachedir=expanduser('~/cache'))
    for redundency in [1]:
        res = mem.cache(run)(redundency=redundency)
        res = run(redundency=redundency)
        # full_res_dict = []
        # for cb, estimator in res:
        #     res_dict = {'solver': estimator.solver,
        #                 'reduction': estimator.reduction,
        #                 'iter': cb.iter, 'times': cb.times,
        #                 'obj': cb.obj}
        #     full_res_dict.append(res_dict)
        # json.dump(full_res_dict, open('results.json', 'w+'), cls=MyEncoder)
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(left=0.15, right=0.6)

        for cb, estimator in res:
            ax.plot(cb.iter[1:], cb.obj[1:],
                    label='Solver: %s\n Sampling: %s %s %s' % (
                        estimator.solver, estimator.weights,
                        estimator.subset_sampling,
                        estimator.dict_subset_sampling))

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylabel('Function value')

    plt.show()


if __name__ == '__main__':
    main()
