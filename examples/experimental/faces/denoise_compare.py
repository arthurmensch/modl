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

from modl.dict_fact import DictFact


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
        self.iter.append(mf.total_counter)


def single_run(data_tr, data_te, **kwargs):
    t0 = time()
    cb = Callback(data_te)
    estimator = DictFact(n_components=100, alpha=1,
                         l1_ratio=0,
                         pen_l1_ratio=.9,
                         learning_rate=0.9,
                         batch_size=10,
                         verbose=2,
                         n_epochs=10,
                         callback=cb,
                         **kwargs)
    V = estimator.fit(data_tr).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return cb, kwargs


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
    exps = [{'reduction': 1, 'solver': 'masked'},
            {'reduction': 4, 'solver': 'average'},
            {'reduction': 4, 'solver': 'gram'}]
    res = Parallel(n_jobs=3, verbose=10, max_nbytes=None,
                   backend='threading')(
        delayed(single_run)(data_tr, data_tr, **this_exp)
        for this_exp in exps)

    return res


def main():
    mem = Memory(cachedir=expanduser('~/cache'))
    for redundency in [4]:
        res = run(redundency=redundency)
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(left=0.15, right=0.6)

        for cb, kwargs in res:
            ax.plot(cb.iter[1:], cb.obj[1:],
                    label=str(kwargs))

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylabel('Function value')

    plt.show()


if __name__ == '__main__':
    main()
