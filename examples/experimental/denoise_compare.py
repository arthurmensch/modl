import json
import os
from copy import copy, deepcopy
from os.path import expanduser, join
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
    redundency = kwargs.pop('redundency', 1)
    data_tr = np.tile(data_tr, (1, redundency))
    data_te = np.tile(data_te, (1, redundency))
    cb = Callback(data_te)
    estimator = DictFact(n_components=100, alpha=1,
                         l1_ratio=0,
                         pen_l1_ratio=.9,
                         learning_rate=0.9,
                         batch_size=10,
                         verbose=2,
                         n_epochs=50,
                         callback=cb,
                         random_state=0,
                         **kwargs)
    V = estimator.fit(data_tr).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return cb


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


def run(n_jobs=1):
    mem = Memory(cachedir=expanduser('~/cache'))
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
    data = extract_patches_2d(distorted[:, :width // 2], patch_size,
                              max_patches=4000, random_state=0)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    data_tr = data[:2000]
    data_te = data[:1000]
    print('done in %.2fs.' % (time() - t0))
    exps = []
    exp_per_redundency = [
        {'reduction': 1, 'solver': 'masked'},
            {'reduction': 8, 'solver': 'average'},
            {'reduction': 8, 'solver': 'masked'},
            {'reduction': 8, 'solver': 'gram'}]
    for redundency in [4, 8, 16, 32, 64]:
        these_exps = deepcopy(exp_per_redundency)
        for this_exp in these_exps:
            this_exp['redundency'] = redundency
            exps.append(this_exp)
    cbs = Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None,
                   backend='multiprocessing')(
        delayed(mem.cache(single_run))(data_tr, data_te, **this_exp)
        for this_exp in exps)
    res = []
    for this_exp, this_cb in zip(exps, cbs):
        this_res = copy(this_exp)
        this_res['iter'] = this_cb.iter
        this_res['obj'] = this_cb.obj
        this_res['times'] = this_cb.times
        res.append(this_res)
    return res


def main():
    res = run(20)
    output_dir = expanduser('~/output/modl/denoise')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, 'compare.json'), 'w+') as f:
        json.dump(res, f, cls=MyEncoder)

def plot():
    output_dir = expanduser('~/output/modl/denoise')
    with open(join(output_dir, 'compare.json'), 'w+') as f:
        json.dump(res, f, cls=MyEncoder)


if __name__ == '__main__':
    main()
