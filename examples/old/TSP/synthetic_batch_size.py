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

import pandas as pd


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


def single_run(data_tr, data_te,
               n_epochs=10,
               redundancy=1,
               **kwargs):
    t0 = time()
    data_tr = np.tile(data_tr, (1, redundancy))
    data_te = np.tile(data_te, (1, redundancy))
    cb = Callback(data_te)
    estimator = DictFact(
        callback=cb,
        n_epochs=n_epochs,
        verbose=2,
        random_state=0,
        n_threads=1,
        **kwargs)
    estimator.fit(data_tr)
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return kwargs, redundancy, cb


def run(n_jobs=1, n_epochs=10):
    # Exp def
    redundancies = [100]
    global_exp = dict(n_components=100, alpha=1,
                      l1_ratio=0,
                      pen_l1_ratio=.9,
                      learning_rate=0.9,
                      reduction=1,
                      Dx_agg='full',
                      G_agg='full',
                      AB_agg='full')
    exps = [dict(batch_size=batch_size)
            for batch_size in np.logspace(1, 2.5, 10).astype('int')]

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
    data_tr = data
    data_te = data

    print('done in %.2fs.' % (time() - t0))

    for i in range(len(exps)):
        exps[i] = dict(**exps[i], **global_exp)

    res = Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None,
                   backend='multiprocessing')(
        delayed(mem.cache(single_run))(data_tr, data_te,
                                       redundancy=redundancy,
                                       n_epochs=n_epochs,
                                       **this_exp)
        for redundancy in redundancies
        for this_exp in exps)
    dict_res = []
    for this_exp, this_redundancy, this_cb in res:
        this_res = copy(this_exp)
        this_res['iter'] = this_cb.iter
        this_res['obj'] = this_cb.obj
        this_res['times'] = this_cb.times
        this_res['redundancy'] = this_redundancy
        dict_res.append(this_res)
    return dict_res


def main(output_dir, n_jobs=1, n_epochs=5):
    res = run(n_jobs=n_jobs, n_epochs=n_epochs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, 'compare.json'), 'w+') as f:
        json.dump(res, f, cls=MyEncoder)


def plot(output_dir):
    with open(join(output_dir, 'compare.json'), 'r+') as f:
        res = json.load(f)
        df = pd.DataFrame(res)

        n_redundancies = len(df['redundancy'].unique())

        fig, axes = plt.subplots(n_redundancies, 1, figsize=(8, 12),
                                 squeeze=False)

        for i, (redundancy, this_df) in enumerate(df.groupby('redundancy')):
            lim = [1000, -1000]
            for idx, line in this_df.iterrows():
                axes[i, 0].plot(line['iter'][1:], line['obj'][1:],
                             label='batch size %s' % (line['batch_size']),
                             marker='o')
                # lim = [min(lim[0], min(line['obj'][1:])),
                #        max(lim[1], max(line['obj'][1:]) * 1.01)]
                # axes[i, 0].set_ylim(lim)
                axes[i, 0].set_ylabel('Train loss')
            axes[i, 0].annotate('Redundancy %i' % line['redundancy'],
                             xy=(0.5, 0.9), xycoords=('axes fraction'),
                             va='center', ha='center')
            axes[i, 0].set_xscale('log')
        fig.subplots_adjust(right=0.7)
        axes[i, 0].set_xlabel('Iter')
        axes[0, 0].legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig(join(output_dir, 'compare.pdf'))


if __name__ == '__main__':
    output_dir = expanduser('~/output/modl/synthetic_batch_size')

    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    main(output_dir, n_jobs=30, n_epochs=5)
    plot(output_dir)
