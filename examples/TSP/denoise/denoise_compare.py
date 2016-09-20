from os.path import expanduser, join
import time

import itertools
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel
from joblib import delayed
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import check_random_state

import json

import os

from modl.dict_fact import DictFact
from math import sqrt
import seaborn.apionly as sns

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        return super(NumpyAwareJSONEncoder, self).default(obj)


patch_size = (8, 8)

class Callback(object):
    def __init__(self, train_data=None,
                 trace_folder=None,
                 test_data=None,
                 raw=False,
                 verbose_iter=None,
                 plot=False):
        self.train_data = train_data
        self.test_data = test_data
        self.trace_folder = trace_folder
        # if self.train_data is not None:
        self.train_obj = []
        # if self.test_data is not None:
        self.test_obj = []
        self.plot = plot
        self.raw = raw
        self.train_obj = []
        self.time = []
        self.iter = []

        self.verbose_iter = verbose_iter

        self.start_time = time.clock()
        self.test_time = 0

        self.profile = []

        self.call = 0

    def display(self, mf):
        if len(self.profile) == 1:
            return

        V = mf.components_

        fig = plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(V[:100]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('Dictionary',
                     fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.savefig(join(self.trace_folder, 'components.png'))
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        fig.subplots_adjust(right=0.8)

        profile = np.array(self.profile)
        iter = np.array(self.iter)

        if self.train_obj:
            train_obj = np.array(self.train_obj[1:])
            axes[0].plot(iter[1:], train_obj, marker='o',
                         label='train set')
        if self.test_obj:
            test_obj = np.array(self.test_obj[1:])
            axes[0].plot(iter[1:], test_obj, marker='o', label='test set')
        axes[0].legend(bbox_to_anchor=(1, 1), loc="upper left")

        # Profile
        profile = profile[:, [0, 1, 2, 3, 4]]
        labels = np.array(['', 'Dx time', 'G time', 'Code time',
                           'Agg time', 'BCD time'])
        average_time = np.zeros((profile.shape[0] - 1, profile.shape[1] + 1))
        average_time[:, 1:] = (profile[1:] - profile[:-1]) \
                           / (iter[1:] - iter[:-1])[:, np.newaxis]
        sort = np.argsort(average_time[-1, :])
        average_time = average_time[:, sort]
        labels = labels[sort]
        average_time = np.cumsum(average_time, axis=1)

        palette = sns.color_palette("deep", 5)
        for i in range(1, 6):
            # axes[1].plot(iter, average_time[:, i],
            #              color=palette[i - 1])
            axes[1].fill_between(iter[1:], average_time[:, i], average_time[:, i - 1],
                                 facecolor=palette[i - 1], label=labels[i])

            # axes[1].plot(iter[1:], average_time[1:], marker='o')
        axes[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
        axes[1].set_ylabel('Average time')
        # axes[1].set_yscale('Log')
        axes[0].set_ylabel('Function value')

        plt.savefig(join(self.trace_folder, 'loss_profile.png'))
        plt.close(fig)

    def __call__(self, mf):
        print('Testing...')
        test_time = time.clock()
        if self.train_data is not None:
            train_obj = mf.score(self.train_data)
            self.train_obj.append(train_obj)
        if self.test_data is not None:
            test_obj = mf.score(self.test_data)
            self.test_obj.append(test_obj)
        if self.trace_folder is not None:
            np.save(join(self.trace_folder, "record_%s" % mf.n_iter),
                    mf.components_)
            with open(join(self.trace_folder, 'callback.json'), 'w+') as f:
                json.dump({'iter': self.iter,
                           'time': self.time,
                           'train_obj': self.train_obj,
                           'test_obj': self.test_obj,
                           'profile': self.profile}, f,
                          cls=NumpyAwareJSONEncoder)
        self.iter.append(mf.n_iter)
        self.profile.append(mf.time)

        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)

        if self.trace_folder is not None and self.call % self.plot:
            test_time = time.clock()
            self.display(mf)
            self.test_time += time.clock() - test_time
        self.call += 1


def prepare_folder(name, n_exp):
    trace_folder = expanduser('~/output/modl/' + name)
    try:
        os.makedirs(trace_folder)
    except OSError:
        pass
    trace_folder_list = []
    for i in range(n_exp):
        this_trace_folder = join(trace_folder, 'experiment_%i' % i)
        try:
            os.makedirs(this_trace_folder)
        except OSError:
            pass
        trace_folder_list.append(this_trace_folder)
    return trace_folder_list


def fetch_data(redundancy=1):
    tile = int(sqrt(redundancy))
    face = misc.face(gray=True)
    face = face / 255
    height, width = face.shape
    data = extract_patches_2d(face[:, :width // 2], patch_size,
                              max_patches=10000, random_state=0)
    tiled_data = np.empty(
        (data.shape[0], data.shape[1] * tile, data.shape[2] * tile))
    for i in range(tile):
        for j in range(tile):
            tiled_data[:, i::tile, j::tile] = data
    data = tiled_data
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    return data, None


def run(exps, n_jobs=1):
    n_exp = len(exps)

    trace_folder_list = prepare_folder('denoise', n_exp)

    train_data, test_data = fetch_data()

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_single)(
        trace_folder_list[idx], train_data, test_data,
        **this_exp) for idx, this_exp in enumerate(exps))


def run_single(trace_folder,
               train_data, test_data,
               G_agg='full', Dx_agg='full', AB_agg='full',
               reduction=1,
               **kwargs):
    cb = Callback(train_data=train_data, trace_folder=trace_folder,
                  plot=False)
    n_samples = train_data.shape[0]
    dico = DictFact(n_components=100, alpha=1,
                    l1_ratio=0,
                    pen_l1_ratio=.9,
                    batch_size=50,
                    learning_rate=.9,
                    sample_learning_rate=None,
                    reduction=reduction,
                    verbose=2,
                    G_agg=G_agg,
                    Dx_agg=Dx_agg,
                    AB_agg=AB_agg,
                    proj='partial',
                    subset_sampling='random',
                    dict_reduction='follow',
                    callback=cb,
                    n_threads=1,
                    n_samples=n_samples,
                    lasso_tol=1e-2,
                    # purge_tol=1e-3,
                    random_state=42,
                    n_epochs=2,
                    **kwargs)
    #     dico.partial_fit(data)
    dico.fit(train_data)
    cb.display(dico)


if __name__ == '__main__':
    exps = [dict(reduction=1, G_agg='full', Dx_agg='full', AB_agg='full'),
            dict(reduction=5, G_agg='average', Dx_agg='average', AB_agg='full')]
    run(exps, n_jobs=2)
