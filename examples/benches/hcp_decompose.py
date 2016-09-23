# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

import json
import os
import time
from os.path import expanduser, join

import itertools
import numpy as np
from joblib import Parallel
from joblib import delayed
from modl.datasets import get_hcp_data
from modl.spca_fmri import SpcaFmri
from nilearn import datasets

import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.plotting import plot_prob_atlas, plot_stat_map
from nilearn.image import index_img
from sklearn.cross_validation import train_test_split
from math import log

import seaborn.apionly as sns

data_dir = expanduser('~/data')


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        return super(NumpyAwareJSONEncoder, self).default(obj)


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
        if self.train_data is not None:
            self.train_obj = []
        if self.test_data is not None:
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

    def display(self, spca_fmri):
        if len(self.profile) == 1:
            return

        fig, axes = plt.subplots(2, 1)
        plot_prob_atlas(spca_fmri.components_, view_type="filled_contours",
                        axes=axes[0])
        plot_stat_map(index_img(spca_fmri.components_, 0),
                      axes=axes[1],
                      colorbar=False,
                      threshold=0)
        plt.savefig(join(self.trace_folder, 'components.png'))

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
        profile = profile[:, [6, 0, 1, 2, 3, 4]]
        labels = np.array(['', 'IO time', 'Dx time', 'G time', 'Code time', 'Agg time',
                        'BCD time'])
        average_time = np.zeros((profile.shape[0] - 1, profile.shape[1] + 1))
        average_time[:, 1:] = (profile[1:] - profile[:-1]) \
                           / (iter[1:] - iter[:-1])[:, np.newaxis]
        sort = np.argsort(average_time[-1, :])
        average_time = average_time[:, sort]
        labels = labels[sort]
        average_time = np.cumsum(average_time, axis=1)

        palette = sns.color_palette("deep", 6)
        for i in range(1, 7):
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

    def __call__(self, spca_fmri):
        print('Testing...')
        test_time = time.clock()
        if self.train_data is not None:
            train_obj = spca_fmri.score(self.train_data, raw=self.raw)
            self.train_obj.append(train_obj)
        if self.test_data is not None:
            test_obj = spca_fmri.score(self.test_data, raw=self.raw)
            self.test_obj.append(test_obj)
        if self.trace_folder is not None:
            spca_fmri.components_.to_filename(join(self.trace_folder,
                                                   "record_"
                                                   "%s.nii.gz"
                                                   % spca_fmri.n_iter_))
            with open(join(self.trace_folder, 'callback.json'), 'w+') as f:
                json.dump({'iter': self.iter,
                           'time': self.time,
                           'train_obj': self.train_obj,
                           'test_obj': self.test_obj,
                           'profile': self.profile}, f,
                          cls=NumpyAwareJSONEncoder)
        self.iter.append(spca_fmri.n_iter_)
        self.profile.append(spca_fmri.timings_)

        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)

        if self.trace_folder is not None and self.plot:
            test_time = time.clock()
            self.display(spca_fmri)
            self.test_time += time.clock() - test_time
        print('Done...')


def fetch_data(source='adhd'):
    print('Fetching data')
    if source == 'adhd':
        mask = expanduser('~/data/ADHD_mask/mask_img.nii.gz')
        adhd_dataset = datasets.fetch_adhd(n_subjects=40)
        data = adhd_dataset.func  # list of 4D nifti files for each subject
        raw = False
    elif source == 'hcp':
        mask, data = get_hcp_data(data_dir, True)
        data = data[:500]
        raw = True
    else:
        raise ValueError('Wrong dataset')

    train_data, test_data = train_test_split(data, test_size=4
    if source == 'hcp' else 4,
                                             random_state=0)

    init = None  # fetch_atlas_smith_2009().rsn70

    print('Train data length: %i. Test data length: %i' % (len(train_data),
                                                           len(test_data)))
    return mask, train_data, test_data, init, raw


def run_single(trace_folder,
               train_data,
               test_data,
               n_jobs=1,
               mask=None,
               raw=False,
               reduction=1,
               n_components=None,
               init=None,
               verbose=0,
               **kwargs):
    n_epochs = int(reduction)
    if verbose != 0:
        verbose_iter = np.unique((np.logspace(1, log(len(train_data) * reduction,
                                                     10), verbose,
                                              base=10) - 10).astype('int'))
    else:
        verbose_iter = None
    cb = Callback(test_data=test_data,
                  trace_folder=trace_folder,
                  verbose_iter=verbose_iter,
                  plot=True,
                  raw=raw)
    spca_fmri = SpcaFmri(smoothing_fwhm=3.,
                         n_components=n_components,
                         dict_init=init,
                         mask=mask,
                         memory=expanduser("~/nilearn_cache"),
                         memory_level=2,
                         verbose=20,
                         random_state=0,
                         reduction=reduction,
                         learning_rate=.9,
                         l1_ratio=0.5,
                         AB_agg='async',
                         G_agg='full',
                         Dx_agg='average',
                         offset=0,
                         shelve=not raw,
                         n_epochs=n_epochs,
                         callback=cb,
                         n_jobs=n_jobs,
                         **kwargs
                         )

    relevent_params = ['batch_size', 'learning_rate', 'offset',
                       'AB_agg', 'G_agg', 'Dx_agg',
                       'reduction',
                       'alpha',
                       'l1_ratio']

    exp_dict = {param: getattr(spca_fmri, param) for param in relevent_params}

    with open(join(trace_folder, 'experiment.json'), 'w+') as f:
        json.dump(exp_dict, f, cls=NumpyAwareJSONEncoder)

    print('[Example] Learning maps')
    spca_fmri.fit(train_data, raw=raw)
    spca_fmri.callback(spca_fmri)


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


def run(dataset='adhd',
        reduction_list=[1],
        alpha_list=[0.001], n_jobs=1):
    n_exp = len(reduction_list) * len(alpha_list)

    trace_folder_list = prepare_folder(dataset, n_exp)

    mask, train_data, test_data, init, raw = fetch_data(dataset)
    if dataset == 'hcp:':
        batch_size = 60
        n_components = 70
        verbose = 50
    else:
        batch_size = 20
        n_components = 20
        verbose = 10

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_single)(
        trace_folder_list[idx], train_data, test_data, reduction=reduction,
        alpha=alpha, mask=mask,
        raw=raw, init=init,
        verbose=verbose,
        n_components=n_components,
        batch_size=batch_size) for idx, (reduction, alpha) in enumerate(
        itertools.product(reduction_list, alpha_list)))


def gather():
    n_exp = len(os.listdir())

    results = []
    for i in range(n_exp):
        exp = json.load(open('experiment_%i/experiment.json' % i, 'r'))
        res = json.load(open('experiment_%i/callback.json' % i, 'r'))
        results.append(dict(**exp, **res))

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(111)
    for reduction, this_df in df.groupby('reduction'):
        for line in this_df.iterrows():
            ax.plot(df.ix[i, 'time'], df.ix[i, ''],
                    label = df.idx[i, 'reduction'])



if __name__ == '__main__':
    # dataset = 'hcp'
    # reduction_list = [1, 2, 4, 8, 12]
    # alpha_list = [1e-3, 1e-4]
    #
    dataset = 'adhd'
    reduction_list = [5]
    alpha_list = [1e-3]

    run(n_jobs=2, reduction_list=reduction_list,
        alpha_list=alpha_list,
        dataset=dataset)
