# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import json
import os
import time
from os.path import expanduser, join

import itertools
import numpy as np
from joblib import Parallel
from joblib import delayed
from modl.spca_fmri import SpcaFmri
from nilearn import datasets

import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show
from nilearn.image import index_img
from sklearn.cross_validation import train_test_split


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        return super(NumpyAwareJSONEncoder, self).default(obj)


def display_maps(index=None):
    name = 'components_' + index + '.png'
    fig, axes = plt.subplots(2, 1)
    plot_prob_atlas(spca_fmri.components_, view_type="filled_contours",
                    axes=axes[0])
    plot_stat_map(index_img(spca_fmri.components_, 0),
                  axes=axes[1],
                  colorbar=False,
                  threshold=0)
    plt.savefig(join(trace_folder, 'components.png'))


class Callback(object):
    def __init__(self, train_data,
                 trace_folder=None,
                 test_data=None,
                 plot=False):
        self.train_data = train_data
        self.test_data = test_data
        self.trace_folder = trace_folder
        if self.test_data is not None:
            self.test_obj = []
        self.train_obj = []
        self.time = []
        self.iter = []

        self.start_time = time.clock()
        self.test_time = 0

        self.profile = []

        self.plot = plot

    def display(self, spca_fmri):
        fig, axes = plt.subplots(2, 1)
        plot_prob_atlas(spca_fmri.components_, view_type="filled_contours",
                        axes=axes[0])
        plot_stat_map(index_img(spca_fmri.components_, 0),
                      axes=axes[1],
                      colorbar=False,
                      threshold=0)
        plt.savefig(join(self.trace_folder, 'components.png'))

        fig, axes = plt.subplots(2, 1, sharex=True)

        profile = np.array(self.profile)
        iter = np.array(self.iter)
        obj = np.array(self.train_obj)
        average_time = np.zeros_like(profile)
        average_time[1:] = (profile[1:] - profile[:-1]) \
                           / (iter[1:] - iter[:-1])[:, np.newaxis]

        axes[0].plot(iter[1:], obj[1:], marker='o')
        axes[1].plot(iter[1:], average_time[1:], marker='o')
        axes[1].legend(['Dx time', 'G time', 'Code time', 'Agg time',
                        'BCD time', 'Total', 'IO time'])
        axes[1].set_ylabel('Average time')
        axes[1].set_yscale('Log')
        axes[0].set_ylabel('Function value')

        plt.savefig(join(self.trace_folder, 'loss_profile.png'))

    def __call__(self, spca_fmri):
        test_time = time.clock()
        train_obj = spca_fmri.score(self.train_data)
        self.train_obj.append(train_obj)
        if self.test_data is not None:
            test_obj = spca_fmri.score(self.test_data)
            self.test_obj.append(test_obj)
        if self.trace_folder is not None:
            spca_fmri.components_.to_filename(join(self.trace_folder,
                                                   "record_"
                                                   "%s.nii.gz"
                                                   % spca_fmri.n_iter))
            with open(join(self.trace_folder, 'callback.json'), 'w+') as f:
                json.dump({'iter': self.iter,
                           'time': self.time,
                           'profile': self.profile}, f,
                          cls=NumpyAwareJSONEncoder)
        self.iter.append(spca_fmri.n_iter)
        self.profile.append(spca_fmri.time)

        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)

        if self.plot:
            test_time = time.clock()
            self.display(spca_fmri)
            self.test_time += time.clock() - test_time


def fetch_data():
    print('Fetching data')
    adhd_dataset = datasets.fetch_adhd(n_subjects=40)

    data = adhd_dataset.func  # list of 4D nifti files for each subject
    train_data, test_data = train_test_split(data, test_size=0.1,
                                             random_state=0)

    print('Train data length: %i. Test data length: %i' % (len(train_data),
                                                           len(test_data)))
    return train_data, test_data


def run(trace_folder, n_jobs=1):
    train_data, test_data = fetch_data()

    reduction_list = [2, 4]
    alpha_list = [1e-3]

    n_exp = len(reduction_list) * len(alpha_list)

    trace_folder_list = []

    for i in range(n_exp):
        this_trace_folder = join(trace_folder, 'experiment_%i' % i)
        try:
            os.makedirs(this_trace_folder)
        except OSError:
            pass
        trace_folder_list.append(this_trace_folder)

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_single)(
        trace_folder_list[idx], train_data, test_data, reduction=reduction,
        alpha=alpha) for idx, (reduction, alpha) in enumerate(
        itertools.product(reduction_list, alpha_list)))


def run_single(trace_folder,
               train_data,
               test_data,
               n_jobs=1,
               **kwargs):
    cb = Callback(train_data, test_data=test_data,
                  trace_folder=trace_folder, plot=True)
    spca_fmri = SpcaFmri(smoothing_fwhm=6.,
                         memory=expanduser("~/nilearn_cache"), memory_level=2,
                         verbose=5,
                         random_state=0,
                         learning_rate=.8,
                         batch_size=50,
                         AB_agg='async',
                         G_agg='full',
                         Dx_agg='average',
                         offset=0,
                         shelve=True,
                         n_epochs=1,
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
    spca_fmri.fit(train_data)
    spca_fmri.callback(spca_fmri)


if __name__ == '__main__':
    trace_folder = expanduser('~/output/modl/adhd_reduced')

    try:
        os.makedirs(trace_folder)
    except OSError:
        pass

    run(trace_folder, n_jobs=2)
