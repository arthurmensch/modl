from os.path import expanduser, join
import time

import itertools
import matplotlib.pyplot as plt
import numpy as np
import skimage
from joblib import Parallel, dump, load
from joblib import delayed
from modl._utils.hyperspectral import fetch_aviris
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import check_random_state
from math import log
from skimage.io import imread
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


class Callback(object):
    def __init__(self, train_data=None,
                 trace_folder=None,
                 test_data=None,
                 raw=False,
                 verbose_iter=None,
                 patch_size=(8, 8),
                 gray=True,
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

        self.components = []

        self.patch_size = patch_size
        self.gray = gray

    def display(self, mf):
        if len(self.profile) == 1:
            return

        V = mf.components_

        fig = plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(V[:100]):
            plt.subplot(10, 10, i + 1)
            if self.gray:
                imgs = comp.reshape((self.patch_size[0], self.patch_size[1]))
            else:
                imgs = comp.reshape((self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            plt.imshow(
                imgs[:, :, :3],
                cmap=plt.cm.gray_r,
                interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('Dictionary',
                     fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.savefig(join(self.trace_folder, 'components.pdf'))
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(self.iter, self.components)
        plt.savefig(join(self.trace_folder, 'components.pdf'))
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
        profile = profile[:, [0, 3, 2, 1, 4]]
        labels = np.array(
            ['', 'Dx time', 'Agg time', 'Code time', 'G time',
             'BCD time'])
        average_time = np.zeros((profile.shape[0] - 1, profile.shape[1] + 1))
        average_time[:, 1:] = (profile[1:] - profile[:-1]) \
                              / (iter[1:] - iter[:-1])[:, np.newaxis]
        average_time = np.cumsum(average_time, axis=1)

        palette = sns.color_palette("deep", 5)
        for i in range(1, 6):
            # axes[1].plot(iter, average_time[:, i],
            #              color=palette[i - 1])
            axes[1].fill_between(iter[1:], average_time[:, i],
                                 average_time[:, i - 1],
                                 facecolor=palette[i - 1], label=labels[i])

            # axes[1].plot(iter[1:], average_time[1:], marker='o')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(reversed(handles), reversed(labels), loc='upper left',
                       bbox_to_anchor=(1, 1), )
        axes[1].set_ylabel('Average time')
        # axes[1].set_yscale('Log')
        axes[0].set_ylabel('Function value')

        plt.savefig(join(self.trace_folder, 'loss_profile.pdf'))
        plt.close(fig)

    def __call__(self, mf):
        print('Testing...')
        test_time = time.clock()
        if self.train_data is not None:
            train_obj = mf.score(self.train_data, n_threads=1)
            self.train_obj.append(train_obj)
        if self.test_data is not None:
            test_obj = mf.score(self.test_data, n_threads=1)
            self.test_obj.append(test_obj)

        self.components.append(mf.components_[0, :100].copy())

        if self.trace_folder is not None:
            np.save(join(self.trace_folder, "record_%s" % mf.n_iter),
                    mf.components_)
            with open(join(self.trace_folder, 'callback.json'), 'w+') as f:
                json.dump({'iter': self.iter,
                           'time': self.time,
                           'train_obj': self.train_obj,
                           'test_obj': self.test_obj,
                           'profile': self.profile,
                           'components': self.components}, f,
                          cls=NumpyAwareJSONEncoder)
        self.iter.append(mf.n_iter)
        self.profile.append(mf.time)

        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)

        if self.trace_folder is not None and self.plot:
            test_time = time.clock()
            self.display(mf)
            self.test_time += time.clock() - test_time


def prepare_folder(name, n_exp, offset=0):
    trace_folder = expanduser('~/output/modl/' + name)
    try:
        os.makedirs(trace_folder)
    except OSError:
        pass
    trace_folder_list = []
    for i in range(offset, offset + n_exp):
        this_trace_folder = join(trace_folder, 'experiment_%i' % i)
        try:
            os.makedirs(this_trace_folder)
        except OSError:
            pass
        trace_folder_list.append(this_trace_folder)
    return trace_folder_list


def fetch_data(patch_size=(8, 8), random_state=0, gray=True):
    full_img = fetch_aviris()
    img = full_img

    n_channels = img.shape[2]
    height, width = img.shape[:-1]

    train_patches = extract_patches_2d(img[:, :width // 2, :], patch_size,
                                       max_patches=20000,
                                       random_state=random_state)
    train_patches = np.reshape(train_patches, (train_patches.shape[0], -1))
    train_patches -= np.mean(train_patches, axis=0)
    train_patches /= np.std(train_patches, axis=0)
    test_patches = extract_patches_2d(img[:, width // 2:, :], patch_size,
                                      max_patches=2000,
                                      random_state=random_state)
    test_patches = np.reshape(test_patches, (test_patches.shape[0], -1))
    test_patches -= np.mean(test_patches, axis=0)
    test_patches /= np.std(test_patches, axis=0)
    return train_patches, test_patches, (patch_size[0], patch_size[1], n_channels)
#
#
# def fetch_data(redundancy=1, patch_size=(8, 8), gray=True):
#     tile = int(sqrt(redundancy))
#     image = imread(expanduser('~/data/images/lisboa.jpg'))
#     # image = image[::2, ::2] + image[::2, ::2] + image[::2, ::2] + image[::2, ::2]
#     # image = image / 4
#     # if gray:
#     #     image = image.mean(axis=2)
#     #     height, width = image.shape
#     # else:
#     #     height, width, n_channels = image.shape
#     #
#     image = misc.face(gray=gray)
#     if gray:
#         height, width = image.shape
#     else:
#         height, width, n_channels = image.shape
#
#     image = image / 255
#
#     datasets = dict()
#     n_samples = dict(train_data=10000, test_data=2000)
#     for dataset in ['train_data', 'test_data']:
#         data = extract_patches_2d(image[:, :width // 2], patch_size,
#                                   max_patches=n_samples[dataset],
#                                   random_state=0)
#         if image.ndim == 3:
#             tiled_data = np.empty(
#                 (data.shape[0], data.shape[1] * tile,
#                  data.shape[2] * tile, 3))
#         else:
#             tiled_data = np.empty(
#                 (data.shape[0], data.shape[1] * tile,
#                  data.shape[2] * tile))
#         for i in range(tile):
#             for j in range(tile):
#                 tiled_data[:, i::tile, j::tile] = data
#         data = tiled_data
#         data = data.reshape(data.shape[0], -1)
#         data -= np.mean(data, axis=0)
#         data /= np.std(data, axis=0)
#         datasets[dataset] = data
#     patch_size = (patch_size[0] * tile, patch_size[1] * tile)
#
#     return datasets['train_data'], datasets['test_data'], patch_size


def run(exps, n_jobs=1, offset=0, redundancy=1, patch_size=(8, 8),
        gray=True):
    n_exp = len(exps)

    trace_folder_list = prepare_folder('denoise', n_exp, offset=offset)

    train_data, test_data, patch_size = fetch_data(
                                                   gray=gray,
                                                   patch_size=patch_size)

    Parallel(n_jobs=n_jobs, verbose=10, mmap_mode=None)(delayed(run_single)(
        trace_folder_list[idx], train_data=train_data, test_data=None,
        patch_size=patch_size,
        gray=gray,
        redundancy=redundancy,
        **this_exp) for idx, this_exp in enumerate(exps))


def run_single(trace_folder,
               train_data, test_data,
               G_agg='full', Dx_agg='full', AB_agg='full',
               patch_size=(8, 8),
               gray=True,
               redundancy=1,
               reduction=1,
               **kwargs):
    train_subset = check_random_state(0).permutation(len(train_data))[:len(train_data) // 10]
    cb = Callback(train_data=train_data[train_subset],
                  test_data=test_data, trace_folder=trace_folder,
                  patch_size=patch_size,
                  gray=gray,
                  plot=False)
    n_samples = train_data.shape[0]
    n_epochs = 20
    linear_verbose_epoch = 0
    verbose_iter = np.unique(np.floor(
        np.logspace(1, log(n_samples * (n_epochs - linear_verbose_epoch),
                           10), 40, base=10)).
                             astype('int') - 10)
    # verbose_iter_last = np.unique(np.floor(
    #     np.linspace(n_samples * (n_epochs - linear_verbose_epoch),
    #                 n_samples * n_epochs, 50)).
    #                          astype('int'))
    # verbose_iter = np.unique(np.concatenate([verbose_iter, verbose_iter_last]))
    dico = DictFact(n_components=50, alpha=.5,
                    l1_ratio=0,
                    pen_l1_ratio=0.9,
                    batch_size=150,
                    learning_rate=0.75,
                    sample_learning_rate=0.9,
                    reduction=reduction,
                    verbose=50,
                    verbose_iter=verbose_iter,
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
                    n_epochs=n_epochs,
                    **kwargs)

    relevent_params = ['batch_size', 'learning_rate', 'offset',
                       'AB_agg', 'G_agg', 'Dx_agg',
                       'reduction',
                       'alpha',
                       'pen_l1_ratio']

    exp_dict = {param: getattr(dico, param) for param in relevent_params}

    exp_dict['redundancy'] = redundancy
    exp_dict['patch_size'] = patch_size

    with open(join(trace_folder, 'experiment.json'), 'w+') as f:
        json.dump(exp_dict, f, cls=NumpyAwareJSONEncoder)

    dico.fit(train_data)
    cb.display(dico)


if __name__ == '__main__':
    exps = [
        dict(reduction=1, G_agg='full', Dx_agg='full', AB_agg='full'),
        dict(reduction=8, G_agg='average', Dx_agg='average', AB_agg='full'),
        dict(reduction=8, G_agg='masked', Dx_agg='masked', AB_agg='full'),
        # dict(reduction=4, G_agg='full', Dx_agg='average', AB_agg='async'),
        # dict(reduction=4, G_agg='full', Dx_agg='full', AB_agg='full'),
        # dict(reduction=4, G_agg='average', Dx_agg='average', AB_agg='full'),
        # dict(reduction=4, G_agg='masked', Dx_agg='masked', AB_agg='full'),
        # dict(reduction=4, G_agg='full', Dx_agg='average', AB_agg='full'),
    ]
    run(exps, n_jobs=3, offset=3200, redundancy=1, patch_size=(16, 16),
        gray=False)
    # run(exps, n_jobs=1, offset=3, redundancy=16, patch_size=(16, 16))
    # run(exps, n_jobs=1, offset=6, redundancy=100, patch_size=(16, 16))
