"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
import json
import time
from os.path import join

import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.cache_mixin import CacheMixin
from nilearn.decomposition.base import BaseDecomposition
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from .dict_fact import DictFact
from math import log, ceil


class SpcaFmri(BaseDecomposition, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    alpha: float, optional, default=1
        Sparsity controlling parameter

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    """

    def __init__(self, n_components=20,
                 n_epochs=1,
                 alpha=0.,
                 dict_init=None,
                 random_state=None,
                 l1_ratio=1,
                 batch_size=20,
                 reduction=1,
                 learning_rate=1,
                 offset=0,
                 shelve=True,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 buffer_size=None,
                 n_jobs=1, verbose=0,
                 G_agg='full',
                 AB_agg='masked',
                 subset_sampling='random',
                 Dx_agg='average', dict_reduction='follow',
                 temp_dir=None,
                 callback=None):
        BaseDecomposition.__init__(self, n_components=n_components,
                                   random_state=random_state,
                                   mask=mask,
                                   smoothing_fwhm=smoothing_fwhm,
                                   standardize=standardize,
                                   detrend=detrend,
                                   low_pass=low_pass, high_pass=high_pass,
                                   t_r=t_r,
                                   target_affine=target_affine,
                                   target_shape=target_shape,
                                   mask_strategy=mask_strategy,
                                   mask_args=mask_args,
                                   memory=memory,
                                   memory_level=memory_level,
                                   n_jobs=n_jobs, verbose=verbose,
                                   )
        self.G_agg = G_agg
        self.AB_agg = AB_agg
        self.Dx_agg = Dx_agg
        self.subset_sampling = subset_sampling
        self.dict_reduction = dict_reduction
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.dict_init = dict_init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reduction = reduction

        self.shelve = shelve

        self.learning_rate = learning_rate
        self.offset = offset

        self.temp_dir = temp_dir
        self.buffer_size = buffer_size
        self.callback = callback

    def fit(self, imgs, y=None, confounds=None, raw=False):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        # Base logic for decomposition estimators
        BaseDecomposition.fit(self, imgs)

        self._io_time = 0

        self.masker_._shelving = self.shelve

        random_state = check_random_state(self.random_state)

        n_epochs = int(self.n_epochs)
        if self.n_epochs < 1:
            raise ValueError('Number of n_epochs should be at least one,'
                             ' got {r}'.format(self.n_epochs))

        if raw:
            data_list = imgs
            n_samples_list = [np.load(img, mmap_mode='r').shape[0] for img in
                              imgs]
        else:
            if confounds is None:
                confounds = itertools.repeat(None)
            if self.shelve:
                data_list = self.masker_.transform(imgs, confounds)
            else:
                data_list = list(zip(imgs, confounds))
            n_samples_list = [check_niimg(img).shape[3] for img in imgs]

        offset_list = np.zeros(len(imgs) + 1, dtype='int')
        offset_list[1:] = np.cumsum(n_samples_list)
        n_samples = offset_list[-1] + 1

        n_voxels = np.sum(check_niimg(self.masker_.mask_img_).get_data() != 0)

        if self.dict_init is not None:
            dict_init = self.masker_.transform(self.dict_init)
        else:
            dict_init = None

        self._dict_fact = DictFact(n_components=self.n_components,
                                   alpha=self.alpha,
                                   reduction=self.reduction,
                                   dict_reduction=self.dict_reduction,
                                   AB_agg=self.AB_agg,
                                   G_agg=self.G_agg,
                                   Dx_agg=self.Dx_agg,
                                   subset_sampling=self.subset_sampling,
                                   learning_rate=self.learning_rate,
                                   offset=self.offset,
                                   n_samples=n_samples,
                                   batch_size=self.batch_size,
                                   random_state=random_state,
                                   dict_init=dict_init,
                                   n_threads=self.n_jobs,
                                   l1_ratio=self.l1_ratio,
                                   buffer_size=self.buffer_size,
                                   temp_dir=self.temp_dir,
                                   pen_l1_ratio=0,
                                   verbose=0)
        self._dict_fact._initialize((n_samples, n_voxels))
        # Preinit
        max_sample_size = max(n_samples_list)
        sample_subset_range = np.arange(max_sample_size, dtype='i4')

        data_array = np.empty((max_sample_size * 4, n_voxels),
                              dtype='float', order='C')

        sample_indices = np.empty(max_sample_size * 4, dtype='i4')

        # Epoch logic
        data_idx = list(itertools.chain(*[random_state.permutation(
            len(imgs)) for _ in range(n_epochs)]))

        if hasattr(self.verbose, '__iter__'):
            verbose_iter = np.array(self.verbose).astype('int')
        else:
            log_verbose = log(len(imgs) * self.n_epochs // 4, 10)
            verbose_iter = np.unique((np.logspace(0, log_verbose,
                                                  self.verbose) - 1e0).astype(
                'int')) * 4
            print(verbose_iter)

        for base_record in range(0, len(data_idx), 4):
            start = 0
            for i in range(4):
                this_record = base_record + i
                if this_record >= len(data_idx):
                    continue
                this_data_idx = data_idx[this_record]
                this_data = data_list[this_data_idx]
                this_n_samples = n_samples_list[this_data_idx]
                stop = start + this_n_samples
                offset = offset_list[this_data_idx]
                sample_indices[start:stop] = offset + sample_subset_range[:this_n_samples]
                if this_record in verbose_iter:
                    print('Streaming record %s' % this_record)
                    if self.callback is not None:
                        self.callback(self)
                t0 = time.time()
                if raw:
                    data_array[start:stop] = np.load(this_data,
                                                          mmap_mode='r')
                else:
                    if self.shelve:
                        data_array[start:stop] = this_data.get()
                    else:
                        data_array[start:stop] = self.masker_.transform(
                            this_data[0],
                            confounds=this_data[1])
                self._io_time += time.time() - t0
                start = stop
            self._dict_fact.partial_fit(data_array[:stop],
                                        sample_indices=sample_indices[:stop],
                                        check_input=False)
        return self

    @property
    def components_(self):
        components = self._dict_fact.components_
        components = _normalize_and_flip(components)
        return self.masker_.inverse_transform(components)

    @property
    def n_iter_(self):
        return self._dict_fact.n_iter_

    @property
    def profiling_(self):
        this_time = self._dict_fact.profiling_
        this_time = np.concatenate([this_time, np.array([self._io_time])])
        return this_time

    def score(self, imgs, confounds=None, raw=False):
        if self.verbose:
            print('Scoring...')
        score = 0
        if raw:
            data_list = imgs
        else:
            if confounds is None:
                confounds = itertools.repeat(None)
            if self.shelve:
                data_list = self.masker_.transform(imgs, confounds)
            else:
                data_list = list(zip(imgs, confounds))
        for idx, data in enumerate(data_list):
            if raw:
                data = np.load(data)
            elif self.shelve:
                data = data.get()
            score += self._dict_fact.score(data)
        score /= len(data_list)
        if self.verbose:
            print('Done.')
        return float(score)


def _normalize_and_flip(components):
    # Post processing normalization
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    # flip signs in each composant positive part is l1 larger
    #  than negative part
    for component in components:
        if np.sum(component < 0) < np.sum(component > 0):
            component *= -1
    return components

#
# def Batcher(object):
#     def __init__(self, n_features=100, nominal_size=100):
#         self.nominal_size = nominal_size
#         self.n_features = n_features
#         self.buffer_list = []
#         self.cursor = 0
#         self.available_size = 0
#
#     def _add_buffer(self, size):
#         size = size - self.available_size
#         n_buffers = int(ceil(size / self.nominal_size))
#         self._buffer_list += [self._new_buffer() for _ in range(n_buffers)]
#         self.available_size += n_buffers * self.nominal_size
#
#     def _new_buffer(self):
#         return np.empty((self.nominal_size, self.n_features))
#
#     @property
#     def ready(self):
#         return len(self.buffer_list) > 1
#
#     def pop(self):
#         if len(self.buffer_list) > 0:
#             return self.buffer_list[0][:self.cursor]
#         self.buffer_list = self.buffer_list[1:]
#
#     def pop_ready(self):
#         if self.ready:
#             return self.pop()
#         else:
#             return None
#
#     def add(self, array):
#         if array.shape[1] != self.n_features:
#             raise ValueError('Wrong input size')
#         else:
#             size = array.shape[0]
#             if size <= self.available_size:
#                 self.buffer_list[0][self.cursor:self.cursor + size] = array
#                 self.available_size -= size
#             else:
#                 self._add_buffers(size)
#                 first_cursor = self.nominal_size - self.cursor
#                 self.buffer_list[-1][self.cursor:] = array[:first_cursor]
#                 self.ready = True
#                 for i, current_cursor in enumerate(range(first_cursor, size,
#                                                          self.nominal_size)):
#                     new_cursor = current_cursor + self.nominal_size
#                     if new_cursor > size:
#                         # last iteration
#                         new_cursor = size
#                         self.cursor = size - new_cursor
#                         self.buffer_list[i + 1][:self.cursor] = array[current_cursor:new_cursor]
#                         self.available_size -= self.cursor
#                     else:
#                         self.buffer_list[i] = array[current_cursor:new_cursor]
#                         self.available_size -= self.nominal_size
#
# def nominal_batches(batches):
#     batcher = Batcher(nominal_size=nominal_size)
#     for batch in batches:


