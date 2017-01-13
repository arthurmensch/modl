"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
import warnings

import nibabel
import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.cache_mixin import CacheMixin
from nilearn.decomposition.base import BaseDecomposition
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state

from .dict_fact import DictFact
from math import log


class fMRIDictFact(BaseDecomposition, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    n_components: int
        Number of components to extract

    n_epochs: int
        number of time to cycle over images

    alpha: float
        Penalty to apply. The larger, the sparser the decomposition will be in
        space

    dict_init: Niimg-like or None
        Initial dictionary (e.g, from ICA). If None, a random initialization
        will be used

    random_state: RandomState or int,
        Control randomness of the algorithm. Different value will lead to different,
        although rather equivalent, decompositions.

    batch_size: int,
        Number of 3D-image to use at each iteration

    reduction: float, > 1
        Subsampling to use in streaming data. The larger, the faster the
        algorithm will go over data.
        Too large reduction may lead to slower convergence

    learning_rate: float in [0.917, 1[
        Learning rate to use in streaming data. 1 means to not forget about past
        iterations, when slower value leads to faster forgetting. Convergence
        is not guaranteed below 0.917.

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

    methods = {'masked': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               'dictionary only': {'G_agg': 'full', 'Dx_agg': 'full'},
               'gram': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               # 1st epoch parameters
               'average': {'G_agg': 'average', 'Dx_agg': 'average'},
               'reducing ratio': {'G_agg': 'masked', 'Dx_agg': 'masked'}}

    def __init__(self,
                 method='masked',
                 n_components=20,
                 n_epochs=1,
                 alpha=0.1,
                 dict_init=None,
                 random_state=None,
                 batch_size=20,
                 reduction=1,
                 learning_rate=1,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=2,
                 buffer_size=None,
                 n_jobs=1, verbose=0,
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
        self.alpha = alpha
        self.dict_init = dict_init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reduction = reduction

        self.method = method

        self.learning_rate = learning_rate

        self.buffer_size = buffer_size
        self.callback = callback

    def fit(self, imgs=None, y=None, confounds=None, raw=False):
        """Compute the mask and the dictionary maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        -------
        self
        """
        if imgs is None or self.n_epochs == 0:
            # Will raise error is mask has not been provided
            if self.mask is None:
                raise ValueError('Please provide a mask if not'
                                 ' providing data.')
            BaseDecomposition.fit(self, None)
            if self.dict_init is not None:
                if self.verbose:
                    print("Preloading initial dictionary")
                masker = NiftiMasker(smoothing_fwhm=0,
                                     mask_img=self.mask_img_).fit()
                dict_init = masker.transform(self.dict_init)
                if self.n_components is not None:
                    dict_init = dict_init[:self.n_components]
                n_components, n_voxels = dict_init.shape
                dtype = dict_init.dtype
                n_samples = 0
                self.dict_fact_ = DictFact(n_components=n_components,
                                           code_alpha=self.alpha,
                                           code_l1_ratio=0,
                                           comp_l1_ratio=1,
                                           reduction=self.reduction,
                                           Dx_agg='masked',
                                           G_agg='masked',
                                           learning_rate=self.learning_rate,
                                           batch_size=self.batch_size,
                                           random_state=self.random_state,
                                           n_threads=self.n_jobs,
                                           verbose=0)
                self.dict_fact_.prepare(n_samples=n_samples,
                                        n_features=n_voxels,
                                        X=dict_init, dtype=dtype)
                return self
            else:
                raise ValueError

        if isinstance(imgs, str) or not hasattr(imgs, '__iter__'):
            imgs = [imgs]
        raw = isinstance(imgs[0], np.ndarray)
        # Base logic for decomposition estimators
        if raw:
            BaseDecomposition.fit(self, None)
        else:
            BaseDecomposition.fit(self, imgs)

        shelving = self.memory is not None and self.memory.cachedir is not None
        self.masker_._shelving = shelving
        # Avoid cache trashing due to nibabel bad design
        _ = self.masker_.mask_img_.get_data()

        self.random_state = check_random_state(self.random_state)

        method = fMRIDictFact.methods[self.method]
        G_agg = method['G_agg']
        Dx_agg = method['Dx_agg']

        n_records = len(imgs)

        if self.verbose:
            print("Preloading data")

        if raw:  # Hack
            data_list = imgs
            dtype = np.load(imgs[0], mmap_mode='r').dtype
            n_samples_list = [np.load(img, mmap_mode='r').shape[0] for img in
                              imgs]
        else:
            if confounds is None:
                confounds = itertools.repeat(None)
            if shelving:
                data_list = self.masker_.transform(imgs, confounds)
                n_samples_list = [data.get().shape[0] for data in data_list]
                dtype = data_list[0].get().dtype
            else:
                data_list = list(zip(imgs, confounds))
                n_samples_list, dtype = _lazy_scan(imgs)

        indices_list = np.zeros(len(imgs) + 1, dtype='int')
        indices_list[1:] = np.cumsum(n_samples_list)
        n_samples = indices_list[-1] + 1

        n_voxels = np.sum(check_niimg(self.masker_.mask_img_).get_data() != 0)
        if self.dict_init is not None:
            if self.verbose:
                print("Preloading initial dictionary")
            masker = NiftiMasker(smoothing_fwhm=0,
                                 mask_img=self.mask_img_).fit()
            dict_init = masker.transform(self.dict_init)
            if self.n_components is not None:
                dict_init = dict_init[:self.n_components]
            n_components = dict_init.shape[0]
        else:
            dict_init = None
            n_components = self.n_components

        if self.verbose:
            print("Learning decomposition")

        self.dict_fact_ = DictFact(n_components=n_components,
                                   code_alpha=self.alpha,
                                   code_l1_ratio=0,
                                   comp_l1_ratio=1,
                                   reduction=self.reduction,
                                   Dx_agg=Dx_agg,
                                   G_agg=G_agg,
                                   learning_rate=self.learning_rate,
                                   batch_size=self.batch_size,
                                   random_state=self.random_state,
                                   n_threads=self.n_jobs,
                                   verbose=0)
        self.dict_fact_.prepare(n_samples=n_samples, n_features=n_voxels,
                                X=dict_init, dtype=dtype)

        if self.verbose:
            log_lim = log(n_records * self.n_epochs, 10)
            self.verbose_iter_ = np.logspace(0, log_lim, self.verbose,
                                             base=10) - 1
            self.verbose_iter_ = self.verbose_iter_.tolist()
        current_n_records = 0
        for i in range(self.n_epochs):
            if self.verbose:
                print('Epoch %i' % (i + 1))
            record_list = self.random_state.permutation(n_records)
            for record in record_list:
                if (self.verbose and self.verbose_iter_ and
                        current_n_records >= self.verbose_iter_[0]):
                    print('Record %i' % current_n_records)
                    if self.callback is not None:
                        self.callback(self)
                    self.verbose_iter_ = self.verbose_iter_[1:]
                data = data_list[record]
                sample_indices = np.arange(indices_list[record],
                                           indices_list[record + 1])
                if raw:
                    data = np.load(data, mmap_mode='r')
                else:
                    if shelving:
                        data = data.get()
                    else:
                        img, confound = data
                        data = self.masker_.transform(img, confound)
                permutation = self.random_state.permutation(n_records)
                data = data[permutation]
                sample_indices = sample_indices[permutation]
                self.dict_fact_.partial_fit(data,
                                            sample_indices=sample_indices)
                current_n_records += 1

        self.masker_._shelving = False

        return self

    def score(self, imgs, confounds=None):
        """
        Score the images on the learning spatial components, based on the
        objective function value that is minimized by the algorithm. Lower
        means better fit.

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        -------
        scores: float
            Average score on all input data
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        raw = isinstance(imgs[0], np.ndarray)

        shelving = self.masker_._shelving

        score = 0.
        if raw:
            data_list = imgs
        else:
            if confounds is None:
                confounds = itertools.repeat(None)
            data_list = self.masker_.transform(imgs, confounds)  # shelved
        for idx, data in enumerate(data_list):
            if raw:
                data = np.load(data)
            else:
                if shelving:
                    data = data.get()
            score += self.dict_fact_.score(data)
        score /= len(data_list)
        return score

    def transform(self, imgs, confounds=None):
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

        Returns
        -------
        codes, list of ndarray, shape = n_images * (n_samples, n_components)
            Loadings for each of the images, and each of the time steps
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        raw = isinstance(imgs[0], np.ndarray)

        shelving = self.masker_._shelving

        codes = []
        if raw:
            data_list = imgs
        else:
            if confounds is None:
                confounds = itertools.repeat(None)
            data_list = self.masker_.transform(imgs, confounds)  # shelved
        for idx, data in enumerate(data_list):
            if raw:
                data = np.load(data)
            else:
                if shelving:
                    data = data.get()
            codes.append(self.dict_fact_.transform(data))
        return codes

    @property
    def components_(self):
        """Property for callback purpose"""
        components = self.dict_fact_.components_
        components = _normalize_and_flip(components)
        return self.masker_.inverse_transform(components)

    @property
    def n_iter_(self):
        """Property for callback purpose"""
        return self.dict_fact_.n_iter_

    def _callback(self):
        if self.callback is not None:
            self.callback(self)


def _normalize_and_flip(components):
    """Flip signs in each composant positive part is l1 larger
    than negative part"""
    for component in components:
        if np.sum(component < 0) > np.sum(component > 0):
            component *= -1
    return components


def _lazy_scan(imgs):
    """Extracts number of samples and dtype
    from a 4D list of Niilike-image, without loading data"""
    n_samples_list = []
    for img in imgs:
        if isinstance(img, str):
            warnings.warn('Provide a cachedir for efficiency')
            this_n_samples = nibabel.load(img).shape[3]
        else:
            this_n_samples = img.shape[3]
        n_samples_list.append(this_n_samples)
    if isinstance(img, str):
        dtype = nibabel.load(img).get_data_dtype()
    else:
        dtype = imgs[0].get_data_dtype()
    return n_samples_list, dtype
