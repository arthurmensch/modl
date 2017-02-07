"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
import time
import warnings
from math import log, sqrt, ceil

import nibabel
import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.niimg_conversions import _iter_check_niimg
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state, gen_batches

from ..input_data.fmri import BaseNilearnEstimator

from .dict_fact import DictFact, Coder

warnings.filterwarnings('ignore', module='scipy.ndimage.interpolation',
                        category=UserWarning,
                        )
warnings.filterwarnings('ignore', module='sklearn.cross_validation',
                        category=DeprecationWarning,
                        )


class fMRICoderMixin(BaseNilearnEstimator, TransformerMixin):
    def __init__(self,
                 n_components=20,
                 alpha=0.1,
                 dict_init=None,
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None),
                 memory_level=2,
                 n_jobs=1, verbose=0, ):
        BaseNilearnEstimator.__init__(self,
                                      mask=mask,
                                      smoothing_fwhm=smoothing_fwhm,
                                      standardize=standardize,
                                      detrend=detrend,
                                      low_pass=low_pass,
                                      high_pass=high_pass,
                                      t_r=t_r,
                                      target_affine=target_affine,
                                      target_shape=target_shape,
                                      mask_strategy=mask_strategy,
                                      mask_args=mask_args,
                                      memory=memory,
                                      memory_level=memory_level,
                                      n_jobs=n_jobs,
                                      verbose=verbose)

        self.n_components = n_components
        self.transform_batch_size = transform_batch_size
        self.dict_init = dict_init
        self.alpha = alpha

    def fit(self, imgs=None, y=None, confounds=None):
        BaseNilearnEstimator.fit(self, imgs, confounds=confounds)

        if self.dict_init is not None:
            if self.verbose:
                print("Loading dictionary")
            masker = NiftiMasker(smoothing_fwhm=0,
                                 mask_img=self.mask_img_).fit()
            self.components_ = masker.transform(self.dict_init)
            if self.n_components is not None:
                self.components_ = self.components_[:self.n_components]
        else:
            self.components_ = None

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
        score: float
            Average score on all input data
        """
        if isinstance(imgs, np.ndarray) and imgs.ndim == 2:
            assert (imgs.shape[1] == self.masker_.mask_img_.get_data().sum())
            raw = True
        elif (isinstance(imgs, str) or not hasattr(imgs, '__iter__')):
            imgs = [imgs]
        opt_batch_size = int(ceil(len(imgs) / self.n_jobs))
        if self.transform_batch_size is None:
            batch_size = opt_batch_size
        else:
            batch_size = min(opt_batch_size, self.transform_batch_size)
        # In case fit is not finished
        if confounds is None:
            confounds = itertools.repeat(None)
        batches = list(gen_batches(len(imgs), batch_size))
        prev_n_jobs = self.masker_.n_jobs
        self.masker_.set_params(n_jobs=1)
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_score_img)(self.dict_fact_, self.masker_,
                                imgs[batch], confounds[batch],
                                raw=raw)
            for batch in batches)
        self.masker_.set_params(n_jobs=prev_n_jobs)
        # Ravel
        scores = np.array(scores)
        len_imgs = np.array([check_niimg(img).get_shape()[3] for img in imgs])
        len_imgs = np.array([np.sum(len_imgs[batch]) for batch in batches])
        score = np.sum(scores * len_imgs) / np.sum(len_imgs)
        return score

    def transform(self, imgs, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        batch_size
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
        if isinstance(imgs, np.ndarray) and imgs.ndim == 2:
            assert (imgs.shape[1] == self.masker_.mask_img_.get_data().sum())
            raw = True
        if (isinstance(imgs, str) or not hasattr(imgs, '__iter__')):
            imgs = [imgs]
        opt_batch_size = int(ceil(len(imgs) / self.n_jobs))
        if self.transform_batch_size is None:
            batch_size = opt_batch_size
        else:
            batch_size = min(opt_batch_size, self.transform_batch_size)
        # In case fit is not finished
        if confounds is None:
            confounds = [None] * len(imgs)
        batches = list(gen_batches(len(imgs), batch_size))
        prev_n_jobs = self.masker_.n_jobs
        self.masker_.set_params(n_jobs=1)
        codes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_transform_img)(self.dict_fact_,
                                    self.masker_,
                                    imgs[batch], confounds[batch],
                                    raw=raw)
            for batch in batches)
        self.masker_.set_params(n_jobs=prev_n_jobs)
        codes = [code for batch_code in codes for code in batch_code]
        return codes


class fMRIDictFact(fMRICoderMixin):
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
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=2,
                 n_jobs=1, verbose=0,
                 callback=None):
        fMRICoderMixin.__init__(self, n_components=n_components,
                                alpha=alpha,
                                dict_init=dict_init,
                                mask=mask,
                                transform_batch_size=transform_batch_size,
                                smoothing_fwhm=smoothing_fwhm,
                                standardize=standardize,
                                detrend=detrend,
                                low_pass=low_pass,
                                high_pass=high_pass,
                                t_r=t_r,
                                target_affine=target_affine,
                                target_shape=target_shape,
                                mask_strategy=mask_strategy,
                                mask_args=mask_args,
                                memory=memory,
                                memory_level=memory_level,
                                n_jobs=n_jobs,
                                verbose=verbose)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reduction = reduction

        self.method = method

        self.learning_rate = learning_rate

        self.random_state = random_state

        self.callback = callback

    def fit(self, imgs=None, y=None, confounds=None):
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
        # Base logic for decomposition estimators
        fMRICoderMixin.fit(self, imgs)

        self.random_state = check_random_state(self.random_state)

        method = fMRIDictFact.methods[self.method]
        G_agg = method['G_agg']
        Dx_agg = method['Dx_agg']

        if imgs is not None:
            n_records = len(imgs)

            if self.verbose:
                print("Scanning data")

            if confounds is None:
                confounds = itertools.repeat(None)
            data_list = list(zip(imgs, confounds))
            n_samples_list, dtype = _lazy_scan(imgs)

            indices_list = np.zeros(len(imgs) + 1, dtype='int')
            indices_list[1:] = np.cumsum(n_samples_list)
            n_samples = indices_list[-1] + 1

            n_voxels = np.sum(
                check_niimg(self.masker_.mask_img_).get_data() != 0)
        else:
            raise ValueError('imgs is None, use fMRICoder instead')

        if self.verbose:
            print("Learning decomposition")

        if self.components_ is not None:
            n_components = self.components_.shape[0]
        else:
            n_components = self.n_components

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
                                X=self.components_, dtype=dtype)
        if n_records > 0:
            if self.verbose:
                log_lim = log(n_records * self.n_epochs, 10)
                self.verbose_iter_ = np.logspace(0, log_lim, self.verbose,
                                                 base=10) - 1
                self.verbose_iter_ = self.verbose_iter_.tolist()
            current_n_records = 0
            for i in range(self.n_epochs):
                if self.verbose:
                    print('Epoch %i' % (i + 1))
                if self.method == 'gram' and i == 2:
                    self.dict_fact_.set_params(G_agg='full',
                                               Dx_agg='average')
                if self.method == 'reducing ratio':
                    reduction = 1 + (self.reduction - 1) / sqrt(i + 1)
                    self.dict_fact_.set_params(reduction=reduction)
                record_list = self.random_state.permutation(n_records)
                for record in record_list:
                    if (self.verbose and self.verbose_iter_ and
                                current_n_records >= self.verbose_iter_[0]):
                        print('Record %i' % current_n_records)
                        if self.callback is not None:
                            self.callback(self)
                        self.verbose_iter_ = self.verbose_iter_[1:]

                    # IO bounded
                    data = data_list[record]
                    img, confound = data
                    img = check_niimg(img)
                    masked_data = self.masker_.transform(img, confound)

                    # CPU bounded
                    permutation = self.random_state. \
                        permutation(masked_data.shape[0])
                    masked_data = masked_data[permutation]
                    sample_indices = np.arange(
                        indices_list[record], indices_list[record + 1])
                    sample_indices = sample_indices[permutation]
                    masked_data = masked_data[permutation]
                    self.dict_fact_.partial_fit(masked_data,
                                                sample_indices=sample_indices)
                    self.components_ = self.dict_fact_.components_
                    current_n_records += 1
        components = _flip(self.components_)
        self.components_img_ = self.masker_.inverse_transform(components)
        return self

    def _callback(self):
        if self.callback is not None:
            self.callback(self)


class fMRICoder(fMRICoderMixin):
    def __init__(self, dictionary,
                 alpha=0.1,
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None),
                 memory_level=2,
                 n_jobs=1, verbose=0, ):
        self.dictionary = dictionary
        fMRICoderMixin.__init__(self,
                                n_components=None,
                                alpha=alpha,
                                dict_init=self.dictionary,
                                mask=mask,
                                smoothing_fwhm=smoothing_fwhm,
                                standardize=standardize,
                                detrend=detrend,
                                low_pass=low_pass,
                                high_pass=high_pass,
                                transform_batch_size=transform_batch_size,
                                t_r=t_r,
                                target_affine=target_affine,
                                target_shape=target_shape,
                                mask_strategy=mask_strategy,
                                mask_args=mask_args,
                                memory=memory,
                                memory_level=memory_level,
                                n_jobs=n_jobs,
                                verbose=verbose)

    def fit(self, imgs=None, y=None, confounds=None):
        fMRICoderMixin.fit(self, imgs, confounds=confounds)
        self.dict_fact_ = Coder(dictionary=self.components_,
                                code_alpha=self.alpha,
                                code_l1_ratio=0,
                                comp_l1_ratio=1,
                                n_threads=self.n_jobs).fit()
        self.components_ = self.dict_fact_.components_
        self.components_img_ = self.masker_.inverse_transform(self.components_)
        return self


def _flip(components):
    """Flip signs in each composant positive part is l1 larger
    than negative part"""
    components = components.copy()
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
            this_n_samples = nibabel.load(img).shape[3]
        else:
            this_n_samples = img.shape[3]
        n_samples_list.append(this_n_samples)
    if isinstance(img, str):
        dtype = nibabel.load(img).get_data_dtype()
    else:
        dtype = imgs[0].get_data_dtype()
    return n_samples_list, dtype


def _transform_img(coding_mixin, masker, imgs, confounds, raw=False):
    if raw:
        data = imgs
    else:
        imgs = list(_iter_check_niimg(imgs))
        data = masker.transform(imgs, confounds)
    data = np.concatenate(data)
    return coding_mixin.transform(data)


def _score_img(coding_mixin, masker, imgs, confounds, raw=False):
    if raw:
        data = imgs
    else:
        imgs = list(_iter_check_niimg(imgs))
        data = masker.transform(imgs, confounds)
    data = np.concatenate(data)
    return coding_mixin.score(data)


def fmri_dict_learning(imgs, confounds=None,
                       mask=None, *,
                       dict_init=None,
                       alpha=1,
                       batch_size=20,
                       learning_rate=1,
                       n_components=20,
                       n_epochs=1,
                       n_jobs=1,
                       reduction=1,
                       smoothing_fwhm=4,
                       method='masked',
                       random_state=None,
                       callback=None,
                       memory=Memory(cachedir=None),
                       memory_level=1,
                       verbose=0):
    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             mask=mask,
                             memory=memory,
                             method=method,
                             memory_level=memory_level,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=random_state,
                             n_components=n_components,
                             dict_init=dict_init,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=callback,
                             )
    dict_fact.fit(imgs, confounds)
    return dict_fact.components_img_, dict_fact.masker_.mask_img_, callback


def compute_loadings(imgs, components, alpha=1, confounds=None,
                     transform_batch_size=None,
                     mask=None, n_jobs=1, verbose=0,
                     raw=False):
    dict_fact = fMRICoder(mask=mask,
                          dictionary=components,
                          alpha=alpha,
                          transform_batch_size=transform_batch_size,
                          verbose=verbose,
                          n_jobs=n_jobs).fit(imgs, confounds=confounds)
    loadings = dict_fact.transform(imgs, confounds=confounds, raw=raw)
    return loadings


class rfMRIDictionaryScorer:
    """Base callback to compute test score"""

    def __init__(self, test_imgs, test_confounds=None):
        self.start_time = time.perf_counter()
        self.test_imgs = test_imgs
        if test_confounds is None:
            test_confounds = itertools.repeat(None)
        self.test_confounds = test_confounds
        self.test_time = 0
        self.score = []
        self.iter = []
        self.time = []

    def __call__(self, dict_fact):
        test_time = time.perf_counter()
        if not hasattr(self, 'data'):
            self.data = dict_fact.masker_.transform(self.test_imgs,
                                                    confounds=self.test_confounds)
        scores = np.array([dict_fact.dict_fact_.score(data)
                           for data in self.data])
        len_imgs = np.array([data.shape[0] for data in self.data])
        score = np.sum(scores * len_imgs) / np.sum(len_imgs)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.iter.append(dict_fact.dict_fact_.n_iter_)
