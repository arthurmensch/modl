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
from math import log, sqrt
from os.path import join

import numpy as np
from nibabel.filebasedimages import ImageFileError
from nilearn._utils import CacheMixin
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

from ..input_data.fmri.base import BaseNilearnEstimator

from .dict_fact import DictFact, Coder

warnings.filterwarnings('ignore', module='scipy.ndimage.interpolation',
                        category=UserWarning,
                        )
warnings.filterwarnings('ignore', module='sklearn.cross_validation',
                        category=DeprecationWarning,
                        )


class fMRICoderMixin(BaseNilearnEstimator, CacheMixin, TransformerMixin):
    def __init__(self,
                 n_components=20,
                 alpha=0.1,
                 dict_init=None,
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
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
        if imgs is not None:
            BaseNilearnEstimator.fit(self, imgs, confounds=confounds)
        elif self.dict_init is not None:
            BaseNilearnEstimator.fit(self, self.dict_init)
        else:
            BaseNilearnEstimator.fit(self)

        self.components_ = _check_dict_init(self.dict_init,
                                            mask_img=self.mask_img_,
                                            n_components=self.n_components)
        if self.components_ is not None:
            self.components_img_ = self.masker_.inverse_transform(
                self.components_)
            self.coder_ = Coder(dictionary=self.components_,
                                code_alpha=self.alpha,
                                code_l1_ratio=0,
                                n_threads=self.n_jobs).fit()

    def score(self, imgs, confounds=None):
        """
        Score the images on the learning spatial pipelining, based on the
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
        if (isinstance(imgs, str) or not hasattr(imgs, '__iter__')):
            imgs = [imgs]
        if confounds is None:
            confounds = itertools.repeat(None)
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cache(_score_img, func_memory_level=1))(
                self.coder_, self.masker_, img, these_confounds)
            for img, these_confounds in zip(imgs, confounds))
        scores = np.array(scores)
        try:
            len_imgs = np.array([check_niimg(img).get_shape()[3]
                                 for img in imgs])
        except ImageFileError:
            len_imgs = np.array([np.load(img, mmap_mode='r').shape[0]
                                 for img in imgs])
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
        if (isinstance(imgs, str) or not hasattr(imgs, '__iter__')):
            imgs = [imgs]
        if confounds is None:
            confounds = itertools.repeat(None)
        codes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cache(_transform_img, func_memory_level=1))(
                self.coder_, self.masker_, img, these_confounds)
            for img, these_confounds in zip(imgs, confounds))
        return codes


class fMRIDictFact(fMRICoderMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    n_components: int
        Number of pipelining to extract

    n_epochs: int
        number of time to cycle over images

    alpha: float
        Penalty to apply. The larger, the sparser the pipelining will be in
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
        Subsampling to use in streaming data. The larger, the _faster the
        algorithm will go over data.
        Too large reduction may lead to slower convergence

    learning_rate: float in [0.917, 1[
        Learning rate to use in streaming data. 1 means to not forget about past
        iterations, when slower value leads to _faster forgetting. Convergence
        is not guaranteed below 0.917.

    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of pipelining to extract

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

    def __init__(self,
                 method='masked',
                 step_size=1,
                 n_components=20,
                 n_epochs=1,
                 alpha=0.1,
                 dict_init=None,
                 random_state=None,
                 batch_size=20,
                 reduction=1,
                 learning_rate=1,
                 positive=False,
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
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
        self.step_size = step_size
        self.positive = positive
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
        # Base logic for pipelining estimators
        if imgs is None:
            raise ValueError('imgs is None, use fMRICoder instead')

        # Fit mask + pipelining
        fMRICoderMixin.fit(self, imgs, confounds=confounds)

        self.components_ = self._cache(_compute_components,
                                       func_memory_level=1,
                                       ignore=['n_jobs',
                                               'verbose'])(
            self.masker_, imgs,
            step_size=self.step_size,
            confounds=confounds,
            dict_init=self.components_,
            alpha=self.alpha,
            reduction=self.reduction,
            learning_rate=self.learning_rate,
            n_components=self.n_components,
            batch_size=self.batch_size,
            positive=self.positive,
            n_epochs=self.n_epochs,
            method=self.method,
            verbose=self.verbose,
            random_state=self.random_state,
            callback=self.callback,
            n_jobs=self.n_jobs)
        self.components_img_ = self.masker_.inverse_transform(self.components_)
        self.coder_ = Coder(dictionary=self.components_,
                            code_alpha=self.alpha,
                            code_l1_ratio=0,
                            n_threads=self.n_jobs).fit()
        return self


class fMRICoder(fMRICoderMixin):
    def __init__(self, dictionary,
                 alpha=0.1,
                 transform_batch_size=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
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


def _check_dict_init(dict_init, mask_img, n_components=None):
    if dict_init is not None:
        if isinstance(dict_init, np.ndarray):
            assert (dict_init.shape[1] == mask_img.get_data().sum())
            components = dict_init
        else:
            masker = NiftiMasker(smoothing_fwhm=None,
                                 mask_img=mask_img).fit()
            components = masker.transform(dict_init)
        if n_components is not None:
            return components[:n_components]
        else:
            return components
    else:
        return None


def _compute_components(masker,
                        imgs,
                        step_size=1,
                        confounds=None,
                        dict_init=None,
                        alpha=1,
                        positive=False,
                        reduction=1,
                        learning_rate=1,
                        n_components=20,
                        batch_size=20,
                        n_epochs=1,
                        method='masked',
                        verbose=0,
                        random_state=None,
                        callback=None,
                        n_jobs=1):
    methods = {'masked': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               'dictionary only': {'G_agg': 'full', 'Dx_agg': 'full'},
               'gram': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               # 1st epoch parameters
               'average': {'G_agg': 'average', 'Dx_agg': 'average'},
               'reducing ratio': {'G_agg': 'masked', 'Dx_agg': 'masked'}}

    masker._check_fitted()
    dict_init = _check_dict_init(dict_init, mask_img=masker.mask_img_,
                                 n_components=n_components)
    # dict_init might have fewer components than asked for
    if dict_init is not None:
        n_components = dict_init.shape[0]
    random_state = check_random_state(random_state)
    if method == 'sgd':
        optimizer = 'sgd'
        G_agg = 'full'
        Dx_agg = 'full'
        reduction = 1
    else:
        method = methods[method]
        G_agg = method['G_agg']
        Dx_agg = method['Dx_agg']
        optimizer = 'variational'

    if verbose:
        print("Scanning data")
    n_records = len(imgs)
    if confounds is None:
        confounds = itertools.repeat(None)
    data_list = list(zip(imgs, confounds))
    n_samples_list, dtype = _lazy_scan(imgs)
    indices_list = np.zeros(len(imgs) + 1, dtype='int')
    indices_list[1:] = np.cumsum(n_samples_list)
    n_samples = indices_list[-1] + 1
    n_voxels = np.sum(check_niimg(masker.mask_img_).get_data() != 0)

    if verbose:
        print("Learning...")
    dict_fact = DictFact(n_components=n_components,
                         code_alpha=alpha,
                         code_l1_ratio=0,
                         comp_l1_ratio=1,
                         comp_pos=positive,
                         reduction=reduction,
                         Dx_agg=Dx_agg,
                         optimizer=optimizer,
                         step_size=step_size,
                         G_agg=G_agg,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         random_state=random_state,
                         n_threads=n_jobs,
                         verbose=0)
    dict_fact.prepare(n_samples=n_samples, n_features=n_voxels,
                      X=dict_init, dtype=dtype)
    cpu_time = 0
    io_time = 0
    if n_records > 0:
        if verbose:
            verbose_iter_ = np.linspace(0, n_records * n_epochs, verbose)
            verbose_iter_ = verbose_iter_.tolist()
        current_n_records = 0
        for i in range(n_epochs):
            if verbose:
                print('Epoch %i' % (i + 1))
            if method == 'gram' and i == 5:
                dict_fact.set_params(G_agg='full',
                                     Dx_agg='average')
            if method == 'reducing ratio':
                reduction = 1 + (reduction - 1) / sqrt(i + 1)
                dict_fact.set_params(reduction=reduction)
            record_list = random_state.permutation(n_records)
            for record in record_list:
                if (verbose and verbose_iter_ and
                            current_n_records >= verbose_iter_[0]):
                    print('Record %i' % current_n_records)
                    if callback is not None:
                        callback(masker, dict_fact, cpu_time, io_time)
                    verbose_iter_ = verbose_iter_[1:]

                # IO bounded
                t0 = time.perf_counter()
                img, these_confounds = data_list[record]
                masked_data = masker.transform(img, confounds=these_confounds)
                masked_data = masked_data.astype(dtype)
                io_time += time.perf_counter() - t0

                # CPU bounded
                t0 = time.perf_counter()
                permutation = random_state.permutation(
                    masked_data.shape[0])
                sample_indices = np.arange(
                    indices_list[record], indices_list[record + 1])
                sample_indices = sample_indices[permutation]
                masked_data = masked_data[permutation]
                dict_fact.partial_fit(masked_data,
                                      sample_indices=sample_indices)
                current_n_records += 1
                cpu_time += time.perf_counter() - t0
    components = _flip(dict_fact.components_)
    return components


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
        try:
            img = check_niimg(img)
            this_n_samples = img.shape[3]
            dtype = img.get_data_dtype()
        except ImageFileError:
            img = np.load(img, mmap_mode='r')
            this_n_samples = img.shape[0]
            dtype = img.dtype
        n_samples_list.append(this_n_samples)
    return n_samples_list, dtype


def _transform_img(coder, masker, img, confounds):
    data = masker.transform(img,
                            confounds=confounds)
    return coder.transform(data)


def _score_img(coder, masker, img, confounds):
    data = masker.transform(img, confounds=confounds)
    return coder.score(data)


class rfMRIDictionaryScorer:
    """Base callback to compute test score"""

    def __init__(self, test_imgs, test_confounds=None,
                 info=None, artifact_dir=None):
        self.start_time = time.perf_counter()
        self.test_imgs = test_imgs
        if test_confounds is None:
            test_confounds = itertools.repeat(None)
        self.test_confounds = test_confounds
        self.test_time = 0
        self.score = []
        self.iter = []
        self.time = []
        self.cpu_time = []
        self.io_time = []
        self.info = info
        self.artifact_dir = artifact_dir

    def __call__(self, masker, dict_fact, cpu_time, io_time):
        test_time = time.perf_counter()
        if not hasattr(self, 'data'):
            self.data = masker.transform(self.test_imgs,
                                         confounds=self.test_confounds)
        scores = np.array([dict_fact.score(data) for data in self.data])
        len_imgs = np.array([data.shape[0] for data in self.data])

        score = np.sum(scores * len_imgs) / np.sum(len_imgs)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.cpu_time.append(cpu_time)
        self.io_time.append(io_time)
        self.iter.append(dict_fact.n_iter_)
        if self.info is not None:
            self.info['time'] = self.cpu_time
            self.info['score'] = self.score
            self.info['iter'] = self.iter

        if self.artifact_dir is not None:
            components = _flip(dict_fact.components_)
            components_img = masker.inverse_transform(components)
            components_img.to_filename(join(self.artifact_dir, 'components_%i.nii.gz')
                                       % dict_fact.n_iter_)