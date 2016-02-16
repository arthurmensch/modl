"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
from math import ceil

import numpy as np
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.niimg import _safe_get_data
from nilearn.decomposition.base import BaseDecomposition, \
    _mask_and_reduce_single
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, randomized_range_finder

from modl.dict_fact import DictMF


class fmriMF(BaseDecomposition, TransformerMixin, CacheMixin):
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
                 n_epochs=1, dict_init=None,
                 alpha=0.,
                 random_state=None,
                 batch_size=20,
                 reduction=1,
                 callback=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, backend='c', verbose=0,
                 ):
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
        self.n_epochs = n_epochs
        self.dict_init = dict_init
        self.batch_size = batch_size
        self.reduction = reduction
        self.callback = callback
        self.backend = backend

    def fit(self, imgs, y=None, confounds=None):
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

        random_state = check_random_state(self.random_state)

        n_epochs = int(self.n_epochs)
        if self.n_epochs < 1:
            raise ValueError('Number of n_epochs should be at least one,'
                             ' got {r}'.format(self.n_epochs))

        if confounds is None:
            confounds = itertools.repeat(None)

        dict_mf = DictMF(n_components=self.n_components,
                         alpha=self.alpha,
                         reduction=self.reduction,
                         batch_size=self.batch_size,
                         random_state=random_state,
                         l1_ratio=1,
                         backend=self.backend,
                         verbose=max(0, self.verbose - 1))

        data_list = mask_and_reduce(self.masker_, imgs, confounds,
                                    n_components=self.n_components,
                                    reduction_method=None,
                                    random_state=self.random_state,
                                    memory=self.memory,
                                    memory_level=
                                    max(0, self.memory_level - 1),
                                    as_shelved_list=True,
                                    n_jobs=self.n_jobs)

        data_list = itertools.chain(*[random_state.permutation(
            data_list) for _ in range(n_epochs)])
        for record, data in enumerate(data_list):
            if self.verbose:
                print('Streaming record %s' % record)
            data = data.get()
            dict_mf.partial_fit(data)

        self.components_ = dict_mf.Q_
        # Post processing normalization
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        # flip signs in each composant positive part is l1 larger
        #  than negative part
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1

        return self

    def _raw_score(self, data, per_component=False):
        if per_component is True:
            raise NotImplementedError
        if hasattr(self, 'components_'):
            component = self.components_
        else:
            raise ValueError('Fit is needed to score')
        return objective_function(data, component,
                                  alpha=self.alpha)


def objective_function(X, components, alpha=0.):
    """Score function based on dictionary learning objective function

        Parameters
        ----------
        X: ndarray,
            Holds single subject data to be tested against components
        components: ndarray,
            Dictionary to compute objective function from

        alpha: regularization

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """

    lr = Ridge(fit_intercept=False, alpha=alpha)
    lr.fit(components.T, X.T)
    residuals = X - lr.coef_.dot(components)
    return np.sum(residuals ** 2) + alpha * np.sum(lr.coef_ ** 2)


def mask_and_reduce(masker, imgs,
                    confounds=None,
                    reduction_ratio='auto',
                    reduction_method=None,
                    n_components=None, random_state=None,
                    memory_level=0,
                    memory=Memory(cachedir=None),
                    as_shelved_list=False,
                    n_jobs=1):
    """Mask and reduce provided 4D images with given masker.

    Uses a PCA (randomized for small reduction ratio) or a range finding matrix
    on time series to reduce data size in time direction. For multiple images,
    the concatenation of data is returned, either as an ndarray or a memorymap
    (useful for big datasets that do not fit in memory).

    Parameters
    ----------
    masker: NiftiMasker or MultiNiftiMasker
        Instance used to mask provided data.

    imgs: list of 4D Niimg-like objects
        See http://nilearn.github.io/manipulating_visualizing/manipulating_images.html#niimg.
        List of subject data to mask, reduce and stack.

    confounds: CSV file path or 2D matrix, optional
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    reduction_method: 'svd' | 'rf' | 'ss' | None

    reduction_ratio: 'auto' or float in [0., 1.], optional
        - Between 0. or 1. : controls compression of data, 1. means no
        compression
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components.

    n_components: integer, optional
        Number of components per subject to be extracted by dimension reduction

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    Returns
    ------
    data: ndarray or memorymap
        Concatenation of reduced data.
    """
    if not hasattr(imgs, '__iter__'):
        imgs = [imgs]

    if reduction_ratio == 'auto':
        if n_components is None:
            # Reduction ratio is 1 if
            # neither n_components nor ratio is provided
            reduction_ratio = 1
    else:
        if reduction_ratio is None:
            reduction_ratio = 1
        else:
            reduction_ratio = float(reduction_ratio)
        if not 0 <= reduction_ratio <= 1:
            raise ValueError('Reduction ratio should be between 0. and 1.,'
                             'got %.2f' % reduction_ratio)

    if confounds is None:
        confounds = itertools.repeat(confounds)

    if reduction_ratio == 'auto':
        n_samples = n_components
        reduction_ratio = None
    else:
        # We'll let _mask_and_reduce_single decide on the number of
        # samples based on the reduction_ratio
        n_samples = None

    if as_shelved_list:
        func = cache(_mask_and_reduce_single, memory=memory,
                     memory_level=memory_level,
                     func_memory_level=0).call_and_shelve
    else:
        func = _mask_and_reduce_single
    data_list = Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(func)(
                    masker,
                    img, confound,
                    reduction_ratio=reduction_ratio,
                    reduction_method=reduction_method,
                    n_samples=n_samples,
                    memory=memory,
                    memory_level=memory_level,
                    random_state=random_state
            ) for img, confound in zip(imgs, confounds))

    if as_shelved_list:
        return data_list
    else:
        subject_n_samples = [subject_data.shape[0]
                             for subject_data in data_list]

        n_samples = np.sum(subject_n_samples)
        n_voxels = np.sum(_safe_get_data(masker.mask_img_))
        data = np.empty((n_samples, n_voxels), order='F',
                        dtype='float64')

        current_position = 0
        for i, next_position in enumerate(np.cumsum(subject_n_samples)):
            data[current_position:next_position] = data_list[i]
            current_position = next_position
            # Clear memory as fast as possible: remove the reference on
            # the corresponding block of data
            data_list[i] = None
        return data


def _mask_and_reduce_single(masker,
                            img, confound,
                            reduction_ratio=None,
                            reduction_method=None,
                            n_samples=None,
                            memory=None,
                            memory_level=0,
                            random_state=None):
    """Utility function for multiprocessing from MaskReducer"""
    this_data = masker.transform(img, confound)
    # Now get rid of the img as fast as possible, to free a
    # reference count on it, and possibly free the corresponding
    # data
    del img
    random_state = check_random_state(random_state)

    data_n_samples, data_n_features = this_data.shape
    if reduction_ratio is None:
        assert n_samples is not None
        n_samples = min(n_samples, data_n_samples)
    else:
        n_samples = int(ceil(data_n_samples * reduction_ratio))
    if reduction_method == 'svd':
        if n_samples <= data_n_features // 4:
            U, S, _ = cache(randomized_svd, memory,
                            memory_level=memory_level,
                            func_memory_level=3)(this_data.T,
                                                 n_samples,
                                                 transpose=True,
                                                 random_state=random_state)
            U = U.T
        else:
            U, S, _ = cache(linalg.svd, memory,
                            memory_level=memory_level,
                            func_memory_level=3)(this_data.T,
                                                 full_matrices=False)
            U = U.T[:n_samples].copy()
            S = S[:n_samples]
        U = U * S[:, np.newaxis]
    elif reduction_method == 'rf':
        Q = cache(randomized_range_finder, memory,
                  memory_level=memory_level,
                  func_memory_level=3)(this_data,
                                       n_samples, n_iter=3,
                                       random_state=random_state)
        U = Q.T.dot(this_data)
    elif reduction_method == 'ss':
        indices = np.floor(np.linspace(0, this_data.shape[0] - 1,
                                       n_samples)).astype('int')
        U = this_data[indices]
    else:
        U = this_data

    return U