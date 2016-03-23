"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
from os.path import join

import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.cache_mixin import CacheMixin
from nilearn.decomposition.base import BaseDecomposition
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from ._utils.masking.multi_nifti_masker import MultiNiftiMasker
from .dict_fact import DictMF


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
                 batch_size=20,
                 reduction=1,
                 full_projection=False,
                 exact_E=None,
                 learning_rate=1,
                 offset=0,
                 impute=False,
                 shelve=True,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, backend='c', verbose=0,
                 trace_folder=None
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
        self.dict_init = dict_init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reduction = reduction
        self.full_projection = full_projection
        self.impute = impute
        self.backend = backend
        self.shelve = shelve
        self.trace_folder = trace_folder

        self.exact_E = exact_E
        self.learning_rate = learning_rate
        self.offset = offset

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

        # Cast to MultiNiftiMasker with shelving
        if self.shelve:
            masker_params = self.masker_.get_params()
            mask_img = self.masker_.mask_img_
            masker_params['mask_img'] = mask_img
            self.masker_ = MultiNiftiMasker(**masker_params).fit()
        random_state = check_random_state(self.random_state)

        n_epochs = int(self.n_epochs)
        if self.n_epochs < 1:
            raise ValueError('Number of n_epochs should be at least one,'
                             ' got {r}'.format(self.n_epochs))

        if confounds is None:
            confounds = itertools.repeat(None)

        if raw:
            data_list = imgs
        else:
            if self.shelve:
                data_list = self.masker_.transform(imgs, confounds,
                                                   shelve=True)
            else:
                data_list = list(zip(imgs, confounds))

        if self.impute:
            record_samples = [check_niimg(img).shape[3] for img in imgs]
            offset_list = np.zeros(len(imgs) + 1, dtype='int')
            offset_list[1:] = np.cumsum(record_samples)

        if self.dict_init is not None:
            dict_init = self.masker_.transform(self.dict_init)
        else:
            dict_init = None

        dict_mf = DictMF(n_components=self.n_components,
                         alpha=self.alpha,
                         reduction=self.reduction,
                         full_projection=self.full_projection,
                         exact_E=self.exact_E,
                         learning_rate=self.learning_rate,
                         offset=self.offset,
                         impute=self.impute,
                         n_samples=offset_list[-1] + 1 if self.impute else None,
                         batch_size=self.batch_size,
                         random_state=random_state,
                         dict_init=dict_init,
                         l1_ratio=1,
                         backend=self.backend,
                         verbose=max(0, self.verbose - 1))

        # Epoch logic
        data_idx = itertools.chain(*[random_state.permutation(
            len(imgs)) for _ in range(n_epochs)])

        for record, this_data_idx in enumerate(data_idx):
            this_data = data_list[this_data_idx]
            if self.impute:
                offset = offset_list[this_data_idx]
            if self.verbose:
                print('Streaming record %s' % record)
            if raw:
                this_data = np.load(this_data)
            else:
                if self.shelve:
                    this_data = this_data.get()
                else:
                    this_data = self.masker_.transform(this_data[0],
                                                       confounds=this_data[1])
            if self.impute:
                dict_mf.partial_fit(this_data, sample_subset=offset + np.arange(this_data.shape[0]))
            else:
                dict_mf.partial_fit(this_data)
            if record % 4 == 0:
                if self.trace_folder is not None:
                    components = dict_mf.Q_.copy()
                    _normalize_and_flip(components)

                    self.masker_.inverse_transform(
                        components).to_filename(join(self.trace_folder,
                                                           'record_%s.nii.gz' % record))

        self.components_ = dict_mf.Q_
        _normalize_and_flip(self.components_)
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


def _normalize_and_flip(components):
    # Post processing normalization
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    # flip signs in each composant positive part is l1 larger
    #  than negative part
    for component in components:
        if np.sum(component > 0) < np.sum(component < 0):
            component *= -1


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
