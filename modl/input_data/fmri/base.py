import copy
import warnings

from nilearn._utils.class_inspect import get_params
from nilearn.input_data import MultiNiftiMasker
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Memory

from nilearn._utils.compat import _basestring
import numpy as np


class BaseNilearnEstimator(BaseEstimator):
    def __init__(self,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None),
                 memory_level=2,
                 n_jobs=1, verbose=0, ):
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, imgs=None, y=None, confounds=None):
        if isinstance(imgs, _basestring) or not hasattr(imgs, '__iter__'):
            # these classes are meant for list of 4D images
            # (multi-subject), we want it to work also on a single
            # subject, so we hack it.
            imgs = [imgs, ]
        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError('Need one or more Niimg-like objects as input, '
                             'an empty list was given.')
        self.masker_ = check_embedded_nifti_masker(self)

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        return self


def safe_to_filename(img, filename):
    img = copy.deepcopy(img)
    img.to_filename(filename)


def check_embedded_nifti_masker(estimator):
    """Base function for using a masker within a BaseEstimator class

    This creates a masker from instance parameters :
    - If instance contains a mask image in mask parameter,
    we use this image as new masker mask_img, forwarding instance parameters to
    new masker : smoothing_fwhm, standardize, detrend, low_pass= high_pass,
    t_r, target_affine, target_shape, mask_strategy, mask_args,
    - If instance contains a masker in mask parameter, we use a copy of
    this masker, overriding all instance masker related parameters.
    In all case, we forward system parameters of instance to new masker :
    memory, memory_level, verbose, n_jobs

    Parameters
    ----------
    instance: object, instance of BaseEstimator
        The object that gives us the values of the parameters

    multi_subject: boolean
        Indicates whether to return a MultiNiftiMasker or a NiftiMasker
        (the default is True)

    Returns
    -------
    masker: MultiNiftiMasker or NiftiMasker
        New masker
    """
    estimator_params = get_params(MultiNiftiMasker, estimator)
    mask = getattr(estimator, 'mask', None)

    if mask is not None and hasattr(mask, 'mask_img'):
        # Creating (Multi)NiftiMasker from provided masker
        masker_class = mask.__class__
        masker_params = get_params(MultiNiftiMasker, mask)
        new_masker_params = masker_params
    else:
        # Creating (Multi)NiftiMasker
        # with parameters extracted from estimator
        masker_class = MultiNiftiMasker
        new_masker_params = estimator_params
        new_masker_params['mask_img'] = mask
    # Forwarding technical params
    new_masker_params['n_jobs'] = estimator.n_jobs
    new_masker_params['memory'] = estimator.memory
    new_masker_params['memory_level'] = max(0, estimator.memory_level - 1)
    new_masker_params['verbose'] = estimator.verbose

    # Raising warning if masker override parameters
    conflict_string = ""
    for param_key in sorted(estimator_params):
        if np.any(new_masker_params[param_key] != estimator_params[param_key]):
            conflict_string += ("Parameter {0} :\n"
                                "    Masker parameter {1}"
                                " - overriding estimator parameter {2}\n"
                                ).format(param_key,
                                         new_masker_params[param_key],
                                         estimator_params[param_key])

    if conflict_string != "":
        warn_str = ("Overriding provided-default estimator parameters with"
                    " provided masker parameters :\n"
                    "{0:s}").format(conflict_string)
        warnings.warn(warn_str)

    masker = masker_class(**new_masker_params)

    # Forwarding potential attribute of provided masker
    if hasattr(mask, 'mask_img_'):
        # Allow free fit of returned mask
        masker.mask_img = mask.mask_img_

    return masker
