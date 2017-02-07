import copy
import os

import nibabel
import nilearn

from nibabel import load as nibabel_load
from nibabel import Nifti1Image as NibabelNifti1Image

from nilearn._utils.compat import _basestring, get_affine
from nilearn._utils.niimg import short_repr, _get_target_dtype
from nilearn.input_data import MultiNiftiMasker
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from sklearn.base import BaseEstimator

from sklearn.externals import joblib as joblib
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib.func_inspect import filter_args
from sklearn.externals.joblib.hashing import NumpyHasher


def load(filename, **kwargs):
    img = nibabel_load(filename, **kwargs)
    img.__class__ = Nifti1Image
    return img


class Nifti1Image(NibabelNifti1Image):
    def __getstate__(self):
        state = {'dataobj': self._dataobj,
                 'header': self.header,
                 'filename': self.get_filename(),
                 'affine': self.affine,
                 'extra': self.extra}
        return state

    def __setstate__(self, state):
        new_self = Nifti1Image(dataobj=state['dataobj'],
                               affine=state['affine'],
                               header=state['header'],
                               extra=state['extra'],
                               )
        self.__dict__ = new_self.__dict__
        if state['filename'] is not None:
            self.set_filename(state['filename'])


class NibabelHasher(NumpyHasher):
    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        NumpyHasher.__init__(self, hash_name=hash_name,
                             coerce_mmap=coerce_mmap)
        import nibabel as nibabel
        self.nibabel = nibabel

    def save(self, obj):
        if isinstance(obj, Nifti1Image):
            filename = obj.get_filename()
            if filename is not None:
                stat = os.stat(filename)
                last_modified = stat.st_mtime
                klass = obj.__class__
                obj = (klass, ('HASHED', filename, last_modified))
        NumpyHasher.save(self, obj)


def our_hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.


        Parameters
        -----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    hasher = NibabelHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    hash = hasher.hash(obj)
    return hash


def our_get_argument_hash(self, *args, **kwargs):
    return our_hash(filter_args(self.func, self.ignore,
                                args, kwargs),
                    coerce_mmap=True)


def our_load_niimg(niimg, dtype=None):
    """Load a niimg, check if it is a nibabel SpatialImage and cast if needed

    Parameters:
    -----------

    niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html.
        Image to load.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns:
    --------
    img: image
        A loaded image object.
    """
    from nilearn.image import new_img_like  # avoid circular imports

    if isinstance(niimg, _basestring):
        # data is a filename, we load it
        niimg = nibabel.load(niimg)
    elif not isinstance(niimg, nibabel.spatialimages.SpatialImage):
        raise TypeError("Data given cannot be loaded because it is"
                        " not compatible with nibabel format:\n"
                        + short_repr(niimg))
    try:
        this_dtype = niimg.get_data_dtype()
    except AttributeError:
        # Nibabel bug
        this_dtype = niimg.get_data().dtype
    dtype = _get_target_dtype(this_dtype, dtype)

    if dtype is not None:
        niimg = new_img_like(niimg, niimg.get_data().astype(dtype),
                             get_affine(niimg))
    return niimg


def our_multi_nifti_masker_transform(self, imgs, confounds=None):
    """ Apply mask, spatial and temporal preprocessing

    Parameters
    ----------
    imgs: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html.
        Data to be preprocessed

    confounds: CSV file path or 2D matrix
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    Returns
    -------
    data: {list of numpy arrays}
        preprocessed images
    """
    self._check_fitted()
    if not hasattr(imgs, '__iter__') \
            or isinstance(imgs, _basestring):
        return self.transform_single_imgs(imgs, confounds=confounds)
    return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs)


def monkey_patch_nifti_image():
    nibabel.load = load
    joblib.memory.MemorizedFunc._get_argument_hash = our_get_argument_hash
    nilearn._utils.niimg.load_niimg = our_load_niimg
    MultiNiftiMasker.transform = our_multi_nifti_masker_transform


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