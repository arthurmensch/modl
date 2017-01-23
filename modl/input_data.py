import itertools

from nilearn import _utils
from nilearn._utils.class_inspect import get_params
from nilearn._utils.compat import izip, _basestring
from nilearn._utils.niimg_conversions import _iter_check_niimg
from nilearn.input_data import MultiNiftiMasker
from nilearn.input_data.nifti_masker import filter_and_mask
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
import numpy as np
import os


class RawMasker(MultiNiftiMasker):
    def __init__(self, mask_img=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):
        # Mask is provided or computed
        self.mask_img = mask_img

        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs

        self.verbose = verbose

        self._shelving = False

    def fit(self, imgs=None, y=None):
        if self.mask_img is None:
            raise ValueError("Raw Masker should be provided with a"
                             "mask image.")
        else:
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)
        self.n_voxels_ = np.sum(self.mask_img_.get_data())
        return self

    def transform_imgs(self, imgs_list, confounds=None, copy=True, n_jobs=1):
        """Prepare multi subject data in parallel

        Parameters
        ----------

        imgs_list: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html.
            List of imgs file to prepare. One item per subject.

        confounds: list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        copy: boolean, optional
            If True, guarantees that output array has no memory in common with
            input array.

        n_jobs: integer, optional
            The number of cpus to use to do the computation. -1 means
            'all cpus'.

        Returns
        -------
        region_signals: list of 2D numpy.ndarray
            List of signal for each element per subject.
            shape: list of (number of scans, number of elements)
        """

        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

        # Ignore the mask-computing params: they are not useful and will
        # just invalidate the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        func = self._cache(load_and_check,
                           shelve=self._shelving)
        data = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(func)(imgs, self.n_voxels_)
            for imgs in imgs_list)
        return data


def load_and_check(imgs, expected_n_voxels):
    if isinstance(imgs, np.ndarray):
        return imgs
    elif isinstance(imgs, str):
        imgs = np.load(imgs)
    n_samples, n_voxels = imgs.shape
    if n_voxels != expected_n_voxels:
        raise ValueError('imgs has wrong dimension')
    return imgs


def unmask_dataset(masker, imgs_list, base_dir, dest_dir,
                   confounds=None, copy=True, n_jobs=1):
    """Prepare multi subject data in parallel

    Parameters
    ----------

    imgs_list: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html.
        List of imgs file to prepare. One item per subject.

    confounds: list of confounds, optional
        List of confounds (2D arrays or filenames pointing to CSV
        files). Must be of same length than imgs_list.

    copy: boolean, optional
        If True, guarantees that output array has no memory in common with
        input array.

    n_jobs: integer, optional
        The number of cpus to use to do the computation. -1 means
        'all cpus'.

    Returns
    -------
    region_signals: list of 2D numpy.ndarray
        List of signal for each element per subject.
        shape: list of (number of scans, number of elements)
    """

    if not hasattr(masker, 'mask_img_'):
        raise ValueError('It seems that %s has not been fitted. '
                         'You must call fit() before calling transform().'
                         % masker.__class__.__name__)
    target_fov = None
    if masker.target_affine is None:
        # Force resampling on first image
        target_fov = 'first'

    niimg_iter = _iter_check_niimg(imgs_list, ensure_ndim=None,
                                   atleast_4d=False,
                                   target_fov=target_fov,
                                   memory=masker.memory,
                                   memory_level=masker.memory_level,
                                   verbose=masker.verbose)

    if confounds is None:
        confounds = itertools.repeat(None, len(imgs_list))

    # Ignore the mask-computing params: they are not useful and will
    # just invalidate the cache for no good reason
    # target_shape and target_affine are conveyed implicitly in mask_img
    params = get_params(masker.__class__, masker,
                        ignore=['mask_img', 'mask_args', 'mask_strategy',
                                'copy'])

    func = masker._cache(filter_mask_and_save,
                         ignore=['verbose', 'memory', 'memory_level',
                                 'copy'],
                         shelve=masker._shelving)
    dest_files = [imgs.replace(base_dir, dest_dir).replace('.nii.gz', '.npy')
                  for imgs in imgs_list]

    data = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(func)(imgs, masker.mask_img_, params,
                      dest_file,
                      memory_level=masker.memory_level,
                      memory=masker.memory,
                      verbose=masker.verbose,
                      confounds=cfs,
                      copy=copy,
                      )
        for imgs, cfs, dest_file in izip(niimg_iter, confounds, dest_files))
    mapping = {}
    for imgs, dest_file in data:
        mapping[imgs] = dest_file
    return mapping


def filter_mask_and_save(imgs, mask_img_, parameters,
                         dest_file,
                         memory_level=0, memory=Memory(cachedir=None),
                         verbose=0,
                         confounds=None,
                         copy=True,
                         ):
    data = filter_and_mask(imgs, mask_img_, parameters,
                           memory_level=memory_level, memory=memory,
                           verbose=verbose,
                           confounds=confounds,
                           copy=copy)
    parentdir = os.path.dirname(dest_file)
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)
    np.save(dest_file, data)
    return imgs, dest_file
