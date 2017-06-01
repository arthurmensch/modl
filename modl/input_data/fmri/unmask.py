import os

import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.compat import _basestring
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory, Parallel, delayed


class MultiRawMasker(MultiNiftiMasker):
    def __init__(self, mask_img=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):
        # Mask is provided or computed
        MultiNiftiMasker.__init__(self, mask_img=mask_img, n_jobs=n_jobs,
                                  smoothing_fwhm=smoothing_fwhm,
                                  standardize=standardize, detrend=detrend,
                                  low_pass=low_pass,
                                  high_pass=high_pass, t_r=t_r,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  mask_strategy=mask_strategy,
                                  mask_args=mask_args,
                                  memory=memory,
                                  memory_level=memory_level,
                                  verbose=verbose)

    def fit(self, imgs=None, y=None):
        self.mask_img_ = check_niimg(self.mask_img)
        self.mask_size_ = np.sum(self.mask_img_.get_data() == 1)
        return self

    def transform_single_imgs(self, imgs, confounds=None, copy=True,
                              mmap_mode=None):
        self._check_fitted()
        if isinstance(imgs, str):
            name, ext = os.path.splitext(imgs)
            if ext == '.npy':
                data = np.load(imgs, mmap_mode=mmap_mode)
            else:
                return MultiNiftiMasker.transform_single_imgs(self, imgs,
                                                              confounds=confounds,
                                                              copy=copy)
        elif isinstance(imgs, np.ndarray):
            data = imgs
        else:
            return MultiNiftiMasker.transform_single_imgs(self, imgs,
                                                          confounds=confounds,
                                                          copy=copy)
        assert (data.ndim == 2 and data.shape[1] == self.mask_size_)
        return data

    def transform_imgs(self, imgs_list, confounds=None, copy=True, n_jobs=1,
                       mmap_mode=None):
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
        self._check_fitted()
        raw = True
        # Check whether all imgs from imgs_list are numpy instance, or fallback
        # to MultiNiftiMasker (could handle hybrid imgs_list but we do not
        #  need it for the moment)
        for imgs in imgs_list:
            if isinstance(imgs, str):
                name, ext = os.path.splitext(imgs)
                if ext != '.npy':
                    raw = False
                    break
            elif not isinstance(imgs, np.ndarray):
                raw = False
                break
        if raw:
            data = Parallel(n_jobs=n_jobs)(delayed(np.load)(imgs,
                                                            mmap_mode=mmap_mode)
                                           for imgs in imgs_list)
            return data
        else:
            return MultiNiftiMasker.transform_imgs(self, imgs_list,
                                                   confounds=confounds,
                                                   copy=copy,
                                                   n_jobs=n_jobs, )

    def transform(self, imgs, confounds=None, mmap_mode=None):
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
            return self.transform_single_imgs(imgs, mmap_mode=mmap_mode)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs,
                                   mmap_mode=mmap_mode)


