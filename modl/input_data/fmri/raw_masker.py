import json
import os
from os.path import join

import numpy as np
from nilearn._utils import check_niimg
from nilearn._utils.compat import _basestring
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel, delayed, dump
import pandas as pd


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
                                  mask_strategy='background',
                                  mask_args=mask_args,
                                  verbose=verbose)

    def fit(self, imgs=None, y=None):
        self.mask_img_ = check_niimg(self.mask_img)
        self.mask_size_ = np.sum(self.mask_img_.get_data() == 1)

    def transform_single_imgs(self, imgs, confounds=None, copy=True):
        self._check_fitted()
        data = np.load(imgs)
        assert (data.ndim == 2 and data.shape[1] == self.mask_size_)
        return data

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
        self._check_fitted()
        data = Parallel(n_jobs=n_jobs)(delayed(np.load)(imgs)
                                       for imgs in imgs_list)
        return data

    def transform(self, imgs, confounds=None):
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
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs)


def create_raw_data(imgs_list,
                    root,
                    raw_dir,
                    masker_params={},
                    n_jobs=1,
                    mock=False):
    """

    Parameters
    ----------
    imgs_list: DataFrame with columns filename, confounds
    root
    raw_dir
    masker_params
    n_jobs
    mock

    Returns
    -------

    """
    if not hasattr(imgs_list, 'index'):
        imgs_list = pd.Series(imgs_list, name='filename').to_frame()
        imgs_list.assign(confounds=[None] * len(imgs_list))
    else:
        if not hasattr(imgs_list, 'columns'):
            imgs_list = imgs_list.to_frame()
            imgs_list.assign(confounds=[None] * len(imgs_list))
        else:
            assert('filename' in imgs_list.columns and 'confounds'
                   in imgs_list.columns)
    masker = MultiNiftiMasker(verbose=1, **masker_params)
    if masker.mask_img is None:
        masker.fit(imgs_list['filename'])
    else:
        masker.fit()


    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    filenames = Parallel(n_jobs=n_jobs)(delayed(_unmask_single_img)(
        masker, imgs, confounds, root, raw_dir, mock=mock)
                                        for imgs, confounds in
                                        zip(imgs_list['filename'],
                                            imgs_list['confounds']))
    imgs_list = imgs_list.assign(unmasked=filenames)
    if not mock:
        imgs_list.to_csv(os.path.join(raw_dir, 'map.csv'))
        mask_img_file = os.path.join(raw_dir, 'mask_img.nii.gz')
        masker.mask_img_.to_filename(mask_img_file)
        params = masker.get_params()
        params.pop('memory')
        params.pop('memory_level')
        params.pop('n_jobs')
        params.pop('verbose')
        params['mask_img'] = mask_img_file
        json.dump(params, open(os.path.join(raw_dir, 'masker.json'), 'w+'))
        dump(masker, os.path.join(raw_dir, 'masker.pkl'))


def _unmask_single_img(masker, imgs, confounds, root,
                       raw_dir, mock=False):
    imgs = check_niimg(imgs)
    if imgs.get_filename() is None:
        raise ValueError('Provided Nifti1Image should be linked to a file.')
    filename = imgs.get_filename()
    raw_filename = filename.replace('.nii.gz', '.npy')
    raw_filename = raw_filename.replace(root, raw_dir)
    dirname = os.path.dirname(raw_filename)
    print('Saving %s to %s' % (filename, raw_filename))
    if not mock:
        data = masker.transform(imgs, confounds=confounds)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(raw_filename, data)
    return raw_filename


def get_raw_data(imgs_list, raw_dir):
    if not os.path.exists(raw_dir):
        raise ValueError('Extraction must be done beforehand.')
    if not hasattr(imgs_list, 'index'):
        imgs_list = pd.Series(imgs_list, name='filename').to_frame()
    else:
        if not hasattr(imgs_list, 'columns'):
            imgs_list = imgs_list.to_frame()
    params = json.load(open(join(raw_dir, 'masker.json'), 'r'))
    masker = MultiRawMasker(**params)
    df = pd.read_csv(join(raw_dir, 'map.csv'))
    df.set_index(['filename', 'confounds'], inplace=True)
    df = df['unmasked']
    unmasked_imgs_list = imgs_list.join(df,
                                        on=['filename', 'confounds'],
                                        how='inner')[['unmasked', 'confounds']]
    unmasked_imgs_list.rename(columns={'unmasked': 'filename'}, inplace=True)
    return masker, unmasked_imgs_list
