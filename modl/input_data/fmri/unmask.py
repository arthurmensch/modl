import json
import os
import traceback
from os.path import join

import numpy as np
import sys
from nilearn._utils import check_niimg
from nilearn._utils.compat import _basestring
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory, Parallel, delayed, dump, load
import pandas as pd
from sklearn.utils import gen_batches


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
            name, ext = os.path.split(imgs)
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
                name, ext = os.path.split(imgs)
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
                                                   n_jobs=n_jobs,)

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


def create_raw_rest_data(imgs_list,
                         root,
                         raw_dir,
                         masker_params=None,
                         n_jobs=1,
                         mock=False,
                         memory=Memory(cachedir=None),
                         overwrite=False):
    """

    Parameters
    ----------
    memory
    imgs_list: DataFrame with columns filename, confounds
    root
    raw_dir
    masker_params
    n_jobs
    mock

    Returns
    -------

    """
    if masker_params is None:
        masker_params = {}
    masker = MultiNiftiMasker(verbose=1, memory=memory,
                              memory_level=1,
                              **masker_params)
    if masker.mask_img is None:
        masker.fit(imgs_list['filename'])
    else:
        masker.fit()

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    filenames = Parallel(n_jobs=n_jobs)(delayed(_unmask_single_img)(
        masker, imgs, confounds, root, raw_dir, mock=mock,
        overwrite=overwrite)
                                        for imgs, confounds in
                                        zip(imgs_list['filename'],
                                            imgs_list['confounds']))
    imgs_list = imgs_list.rename(columns={'filename': 'orig_filename',
                              'confounds': 'orig_confounds'})
    imgs_list = imgs_list.assign(filename=filenames)
    imgs_list = imgs_list.assign(confounds=None)
    if not mock:
        imgs_list.to_csv(os.path.join(raw_dir, 'data.csv'), mode='w+')
        mask_img_file = os.path.join(raw_dir, 'mask_img.nii.gz')
        masker.mask_img_.to_filename(mask_img_file)
        params = masker.get_params()
        params.pop('memory')
        params.pop('memory_level')
        params.pop('n_jobs')
        params.pop('verbose')
        params['mask_img'] = mask_img_file
        json.dump(params, open(os.path.join(raw_dir, 'masker.json'), 'w+'))


def _unmask_single_img(masker, imgs, confounds, root,
                       raw_dir, mock=False, overwrite=False):
    imgs = check_niimg(imgs)
    if imgs.get_filename() is None:
        raise ValueError('Provided Nifti1Image should be linked to a file.')
    filename = imgs.get_filename()
    raw_filename = filename.replace('.nii.gz', '.npy')
    raw_filename = raw_filename.replace(root, raw_dir)
    dirname = os.path.dirname(raw_filename)
    print('Saving %s to %s' % (filename, raw_filename))
    if not mock:
        if overwrite or not os.path.exists(raw_filename):
            try:
                data = masker.transform(imgs, confounds=confounds)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                np.save(raw_filename, data)
            except EOFError:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                msg = '\n'.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback))
                raw_filename += '-error'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                with open(raw_filename, 'w+') as f:
                    f.write(msg)
        else:
            print('File already exists: skipping.')
    return raw_filename


def get_raw_rest_data(raw_dir):
    if not os.path.exists(raw_dir):
        raise ValueError('Unmask directory %s does not exist.'
                         'Unmasking must be done beforehand.' % raw_dir)
    params = json.load(open(join(raw_dir, 'masker.json'), 'r'))
    masker = MultiRawMasker(**params)
    unmasked_imgs_list = pd.read_csv(join(raw_dir, 'data.csv'))
    return masker, unmasked_imgs_list


def create_raw_contrast_data(imgs, mask, raw_dir,
                             memory=Memory(cachedir=None),
                             n_jobs=1, batch_size=100):
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # Selection of contrasts
    masker = MultiNiftiMasker(smoothing_fwhm=0,
                              mask_img=mask,
                              memory=memory,
                              memory_level=1,
                              n_jobs=n_jobs).fit()
    mask_img_file = os.path.join(raw_dir, 'mask_img.nii.gz')
    masker.mask_img_.to_filename(mask_img_file)

    batches = gen_batches(len(imgs), batch_size)

    dump(imgs.index, join(raw_dir, 'index.pkl'))

    data = np.lib.format.open_memmap(join(raw_dir,
                                          'z_maps.npy'),
                                     mode='w+',
                                     shape=(len(imgs),
                                            masker.mask_img_.get_data().sum()),
                                     dtype=np.float32)
    for i, batch in enumerate(batches):
        print('Batch %i' % i)
        this_data = masker.transform(imgs['z_map'].values[batch],
                                     )
        data[batch] = this_data


def get_raw_contrast_data(raw_dir):
    mask_img = os.path.join(raw_dir, 'mask_img.nii.gz')
    masker = MultiRawMasker(smoothing_fwhm=0, mask_img=mask_img)
    masker.fit()
    data = np.load(join(raw_dir, 'z_maps.npy'),
                   mmap_mode='r')
    index = load(join(raw_dir, 'index.pkl'))
    imgs = pd.DataFrame(data=data, index=index)
    return masker, imgs
