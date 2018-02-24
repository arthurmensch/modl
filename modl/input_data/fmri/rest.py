import json
import os
import sys
import traceback
from itertools import repeat
from os.path import join

import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory, Parallel, delayed

from modl.input_data.fmri.unmask import MultiRawMasker


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

    if 'confounds' in imgs_list.columns:
        confounds = imgs_list['confounds']
        imgs_list.rename(columns={'confounds': 'orig_confounds'})
    else:
        confounds = repeat(None)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    filenames = Parallel(n_jobs=n_jobs)(delayed(_unmask_single_img)(
        masker, imgs, confounds, root, raw_dir, mock=mock,
        overwrite=overwrite)
                                        for imgs, confounds in
                                        zip(imgs_list['filename'],
                                            confounds))
    imgs_list = imgs_list.rename(columns={'filename': 'orig_filename'})
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