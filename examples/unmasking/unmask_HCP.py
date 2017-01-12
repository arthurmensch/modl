import glob
import json
import os
from os.path import join

import numpy as np
from modl.datasets.hcp import fetch_hcp_rest
from modl.utils.system import get_data_dirs
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


def _single_mask(masker, img, dest_data_dir, source_dir):
    dest_file = img.replace(source_dir, dest_data_dir)
    dest_file = dest_file.replace('.nii.gz', '')
    dest_dir = os.path.abspath(os.path.join(dest_file, os.pardir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print('Unmasking %s' % img)
    data = masker.transform(img)
    np.save(dest_file, data)
    origin = dict(array=dest_file + '.npy', img=img)
    with open(join(dest_dir, 'origin.json'), 'w+') as f:
        json.dump(origin, f)


def _gather(dest_data_dir):
    data_dict = []
    for this_dict in glob.glob(join(dest_data_dir, '**/origin.json'),
                               recursive=True):
        with open(this_dict, 'r') as f:
            data_dict.append(json.load(f))
    mapping = {}
    for this_data in data_dict:
        mapping[this_data['img']] = this_data['array']
    with open(join(dest_data_dir, 'data.json'), 'w+') as f:
        json.dump(data_dict, f)
    with open(join(dest_data_dir, 'mapping.json'), 'w+') as f:
        json.dump(mapping, f)


def prepare_hcp_raw_data(data_dir=None,
                         dest_data_dir='HCP_unmasked',
                         n_jobs=1, smoothing_fwhm=3, n_subject=500):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'HCP')
    dest_data_dir = join(data_dir, dest_data_dir)
    dataset = fetch_hcp_rest(data_dir=data_dir, n_subjects=n_subject)
    imgs = dataset.func
    imgs = [img for subject_imgs in imgs for img in subject_imgs]
    try:
        os.mkdir(dest_data_dir)
    except OSError:
        raise ValueError('%s already exist,'
                         'please delete manually before proceeding'
                         % dest_data_dir)
    mask = join(data_dir, 'HCP_extra', 'mask_img.nii.gz')

    masker = NiftiMasker(mask_img=mask,
                         smoothing_fwhm=smoothing_fwhm,
                         standardize=True).fit()
    Parallel(n_jobs=n_jobs)(delayed(_single_mask)(masker, img,
                                                  dest_data_dir, source_dir) for
                            img in imgs)
    _gather(dest_data_dir)

if __name__ == '__main__':
    prepare_hcp_raw_data(n_subject=2, dest_data_dir='HCP_unmasked_', n_jobs=10)
