import glob
import json
import os
from os.path import join

import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

from modl.datasets import get_data_home

data_home = get_data_home()


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


def _single_mask(masker, img, dest_data_dir, data_dir):
    dest_file = img.replace('HCP', 'HCP_unmasked')
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


def fetch_hcp_rest(data_dir=data_home, n_subjects=40):
    source_dir = join(data_dir, 'HCP')
    extra_dir = join(data_dir, 'HCP_extra')
    mask = join(extra_dir, 'mask_img.nii.gz')
    func = []
    meta = []
    ids = []

    list_dir = glob.glob(join(source_dir, '*/*/MNINonLinear/Results'))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        serie_id = dirpath_split[-4]

        subject_id = int(subject_id)

        ids.append(subject_id)

        kwargs = {'subject_id': subject_id,
                  'serie_id': serie_id}

        meta.append(kwargs)

        subject_func = []

        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if name in ('rfMRI_REST1_RL', 'rfMRI_REST1_LR',
                        'rfMRI_REST2_RL',
                        'rfMRI_REST2_LR'):
                filename = join(dirpath, filename, filename + '.nii.gz')
                subject_func.append(filename)
        func.append(subject_func)

    results = {'func': func, 'meta': meta,
               'mask': mask,
               'description': "'Human connectome project"}
    return Bunch(**results)


def prepare_hcp_raw_data(n_jobs=1, data_dir=data_home):
    dataset = fetch_hcp_rest(data_dir=data_dir, n_subjects=500)

    imgs = [img for subject_imgs in dataset.func for img in subject_imgs]

    dest_data_dir = join(data_dir, 'HCP_unmasked')
    mask = dataset.mask

    masker = NiftiMasker(mask_img=mask,
                         smoothing_fwhm=3, standardize=True).fit()
    Parallel(n_jobs=n_jobs)(delayed(_single_mask)(masker, img,
                                              dest_data_dir, data_dir) for
                        img in imgs)
    _gather(dest_data_dir)


def get_hcp_data(data_dir=data_home, raw=False):
    if not os.path.exists(join(data_dir, 'HCP_extra')):
        raise IOError(
            'Please download HCP_extra folder using make hcp'
            ' first.')
    if raw:
        mask = join(data_dir, 'HCP_extra/mask_img.nii.gz')
        try:
            mapping = json.load(
                open(join(data_dir, 'HCP_unmasked/mapping.json'), 'r'))
        except FileNotFoundError:
            raise IOError(
                'Please unmask the data using hcp_prepare.py first.')
        func_filenames = sorted(list(mapping.values()))
    else:
        hcp_dataset = fetch_hcp_rest(data_dir=data_dir,
                                     n_subjects=2000)
        mask = hcp_dataset.mask
        # list of 4D nifti files for each subject
        func_filenames = hcp_dataset.func
        # Flatten it
        func_filenames = [record for subject in func_filenames for record in subject]

        # print basic information on the dataset
        print('First functional nifti image (4D) is at: %s' %
              hcp_dataset.func[0])  # 4D data
    return mask, func_filenames
