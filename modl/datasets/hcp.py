import glob
import json
import os
from os.path import join

import numpy as np
from nilearn import _utils
from nilearn.datasets.utils import _get_dataset_dir
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel, Memory
from sklearn.externals.joblib import delayed


def fetch_hcp_rest(data_dir, n_subjects=40):
    dataset_name = 'HCP'
    source_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                  verbose=0)
    func = []
    meta = []
    list_dir = glob.glob(join(source_dir, '*/*/MNINonLinear/Results'))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        serie_id = dirpath_split[-4]
        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if name in ('rfMRI_REST1_RL', 'rfMRI_REST1_LR',
                        'rfMRI_REST2_RL',
                        'rfMRI_REST2_LR'):
                filename = join(dirpath, filename, filename + '.nii.gz')
                func.append(filename)
                kwargs = {'record': name, 'subject_id': subject_id,
                          'serie_id': serie_id,
                          'filename': filename}
                meta.append(kwargs)
    results = {'func': func, 'meta': meta,
               'description': "'Human connectome project"}
    return Bunch(**results)


def unmask_HCP(data_dir, dest_data_dir):
    dataset = fetch_hcp_rest(data_dir='/storage/data', n_subjects=500)

    masker = NiftiMasker(mask_img=join(data_dir, '/HCP_mask/mask_img.nii.gz'),
                         smoothing_fwhm=3, standardize=True).fit()
    Parallel(n_jobs=16)(delayed(_single_mask)(masker, this_metadata) for
                        this_metadata in dataset.meta)
    _gather(dest_data_dir)


def _gather(dest_data_dir):
    data_dict = []
    for this_dict in glob.glob(join(dest_data_dir, '**/origin.json'),
                               recursive=True):
        with open(this_dict, 'r') as f:
            data_dict.append(json.load(f))
    mapping = {}
    for this_data in data_dict:
        mapping[this_data['filename']] = this_data['array']
    with open(join(dest_data_dir, 'data.json'), 'w+') as f:
        json.dump(data_dict, f)
    with open(join(dest_data_dir, 'mapping.json'), 'w+') as f:
        json.dump(mapping, f)


def _single_mask(masker, metadata, data_dir, dest_data_dir):
    img = metadata['filename']
    dest_file = img.replace(join(data_dir, 'HCP'), dest_data_dir)
    dest_file = dest_file.replace('.nii.gz', '')
    dest_dir = os.path.abspath(os.path.join(dest_file, os.pardir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    data = masker.transform(img)
    np.save(dest_file, data)
    origin = dict(**metadata, array=dest_file + '.npy', )
    with open(join(dest_dir, 'origin.json'), 'w+') as f:
        json.dump(origin, f)


class DummyMasker(MultiNiftiMasker):
    def __init__(self, data_dir=None, mask_img=None, smoothing_fwhm=None,
                 standardize=False,
                 detrend=False, low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0, n_jobs=1,
                 verbose=0):
        super().__init__(mask_img, smoothing_fwhm, standardize, detrend,
                         low_pass, high_pass, t_r, target_affine, target_shape,
                         mask_strategy, mask_args, memory, memory_level,
                         n_jobs, verbose)
        self.data_dir = data_dir

    def fit(self, imgs=None, y=None):
        self.mask_img_ = _utils.check_niimg_3d('/storage/data/HCP_mask/'
                                               'mask_img.nii.gz')
        with open(join(self.data_dir, 'mapping.json'), 'r') as f:
            self.mapping_ = json.load(f)

    def transform_single_imgs(self, imgs, confounds=None, copy=True):
        return np.load(self.mapping_[imgs])
