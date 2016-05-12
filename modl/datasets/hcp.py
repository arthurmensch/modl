import glob
import json
import os
from os.path import join

import numpy as np
import pandas as pd
from nilearn.datasets.utils import _get_dataset_dir
from nilearn.input_data import NiftiMasker
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


def _fetch_hcp_behavioral_data(resource_dir):
    exc_vars_file = os.path.join(resource_dir, 'excluded_scores.txt')
    vars_file = os.path.join(resource_dir, 'hcp_scores.txt')
    csv = os.path.join(resource_dir, 'unrestricted_hcp_s500.csv')

    # Smith's excluded scores
    exc_ind = np.loadtxt(exc_vars_file, dtype=np.int)
    vars_list = np.loadtxt(vars_file, dtype=bytes, delimiter='\n').astype(str)

    # unrestricted scores
    df = pd.read_csv(csv)
    vars_csv = df.columns.values

    # intersection
    vars_remaining = np.intersect1d(vars_csv, vars_list[~exc_ind]).tolist()
    df.set_index('Subject', inplace=True)

    vars_remaining.append('Age')
    df['Age'] = df['Age'].map({'26-30': 28,
                               '31-35': 33,
                               '22-25': 23.5,
                               '36+': 36})

    return df[vars_remaining]


def fetch_hcp_rest(data_dir, n_subjects=40):
    dataset_name = 'HCP'
    source_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                  verbose=0)
    extra_dir = _get_dataset_dir('HCP_extra', data_dir=data_dir,
                                 verbose=0)
    mask = join(extra_dir, 'mask_img.nii.gz')
    behavioral_df = _fetch_hcp_behavioral_data(join(extra_dir, 'behavioral'))
    func = []
    meta = []
    ids = []

    list_dir = glob.glob(join(source_dir, '*/*/MNINonLinear/Results'))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        serie_id = dirpath_split[-4]

        subject_id = int(subject_id)

        try:
            this_behavioral = behavioral_df.loc[subject_id]
        except KeyError:
            # Ignore subject without behavior data
            continue

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
               'description': "'Human connectome project",
               'behavioral': behavioral_df.loc[ids]}
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
    origin = dict(array=dest_file + '.npy', **metadata)
    with open(join(dest_dir, 'origin.json'), 'w+') as f:
        json.dump(origin, f)