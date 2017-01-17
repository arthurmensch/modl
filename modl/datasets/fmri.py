import json
import os
from os.path import join

from nilearn import datasets
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.datasets import fetch_adhd as nilearn_fetch_adhd
from nilearn.datasets.utils import _fetch_file
from sklearn.datasets.base import Bunch
from sklearn.model_selection import train_test_split

from modl.utils.system import get_data_dirs
from .hcp import fetch_hcp_rest


def load_rest_func(dataset='adhd',
                   n_subjects=40, test_size=0.1, raw=False, random_state=None):
    data_dir = get_data_dirs()[0]
    if dataset == 'adhd':
        adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
        mask = join(data_dir, 'ADHD_mask', 'mask_img.nii.gz')
        data = adhd_dataset.func  # list of 4D nifti files for each subject
    elif dataset == 'hcp':
        if not os.path.exists(join(data_dir, 'HCP_extra')):
            raise ValueError(
                'Please download HCP_extra folder using make '
                'download-hcp_extra '
                ' first.')
        if raw:
            mask = join(data_dir, 'HCP_extra/mask_img.nii.gz')
            try:
                mapping = json.load(
                    open(join(data_dir, 'HCP_unmasked/mapping.json'), 'r'))
            except FileNotFoundError:
                raise ValueError(
                    'Please unmask the data using hcp_prepare.py first.')
            data = sorted(list(mapping.values()))
        else:
            hcp_dataset = fetch_hcp_rest(data_dir=data_dir,
                                         n_subjects=n_subjects)
            mask = hcp_dataset.mask
            # list of 4D nifti files for each subject
            data = hcp_dataset.func
            # Flatten it
            data = [(record for record in subject) for subject in data]
    else:
        raise NotImplementedError
    train_data, test_data = train_test_split(data,
                                             test_size=test_size,
                                             random_state=random_state)
    return train_data, test_data, mask


def load_atlas_init(source=None, n_components=20):
    if source == 'smith':
        if n_components == 70:
            init = fetch_atlas_smith_2009().rsn70
        elif n_components == 20:
            init = fetch_atlas_smith_2009().rsn20
        else:
            raise NotImplementedError('Unexpected argument')
    else:
        init = None
    return init