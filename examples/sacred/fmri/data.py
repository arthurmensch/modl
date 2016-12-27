import json
import os
from os.path import expanduser, join

from modl.datasets import get_data_dirs
from modl.datasets.hcp import fetch_hcp_rest
from sacred import Ingredient
from sklearn.model_selection import train_test_split
from nilearn import datasets
from nilearn.datasets import fetch_atlas_smith_2009

data_ing = Ingredient('data')
init_ing = Ingredient('init')

@init_ing.config
def config():
    n_components = 20
    source = None

@data_ing.config
def config():
    dataset = 'adhd'
    n_subjects = 40
    test_size = 0.1
    raw = True

@data_ing.capture
def load_data(dataset, n_subjects, test_size,
              raw,
              _seed):
    data_dir = get_data_dirs()[0]
    if dataset == 'adhd':
        adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
        mask = join(data_dir, 'ADHD_mask', 'mask_img.nii.gz')

        data = adhd_dataset.func  # list of 4D nifti files for each subject
        print('seed split', _seed)
        train_data, test_data = train_test_split(data,
                                                 test_size=test_size,
                                                 random_state=_seed)
        return train_data, test_data, mask, False
    elif dataset == 'hcp':
        if not os.path.exists(join(data_dir, 'HCP_extra')):
            raise ValueError(
                'Please download HCP_extra folder using make download-hcp_extra'
                ' first.')
        if raw:
            mask = join(data_dir, 'HCP_extra/mask_img.nii.gz')
            try:
                mapping = json.load(
                    open(join(data_dir, 'HCP_unmasked/mapping.json'), 'r'))
            except FileNotFoundError:
                raise ValueError(
                    'Please unmask the data using hcp_prepare.py first.')
            func_filenames = sorted(list(mapping.values()))
        else:
            hcp_dataset = fetch_hcp_rest(data_dir=data_dir,
                                         n_subjects=n_subjects)
            mask = hcp_dataset.mask
            # list of 4D nifti files for each subject
            func_filenames = hcp_dataset.func
            # Flatten it
            func_filenames = [(record for record in subject)
                              for subject in func_filenames]

            # print basic information on the dataset
            print('First functional nifti image (4D) is at: %s' %
                  hcp_dataset.func[0])  # 4D data
        train_data, test_data = train_test_split(func_filenames,
                                                 test_size=test_size,
                                                 random_state=_seed)
        return train_data, test_data, mask, raw

    else:
        raise NotImplementedError


@init_ing.capture
def load_init(n_components, source):
    if source == 'smith':
        if n_components == 70:
            init = fetch_atlas_smith_2009().rsn70
        elif n_components == 20:
            init = fetch_atlas_smith_2009().rsn20
        else:
            raise NotImplementedError('Unexpected argument')
    else:
        init = None
    return n_components, init
