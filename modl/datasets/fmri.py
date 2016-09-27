import json
import os
from os.path import expanduser, join

from modl.datasets.hcp import fetch_hcp_rest
from sacred import Ingredient
from sklearn.model_selection import train_test_split
from nilearn import datasets

data_path_ingredient = Ingredient('data_path')
fmri_data_ingredient = Ingredient('fmri_data',
                                  ingredients=[data_path_ingredient])

# noinspection PyUnusedLocal
@data_path_ingredient.config
def config():
    path = '/storage/data'
    raw = True


# noinspection PyUnusedLocal
@fmri_data_ingredient.config
def config():
    dataset = 'adhd'
    n_subjets = 40
    test_size = 0.1

@fmri_data_ingredient.capture
def load_data(dataset, n_subjets, test_size,
              data_path,
              _seed):
    data_dir = data_path['path']
    raw = data_path['raw']
    if dataset == 'adhd':
        adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjets)
        mask = expanduser('~/data/ADHD_mask/mask_img.nii.gz')

        data = adhd_dataset.func  # list of 4D nifti files for each subject
        print('seed split', _seed)
        train_data, test_data = train_test_split(data,
                                                 test_size=test_size,
                                                 random_state=_seed)
        return train_data, test_data, mask
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
                                         n_subjects=2000)
            mask = hcp_dataset.mask
            # list of 4D nifti files for each subject
            func_filenames = hcp_dataset.func
            # Flatten it
            func_filenames = [(record for record in subject)
                              for subject in func_filenames]

            # print basic information on the dataset
            print('First functional nifti image (4D) is at: %s' %
                  hcp_dataset.func[0])  # 4D data
        return mask, func_filenames

    else:
        raise NotImplementedError