from os.path import expanduser

from sacred import Ingredient
from sklearn.model_selection import train_test_split
from nilearn import datasets

fmri_data_ingredient = Ingredient('dataset')

# noinspection PyUnusedLocal
@fmri_data_ingredient.config
def config():
    dataset = 'adhd'
    n_subjets = 40
    test_size = 0.1

@fmri_data_ingredient.capture
def load_data(dataset, n_subjets, test_size, _seed):
    if dataset == 'adhd':
        adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjets)
        mask = expanduser('~/data/ADHD_mask/mask_img.nii.gz')

        data = adhd_dataset.func  # list of 4D nifti files for each subject
        train_data, test_data = train_test_split(data,
                                                 test_size=test_size,
                                                 random_state=_seed)
        return train_data, test_data, mask
    else:
        raise NotImplementedError