from os.path import join

from modl.datasets import get_data_dirs
from nilearn.datasets.utils import _fetch_file
from sklearn.datasets.base import Bunch

from nilearn.datasets import fetch_adhd as nilearn_fetch_adhd

import pandas as pd
import os


def fetch_adhd(n_subjects=40, data_dir=None,
               url=None, resume=True,
               modl_data_dir=None,
               mask_url=None,
               verbose=1):
    dataset = nilearn_fetch_adhd(n_subjects=n_subjects,
                                 data_dir=data_dir, url=url, resume=resume,
                                 verbose=verbose)
    root_dir = dataset.func[0]
    tail_dir = ''
    while tail_dir != 'adhd':
        root_dir, tail_dir = os.path.split(root_dir)
    root_dir = os.path.join(root_dir, tail_dir)

    modl_data_dir = get_data_dirs(modl_data_dir)[0]
    mask_data_dir = join(modl_data_dir, 'adhd')
    if mask_url is None:
        mask_url = 'http://amensch.fr/data/adhd/mask_img.nii.gz'
    _fetch_file(mask_url, mask_data_dir, resume=resume)
    mask_img = join(mask_data_dir, 'mask_img.nii.gz')
    behavioral = pd.DataFrame(dataset.phenotypic)
    behavioral.loc[:, 'Subject'] = pd.to_numeric(behavioral.loc[:, 'Subject'])
    behavioral.set_index('Subject', inplace=True)
    behavioral.index.names = ['subject']
    rest = pd.DataFrame(data=list(zip(dataset.func, dataset.confounds)),
                        columns=['filename', 'confounds'],
                        index=behavioral.index)
    return Bunch(rest=rest,
                 behavioral=behavioral, description=dataset.description,
                 mask=mask_img, root=root_dir)
