import glob
import os
from os.path import join

import pandas as pd

from modl.datasets import get_data_dirs


def fetch_archi(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'archi', 'unsmoothed')
    if not os.path.exists(source_dir):
        raise ValueError('Please ensure that archi can be found under $MODL_DATA'
                         'repository.')
    z_maps = glob.glob(join(source_dir, '*', '*_z_map.nii'))
    subjects = []
    contrasts = []
    for z_map in z_maps:
        dirname, filename = os.path.split(z_map)
        _, subject = os.path.split(dirname)
        subject = int(subject[-3:])
        contrast = filename.replace('z_map.nii', '')
        subjects.append(subject)
        contrasts.append(contrast)
    df = pd.DataFrame(data=[subjects, contrasts, z_maps], name=['subject',
                                                                'contrast',
                                                                'filename'])
    df.set_index(['subject', 'contrast'], inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df

