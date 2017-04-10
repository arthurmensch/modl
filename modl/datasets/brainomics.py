import glob
import os
from os.path import join

import pandas as pd

from modl.datasets import get_data_dirs
import numpy as np
import re


def fetch_brainomics(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'brainomics')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*', 'c_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    regex = re.compile('.*vs.*')
    for z_map in z_maps:
        match = re.match(regex, z_map)
        if match is None:
            dirname, contrast = os.path.split(z_map)
            contrast = contrast[6:-7]
            subject = int(dirname[-2:])
            subjects.append(subject)
            contrasts.append(contrast)
            tasks.append('localizer')
            filtered_z_maps.append(z_map)
            directions.append('level1')
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df
