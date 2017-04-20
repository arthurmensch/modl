import glob
import os
from os.path import join

import pandas as pd

from modl.datasets import get_data_dirs
import numpy as np
import re


def fetch_human_voice(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'human_voice', 'ds000158_R1.0.1', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that human_voice can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*/*/*', 'z_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        contrast = contrast[2:-7]
        dirname, _ = os.path.split(dirname)
        dirname, task = os.path.split(dirname)
        dirname, subject = os.path.split(dirname)
        subject = int(subject[-3:])
        subjects.append(subject)
        contrasts.append(contrast)
        tasks.append(task)
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