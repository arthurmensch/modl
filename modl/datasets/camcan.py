import glob
import os
from os.path import join

import pandas as pd

from modl.datasets import get_data_dirs

def fetch_camcan(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'camcan', 'camcan_smt_maps')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*', '*_z_score.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        _, dirname = os.path.split(dirname)
        contrast = contrast[13:-15]
        subject = int(dirname[6:])
        if contrast in ['AudOnly', 'VidOnly', 'AudVid1200',
                        'AudVid300', 'AudVid600']:
            subjects.append(subject)
            contrasts.append(contrast)
            if contrast in ['AudOnly', 'VidOnly']:
                tasks.append('audio-video')
            else:
                tasks.append('AV-freq')
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
