import glob
import os
from os.path import join

from sklearn.datasets.base import Bunch

from . import get_data_dirs
from nilearn.datasets.utils import _fetch_file
import pandas as pd

import numpy as np


contrasts_description = {'2BK': {'Cognitive Task': 'Two-Back Memory',
                                 'Instruction to participants': 'Indicate whether current stimulus is the same as two items earlier',
                                 'Stimulus material': 'Task Pictures'},
                         'BODY-AVG': {'Cognitive Task': 'View Bodies',
                                      'Instruction to participants': 'Passive watching',
                                      'Stimulus material': 'Pictures'},
                         'FACE-AVG': {'Cognitive Task': 'View Faces',
                                      'Instruction to participants': 'Passive watching',
                                      'Stimulus material': 'Pictures'},
                         'FACES': {'Cognitive Task': 'Shapes',
                                   'Instruction to participants': 'Decide which of two shapes matches another shape geometry-wise',
                                   'Stimulus material': 'Shape pictures'},
                         'FACES-SHAPES': {'Cognitive Task': 'Faces',
                                          'Instruction to participants': 'Decide which of two faces matches another face emotion-wise',
                                          'Stimulus material': 'Face pictures'},
                         'LF': {'Cognitive Task': 'Food movement',
                                'Instruction to participants': 'Squeezing of the left or right toe',
                                'Stimulus material': 'Visual cues'},
                         'LH': {'Cognitive Task': 'Hand movement',
                                'Instruction to participants': 'Tapping of the left or right finger',
                                'Stimulus material': 'Visual cues'},
                         'MATCH': {'Cognitive Task': 'Matching',
                                   'Instruction to participants': 'Decide whether two objects match in shape or texture',
                                   'Stimulus material': 'Pictures'},
                         'MATH': {'Cognitive Task': 'Mathematics',
                                  'Instruction to participants': 'Complete addition and subtraction problems',
                                  'Stimulus material': 'Spoken numbers'},
                         'PLACE-AVG': {'Cognitive Task': 'View Places',
                                       'Instruction to participants': 'Passive watching',
                                       'Stimulus material': 'Pictures'},
                         'PUNISH': {'Cognitive Task': 'Reward',
                                    'Instruction to participants': 'Guess the number of mystery card for gain/loss of money',
                                    'Stimulus material': 'Card game'},
                         'RANDOM': {'Cognitive Task': 'Random',
                                    'Instruction to participants': 'Decide whether the objects act randomly or intentionally',
                                    'Stimulus material': 'Videos with objects'},
                         'REL': {'Cognitive Task': 'Relations',
                                 'Instruction to participants': 'Decide whether object pairs differ both along either shapeor texture',
                                 'Stimulus material': 'Pictures'},
                         'REWARD': {'Cognitive Task': 'Punish',
                                    'Instruction to participants': 'Guess the number of mystery card for gain/loss of money',
                                    'Stimulus material': 'Card game'},
                         'STORY': {'Cognitive Task': 'Language',
                                   'Instruction to participants': 'Choose answer about the topic of the story',
                                   'Stimulus material': 'Auditory stories'},
                         'T': {'Cognitive Task': 'Tongue movement',
                               'Instruction to participants': 'Move tongue',
                               'Stimulus material': 'Visual cues'},
                         'TOM': {'Cognitive Task': 'Theory of mind',
                                 'Instruction to participants': 'Decide whether the objects act randomly or intentionally',
                                 'Stimulus material': 'Videos with objects'},
                         'TOOL-AVG': {'Cognitive Task': 'View Tools',
                                      'Instruction to participants': 'Passive watching',
                                      'Stimulus material': 'Pictures'}}



def fetch_hcp_behavioral(data_dir=None, release='HCP900'):
    if release != 'HCP900':
        raise ValueError("Unsupported release %s, should be 'HCP900'"
                         % release)
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'HCP900')
    df_unrestricted = pd.read_csv(join(source_dir, 'behavioral',
                                       'hcp_unrestricted_data.csv'))
    df_restricted = pd.read_csv(join(source_dir, 'behavioral',
                                                 'hcp_restricted_data.csv'))
    df_unrestricted.set_index('Subject', inplace=True)
    df_restricted.set_index('Subject', inplace=True)
    df = df_unrestricted.join(df_restricted, how='outer')
    df.sort_index(ascending=True, inplace=True)
    return df


def fetch_hcp_task(data_dir=None, release='HCP900',
                   output='nistats',
                   n_subjects=788,
                   level=2):
    """Nilearn like fetcher"""
    data_dir = get_data_dirs(data_dir)[0]

    tasks = [["WM", 1, "2BK_BODY"],
             ["WM", 2, "2BK_FACE"],
             ["WM", 3, "2BK_PLACE"],
             ["WM", 4, "2BK_TOOL"],
             ["WM", 5, "0BK_BODY"],
             ["WM", 6, "0BK_FACE"],
             ["WM", 7, "0BK_PLACE"],
             ["WM", 8, "0BK_TOOL"],
             ["WM", 9, "2BK"],
             ["WM", 10, "0BK"],
             ["WM", 11, "2BK-0BK"],
             ["WM", 12, "neg_2BK"],
             ["WM", 13, "neg_0BK"],
             ["WM", 14, "0BK-2BK"],
             ["WM", 15, "BODY"],
             ["WM", 16, "FACE"],
             ["WM", 17, "PLACE"],
             ["WM", 18, "TOOL"],
             ["WM", 19, "BODY-AVG"],
             ["WM", 20, "FACE-AVG"],
             ["WM", 21, "PLACE-AVG"],
             ["WM", 22, "TOOL-AVG"],
             ["WM", 23, "neg_BODY"],
             ["WM", 24, "neg_FACE"],
             ["WM", 25, "neg_PLACE"],
             ["WM", 26, "neg_TOOL"],
             ["WM", 27, "AVG-BODY"],
             ["WM", 28, "AVG-FACE"],
             ["WM", 29, "AVG-PLACE"],
             ["WM", 30, "AVG-TOOL"],
             ["GAMBLING", 1, "PUNISH"],
             ["GAMBLING", 2, "REWARD"],
             ["GAMBLING", 3, "PUNISH-REWARD"],
             ["GAMBLING", 4, "neg_PUNISH"],
             ["GAMBLING", 5, "neg_REWARD"],
             ["GAMBLING", 6, "REWARD-PUNISH"],
             ["MOTOR", 1, "CUE"],
             ["MOTOR", 2, "LF"],
             ["MOTOR", 3, "LH"],
             ["MOTOR", 4, "RF"],
             ["MOTOR", 5, "RH"],
             ["MOTOR", 6, "T"],
             ["MOTOR", 7, "AVG"],
             ["MOTOR", 8, "CUE-AVG"],
             ["MOTOR", 9, "LF-AVG"],
             ["MOTOR", 10, "LH-AVG"],
             ["MOTOR", 11, "RF-AVG"],
             ["MOTOR", 12, "RH-AVG"],
             ["MOTOR", 13, "T-AVG"],
             ["MOTOR", 14, "neg_CUE"],
             ["MOTOR", 15, "neg_LF"],
             ["MOTOR", 16, "neg_LH"],
             ["MOTOR", 17, "neg_RF"],
             ["MOTOR", 18, "neg_RH"],
             ["MOTOR", 19, "neg_T"],
             ["MOTOR", 20, "neg_AVG"],
             ["MOTOR", 21, "AVG-CUE"],
             ["MOTOR", 22, "AVG-LF"],
             ["MOTOR", 23, "AVG-LH"],
             ["MOTOR", 24, "AVG-RF"],
             ["MOTOR", 25, "AVG-RH"],
             ["MOTOR", 26, "AVG-T"],
             ["LANGUAGE", 1, "MATH"],
             ["LANGUAGE", 2, "STORY"],
             ["LANGUAGE", 3, "MATH-STORY"],
             ["LANGUAGE", 4, "STORY-MATH"],
             ["LANGUAGE", 5, "neg_MATH"],
             ["LANGUAGE", 6, "neg_STORY"],
             ["SOCIAL", 1, "RANDOM"],
             ["SOCIAL", 2, "TOM"],
             ["SOCIAL", 3, "RANDOM-TOM"],
             ["SOCIAL", 4, "neg_RANDOM"],
             ["SOCIAL", 5, "neg_TOM"],
             ["SOCIAL", 6, "TOM-RANDOM"],
             ["RELATIONAL", 1, "MATCH"],
             ["RELATIONAL", 2, "REL"],
             ["RELATIONAL", 3, "MATCH-REL"],
             ["RELATIONAL", 4, "REL-MATCH"],
             ["RELATIONAL", 5, "neg_MATCH"],
             ["RELATIONAL", 6, "neg_REL"],
             ["EMOTION", 1, "FACES"],
             ["EMOTION", 2, "SHAPES"],
             ["EMOTION", 3, "FACES-SHAPES"],
             ["EMOTION", 4, "neg_FACES"],
             ["EMOTION", 5, "neg_SHAPES"],
             ["EMOTION", 6, "SHAPES-FACES"]]

    res = []
    if release == 'HCP500' or release == 'HCP900' and output == 'fsl':
        source_dir = join(data_dir, release)
        if not os.path.exists(source_dir):
            raise ValueError('Please make sure that a directory %s can '
                             'be found '
                             'in the $MODL_DATA directory' % release)
        if release == 'HCP500':
            list_dir = sorted(glob.glob(join(source_dir,
                                             '*/*/MNINonLinear/Results')))
        else:
            list_dir = sorted(glob.glob(join(source_dir,
                                             '*/MNINonLinear/Results')))
        for dirpath in list_dir[:n_subjects]:
            dirpath_split = dirpath.split(os.sep)
            subject_id = dirpath_split[-3]
            subject_id = int(subject_id)

            for i, task in enumerate(tasks):
                task_name = task[0]
                contrast_idx = task[1]
                this_contrast = task[2]
                if level == 2:
                    filename = join(dirpath, "tfMRI_%s/tfMRI_%s_hp200_s4_"
                                              "level2vol.feat/cope%i.feat/"
                                              "stats/zstat1.nii.gz" % (
                                         task_name, task_name,
                                         contrast_idx))
                    if os.path.exists(filename):
                        res.append({'filename': filename,
                                    'subject': subject_id,
                                    'task': task_name,
                                    'contrast': this_contrast,
                                    'direction': 'level2'
                                    })
                else:
                    raise ValueError('Can only output level 2 images'
                                     'for release %s with output %s'
                                     % (release, output))
    elif release == 'HCP900':
        source_dir = join(data_dir, 'HCP900', 'glm')
        if not os.path.exists(source_dir):
            raise ValueError('Please make sure that a directory HCP900 can '
                             'be found '
                             'in the $MODL_DATA directory')
        if level == 2:
            directions = ['level2']
        elif level == 1:
            directions = ['LR', 'RL']
        else:
            raise ValueError('Level should be 1 or 2, got %s' % level)
        subject_ids = os.listdir(source_dir)
        for subject_id in subject_ids[:n_subjects]:
            tasks = os.listdir(join(source_dir, subject_id))
            for task in tasks:
                for direction in directions:
                    z_dir = join(source_dir, subject_id, task, direction,
                                 'z_maps')
                    if os.path.exists(z_dir):
                        z_maps = os.listdir(z_dir)
                        for z_map in z_maps:
                            filename = join(z_dir, z_map)
                            this_contrast = z_map[2:-7]
                            if os.path.exists(filename):
                                res.append({'filename': filename,
                                            'subject': int(subject_id),
                                            'task': task,
                                            'contrast': this_contrast,
                                            'direction': direction
                                            })
    else:
        raise ValueError('Release should be `HCP900` or `HCP500`,'
                         ' got %s' % release)
    z_maps = pd.DataFrame(res)
    z_maps.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    z_maps.sort_index(ascending=True, inplace=True)
    return z_maps


def fetch_hcp_mask(data_dir=None, url=None, resume=True):
    modl_data_dir = get_data_dirs(data_dir)[0]
    data_dir = join(modl_data_dir, 'HCP_own')
    if url is None:
        url = 'http://amensch.fr/data/HCP_own/mask_img.nii.gz'
    _fetch_file(url, data_dir, resume=resume)
    return join(data_dir, 'mask_img.nii.gz')


def fetch_hcp_rest(data_dir=None, release='HCP900', n_subjects=500):
    """Nilearn like fetcher"""
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, release)
    if not os.path.exists(source_dir):
        raise ValueError('Please make sure that a directory %s can be found '
                         'in the $MODL_DATA directory' % release)
    res = []
    if release == 'HCP900':
        list_dir = sorted(
            glob.glob(join(source_dir, '*/MNINonLinear/Results')))
    else:
        list_dir = sorted(
            glob.glob(join(source_dir, '*/*/MNINonLinear/Results')))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        subject_id = int(subject_id)

        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if name in ('rfMRI_REST1_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_RL',
                        'rfMRI_REST2_LR'):
                filename = join(dirpath, filename, filename + '.nii.gz')
                if os.path.exists(filename):
                    res.append(
                        {'filename': filename,
                         'subject': int(subject_id),
                         'direction': name[-2:],
                         'series': int(name[-4])
                         })

    rest = pd.DataFrame(res)
    rest.set_index(['subject', 'series', 'direction'], inplace=True)
    rest.sort_index(ascending=True, inplace=True)
    return rest


def fetch_hcp(data_dir=None, n_subjects=100):
    rest = fetch_hcp_rest(data_dir, release='HCP900', n_subjects=n_subjects)
    task = fetch_hcp_task(data_dir, release='HCP900', output='nistats',
                          n_subjects=n_subjects)
    mask = fetch_hcp_mask(data_dir)
    subjects = np.unique(np.concatenate([rest.index.get_level_values(0).values,
                                            task.index.
                                        get_level_values(0).values]))
    behavioral = fetch_hcp_behavioral(data_dir, release='HCP900')
    behavioral = behavioral.loc[subjects, :]
    return Bunch(rest=rest, task=task, behavioral=behavioral, mask=mask)
