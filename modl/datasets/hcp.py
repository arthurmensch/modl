import glob
import json
import os
from os.path import join

import numpy as np
from nilearn.datasets.utils import _fetch_file

from modl.utils.system import get_data_dirs
from sklearn.datasets.base import Bunch

# Danilo paper
task_ids = [30, 31,
            80, 82,
            68, 69,
            62, 63,
            41, 37, 38,  # should be 37 + 40, 38 + 41
            74, 75,
            14, 15, 16, 17,
            8]

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


def fetch_hcp_behavioral(data_dir=None, n_subjects=500):
    import pandas as pd
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'HCP')
    df = pd.read_csv(join(source_dir,
                          'restricted_scores.csv'))
    list_dir = sorted(glob.glob(join(source_dir, '*/*/MNINonLinear/Results')))
    subjects = []
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = int(dirpath_split[-3])
        subjects.append(subject_id)
    indices = [subject in subjects for subject in df['Subject']]
    df = df.loc[indices]
    return df


def fetch_hcp_task(data_dir=None, n_subjects=500, mask_url=None, resume=True):
    """Nilearn like fetcher"""
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

    mask = fetch_hcp_mask(data_dir, url=mask_url, resume=resume)
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'HCP')
    if not os.path.exists(source_dir):
        raise ValueError('Please make sure that a directory HCP can be found '
                         'in the $MODL_DATA directory')
    func = []
    contrast = []

    list_dir = sorted(glob.glob(join(source_dir, '*/*/MNINonLinear/Results')))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        subject_id = int(subject_id)

        for i, task in enumerate(tasks):
            task_name = task[0]
            contrast_idx = task[1]
            this_contrast = task[2]
            this_func = join(dirpath, "tfMRI_%s/tfMRI_%s_hp200_s4_"
                                      "level2vol.feat/cope%i.feat/"
                                      "stats/zstat1.nii.gz" % (
                                 task_name, task_name,
                                 contrast_idx))
            if os.path.exists(this_func):
                contrast.append({'subject_id': subject_id,
                                 'task_name': task_name,
                                 'contrast': this_contrast
                                 })
                func.append(this_func)

    results = {'func': func, 'contrast': contrast,
               'mask': mask, 'description': "The Human connectome project"}
    return Bunch(**results)


def fetch_hcp_mask(data_dir=None, url=None, resume=True):
    modl_data_dir = get_data_dirs(data_dir)[0]
    data_dir = join(modl_data_dir, 'HCP_own')
    if url is None:
        url = 'http://amensch.fr/data/HCP_own/mask_img.nii.gz'
    _fetch_file(url, data_dir, resume=resume)
    return join(data_dir, 'mask_img.nii.gz')


def fetch_hcp_rest(data_dir=None, n_subjects=500, mask_url=None, resume=True):
    """Nilearn like fetcher"""
    mask = fetch_hcp_mask(data_dir, url=mask_url, resume=resume)
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'HCP')
    if not os.path.exists(source_dir):
        raise ValueError('Please make sure that a directory HCP can be found '
                         'in the $MODL_DATA directory')
    func = []
    phenotypic = []
    ids = []

    list_dir = sorted(glob.glob(join(source_dir, '*/*/MNINonLinear/Results')))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        serie_id = dirpath_split[-4]

        subject_id = int(subject_id)

        ids.append(subject_id)

        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if name in ('rfMRI_REST1_RL', 'rfMRI_REST1_LR',
                        'rfMRI_REST2_RL',
                        'rfMRI_REST2_LR'):
                filename = join(dirpath, filename, filename + '.nii.gz')
                func.append(filename)
                phenotypic.append(
                    {'subject_id': subject_id, 'serie_id': serie_id,
                     'record_direction': name[-2:],
                     'record_serie': name[-4]})

    results = {'func': func, 'phenotypic': phenotypic,
               'mask': mask,
               'description': "The Human connectome project"}
    return Bunch(**results)


def get_hcp_data(raw=False, data_dir=None):
    data_dir = get_data_dirs(data_dir)[0]
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
            raise IOError(
                'Please unmask the data using hcp_prepare.py first.')
        func_filenames = sorted(list(mapping.values()))
    else:
        hcp_dataset = fetch_hcp_rest(data_dir=data_dir,
                                     n_subjects=500)
        mask = hcp_dataset.mask
        # list of 4D nifti files for each subject
        func_filenames = hcp_dataset.func
        # Flatten it
        func_filenames = [record for subject in func_filenames for record in
                          subject]

        # print basic information on the dataset
        print('First functional nifti image (4D) is at: %s' %
              hcp_dataset.func[0])  # 4D data
    return mask, func_filenames
