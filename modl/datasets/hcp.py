from os.path import join

from hcp_builder.dataset import fetch_hcp as _hcpbuild_fetch_hcp
from sklearn.datasets.base import Bunch

from modl.datasets import get_data_dirs

import pandas as pd

idx = pd.IndexSlice

INTERESTING_CONTRASTS_DICT = {'2BK': {'Cognitive Task': 'Two-Back Memory',
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
                              'SHAPES': {'Cognitive Task': 'Faces',
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

INTERESTING_CONTRASTS = list(INTERESTING_CONTRASTS_DICT.keys())

BASE_CONTRASTS = ['FACES', 'SHAPES', 'PUNISH', 'REWARD',
                  'MATH', 'STORY', 'MATCH', 'REL',
                  'RANDOM', 'TOM',
                  'LF', 'RF', 'LH', 'RH', 'CUE',
                  '0BK_BODY', '0BK_FACE', '0BK_PLACE',
                  '0BK_TOOL',
                  '2BK_BODY', '2BK_FACE', '2BK_PLACE',
                  '2BK_TOOL',
                  ]


def fetch_hcp(data_dir=None, n_subjects=None, subjects=None,
              from_file=True):
    data_dir = join(get_data_dirs(data_dir)[0], 'HCP900')
    res = _hcpbuild_fetch_hcp(data_dir=data_dir, n_subjects=n_subjects,
                              from_file=from_file,
                              subjects=subjects, on_disk=True)
    rest = res.rest.assign(confounds=[None] * res.rest.shape[0])
    task = res.task.assign(confounds=[None] * res.task.shape[0])
    contrasts = res.contrasts.loc[idx[:, :, BASE_CONTRASTS, :], :].copy()
    contrasts.sort_index(inplace=True)
    task.sort_index(inplace=True)
    rest.sort_index(inplace=True)

    return Bunch(rest=rest,
                 contrasts=contrasts,
                 task=task,
                 behavioral=res.behavioral,
                 mask=res.mask,
                 root=res.root)
