from os.path import join

from hcp_builder.dataset import fetch_hcp as _hcpbuild_fetch_hcp
from sklearn.datasets.base import Bunch

from modl.datasets import get_data_dirs

import pandas as pd

idx = pd.IndexSlice

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
