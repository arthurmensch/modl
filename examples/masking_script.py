import json
from os.path import expanduser, join

from hcp_builder.dataset import fetch_hcp
from nilearn.input_data import MultiNiftiMasker

from modl.input_data.fmri.unmask import create_raw_rest_data

import os

import pandas as pd


def get_raw_rest_data(raw_dir):
    if not os.path.exists(raw_dir):
        raise ValueError('Unmask directory %s does not exist.'
                         'Unmasking must be done beforehand.' % raw_dir)
    params = json.load(open(join(raw_dir, 'masker.json'), 'r'))
    masker = MultiNiftiMasker(**params)
    unmasked_imgs_list = pd.read_csv(join(raw_dir, 'data.csv'))
    return masker, unmasked_imgs_list

source = 'hcp'
smoothing_fwhm = 4
n_jobs = 30

dataset = fetch_hcp(n_subjects=16)

imgs_list = dataset.rest
root = dataset.root
mask_img = dataset.mask

unmasking_dir = expanduser('~/data/hcp_olivier')

create_raw_rest_data(imgs_list, root=root, raw_dir=unmasking_dir,
                     masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                        detrend=True,
                                        standardize=True,
                                        mask_img=mask_img),
                     overwrite=True,
                     n_jobs=n_jobs)
mask, unmasked_imgs = get_raw_rest_data(unmasking_dir)
print(unmasked_imgs)
