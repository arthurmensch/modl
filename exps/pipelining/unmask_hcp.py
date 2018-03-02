import os
from os.path import join

from cogspaces.datasets.utils import fetch_mask
from hcp_builder.dataset import fetch_hcp, fetch_hcp_timeseries
from modl.input_data.fmri.rest import create_raw_rest_data
from modl.utils.system import get_output_dir

smoothing_fwhm = 4
n_jobs = 20

imgs_list = fetch_hcp_timeseries(None, data_type='rest',
                                 n_subjects=None, subjects=None,
                                 on_disk=True)

root = '/storage/store/data/HCP900'
mask_img = fetch_mask()['icbm_gm']

artifact_dir = join(get_output_dir(), 'unmasked', 'hcp_icbm_gm')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

create_raw_rest_data(imgs_list,
                     root=root,
                     raw_dir=artifact_dir,
                     overwrite=False,
                     mock=False,
                     masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                        detrend=True,
                                        standardize=True,
                                        mask_img=mask_img),
                     memory=None,
                     n_jobs=n_jobs)
