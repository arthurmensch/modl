import os
from os.path import join

from modl.input_data.fmri.fixes import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sklearn.externals.joblib import Memory

from modl.input_data.fmri.rest import create_raw_rest_data
from modl.utils.system import get_cache_dirs, get_output_dir

from hcp_builder.dataset import fetch_hcp

smoothing_fwhm = 4
n_jobs = 3

dataset = fetch_hcp()

memory = Memory(cachedir=get_cache_dirs()[0])
imgs_list = dataset.rest
root = dataset.root
mask_img = dataset.mask

artifact_dir = join(get_output_dir(), 'unmasked', 'hcp')
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
                     memory=memory,
                     n_jobs=n_jobs)
