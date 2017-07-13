import os
from os.path import join

from modl.datasets import fetch_adhd
from modl.input_data.fmri.fixes import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sklearn.externals.joblib import Memory

from modl.input_data.fmri.rest import create_raw_rest_data
from modl.utils.system import get_cache_dirs, get_output_dir

smoothing_fwhm = 6
n_jobs = 20

dataset = fetch_adhd()

memory = Memory(cachedir=get_cache_dirs()[0])
imgs_list = dataset.rest
root = dataset.root
mask_img = dataset.mask

artifact_dir = join(get_output_dir(), 'unmasked', 'adhd_6')
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
