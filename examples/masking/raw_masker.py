from os.path import join

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from nilearn.input_data import MultiNiftiMasker

from modl.datasets import get_data_dirs
from modl.datasets.hcp import fetch_hcp
from modl.input_data.fmri.raw_masker import create_raw_data, get_raw_data

smoothing_fwhm = 4

dataset = fetch_hcp(n_subjects=1)
imgs_list = dataset.rest
mask = dataset.mask
masker = MultiNiftiMasker(smoothing_fwhm=smoothing_fwhm, detrend=True,
                          standardize=True, mask_img=dataset.mask,
                          verbose=1).fit()

raw_dir = join(get_data_dirs()[0], 'hcp', str(smoothing_fwhm))
root = dataset.root

create_raw_data(imgs_list, root=root,
                masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                   detrend=True,
                                   standardize=True,
                                   mask_img=mask),
                raw_dir=raw_dir, mock=False,
                n_jobs=1)