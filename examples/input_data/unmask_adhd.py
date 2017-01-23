import os
from os.path import expanduser

from nilearn.input_data import MultiNiftiMasker

from modl.input_data import unmask_dataset
from modl.datasets import fetch_adhd
import re

dataset = fetch_adhd(3)

imgs = dataset.func
confounds = dataset.confounds
mask = dataset.mask

regex = re.compile(r"adhd.*")
base_dir = regex.sub('adhd', imgs[0])
dest_dir = expanduser('~/pipelines/unmask/adhd')

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=4, detrend=True,
                          standardize=True).fit()
masker.mask_img_.get_data()
unmask_dataset(masker, imgs, base_dir=base_dir, dest_dir=dest_dir)