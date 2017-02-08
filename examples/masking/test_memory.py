from os.path import join

import numpy as np
import nibabel

from modl.datasets.hcp import fetch_hcp
from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

dataset = fetch_hcp(n_subjects=1)

imgs = dataset.rest.iloc[0]['filename']

imgs = nibabel.load(imgs)

data = np.asanyarray(imgs._dataobj)
