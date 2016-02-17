import json
from os.path import join

import numpy as np
from nilearn import _utils
from sklearn.externals.joblib import Memory

from modl._utils.masking.multi_nifti_masker import MultiNiftiMasker


class DummyMasker(MultiNiftiMasker):
    def __init__(self, data_dir=None,
                 mmap_mode='r',
                 mask_img=None, smoothing_fwhm=None,
                 standardize=False,
                 detrend=False, low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0, n_jobs=1,
                 verbose=0):
        super().__init__(mask_img, smoothing_fwhm, standardize, detrend,
                         low_pass, high_pass, t_r, target_affine, target_shape,
                         mask_strategy, mask_args, memory, memory_level,
                         n_jobs, verbose)
        self.data_dir = data_dir
        self.mmap_mode = mmap_mode

    def fit(self, imgs=None, y=None):
        self.mask_img_ = _utils.check_niimg_3d(self.mask_img)
        with open(join(self.data_dir, 'mapping.json'), 'r') as f:
            self.mapping_ = json.load(f)

    def transform_single_imgs(self, imgs, confounds=None, copy=True,
                              ):
        return np.load(self.mapping_[imgs], mmap_mode=self.mmap_mode)
