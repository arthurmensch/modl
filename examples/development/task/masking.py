import os
from os.path import join

from modl.utils.nifti import monkey_patch_nifti_image

monkey_patch_nifti_image()

import pandas as pd
from modl.datasets import fetch_adhd
from modl.datasets.hcp import fetch_hcp
from modl.utils.system import get_cache_dirs
from nilearn.input_data import MultiNiftiMasker, NiftiMasker
from nilearn._utils import check_niimg
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


def mask_and_dismiss(masker, index, img):
    try:
        img = check_niimg(img)
        print('Masking %s %s %s' % index)
        data = masker.transform(img, confounds=None)
        del data
    except:
        print('Bad record %s %s %s' % index)
        failure = join(get_cache_dirs()[0], 'cache_failure')
        with open(join(failure, 'failure_%s_%s_%s' % index), 'w+') as f:
            f.write('Cache failed')


def main():
    n_jobs = 3
    source = 'hcp'

    mem = Memory(cachedir=get_cache_dirs()[0], verbose=10)

    failure = join(get_cache_dirs()[0], 'cache_failure')
    if not os.path.exists(failure):
        os.makedirs(failure)

    if source == 'hcp':
        data = fetch_hcp(n_subjects=788)
        rest_data = data.rest
        rest_imgs = rest_data.loc[:, 'filename']
        rest_mask = data.mask
        smoothing_fwhm = 4
    elif source == 'adhd':
        data = fetch_adhd(n_subjects=40)
        rest_imgs = pd.Series(data.func)
        rest_mask = data.mask
        smoothing_fwhm = 6

    masker = MultiNiftiMasker(smoothing_fwhm=smoothing_fwhm,
                              standardize=True,
                              detrend=True,
                              mask_img=rest_mask,
                              memory=mem,
                              memory_level=1).fit()
    Parallel(n_jobs=n_jobs)(delayed(mask_and_dismiss)(masker,
                                                      index,
                                                      img)
                            for index, img in rest_imgs.iteritems())


if __name__ == '__main__':
    main()
