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

from modl.utils.nifti import NibabelHasher


def mask_and_dismiss(masker, index, img):
    try:
        img = check_niimg(img)
        hasher = NibabelHasher('md5', coerce_mmap=False)
        hash = hasher.hash(img)
        print('Masking %s ' % index)
        print('Hash %s' % hash)
        hasher = NibabelHasher('md5', coerce_mmap=False)
        hash = hasher.hash(masker.mask_img_)
        print('Mask hash %s' % hash)

        data = masker.transform(img, confounds=None)
        del data
    except:
        name = index
        print('Bad record %s' % name)
        failure = join(get_cache_dirs()[0], 'cache_failure')
        with open(join(failure, 'failure_%s' % name), 'w+') as f:
            f.write('Cache failed')


def main():
    n_jobs = 3
    source = 'adhd'

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
        data = fetch_adhd(n_subjects=4)
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
