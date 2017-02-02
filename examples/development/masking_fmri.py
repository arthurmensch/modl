import os
from os.path import join

from modl.utils.nifti import monkey_patch_nifti_image
from sacred import Experiment

monkey_patch_nifti_image()

from modl.datasets import fetch_adhd
from modl.datasets.hcp import fetch_hcp
from modl.utils.system import get_cache_dirs
from nilearn.input_data import MultiNiftiMasker
from nilearn._utils import check_niimg
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


masking_ex = Experiment('masking_fmri')

@masking_ex.config
def config():
    n_jobs = 3
    source = 'hcp'


def mask_and_dismiss(masker, index, img, confounds):
    try:
        img = check_niimg(img)
    except EOFError:
        print('Bad record %s' % str(index))
        failure = join(get_cache_dirs()[0], 'cache_failure')
        with open(join(failure, 'failure_%s' % str(index)), 'w+') as f:
            f.write('Cache failed')
    print('Masking %s' % str(index))
    data = masker.transform(img, confounds=confounds)
    del data

@masking_ex.automain
def main(n_jobs, source):

    mem = Memory(cachedir=get_cache_dirs()[0], verbose=10)

    failure = join(get_cache_dirs()[0], 'cache_failure')
    if not os.path.exists(failure):
        os.makedirs(failure)

    if source == 'hcp':
        data = fetch_hcp(n_subjects=788)
        smoothing_fwhm = 4
    elif source == 'adhd':
        data = fetch_adhd(n_subjects=40)
        smoothing_fwhm = 6
    else:
        raise ValueError('Wrong source')
    rest = data.rest
    mask = data.mask

    masker = MultiNiftiMasker(smoothing_fwhm=smoothing_fwhm,
                              standardize=True,
                              detrend=True,
                              mask_img=mask,
                              memory=mem,
                              memory_level=1).fit()
    iter_df = rest.loc[:, ['filename', 'confounds']].iterrows()
    Parallel(n_jobs=n_jobs)(delayed(mask_and_dismiss)(masker,
                                                      index,
                                                      img,
                                                      confounds)
                            for index, (img, confounds) in iter_df)