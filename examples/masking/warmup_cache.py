import os
from os.path import join

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sacred import Experiment

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
    n_jobs = 1
    source = 'hcp'
    n_subjects = 3
    smoothing_fwhm = 4


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
def main(n_jobs, source, n_subjects, smoothing_fwhm):

    mem = Memory(cachedir=get_cache_dirs()[0], verbose=10)

    failure = join(get_cache_dirs()[0], 'cache_failure')
    if not os.path.exists(failure):
        os.makedirs(failure)

    if source == 'hcp':
        data = fetch_hcp(n_subjects=n_subjects)
    elif source == 'adhd':
        data = fetch_adhd(n_subjects=n_subjects)
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
    Parallel(n_jobs=n_jobs)(delayed(mask_and_dismiss)(masker,
                                                      index,
                                                      img,
                                                      confounds)
                            for index, (img, confounds) in
                            rest.loc[:, ['filename', 'confounds']].iterrows())
