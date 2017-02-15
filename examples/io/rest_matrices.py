from os.path import join

from sacred import Experiment

from modl.datasets import fetch_adhd
from modl.datasets import fetch_hcp, get_data_dirs
from modl.input_data.fmri.raw_masker import create_raw_data, get_raw_data

unmask = Experiment('unmask')

@unmask.config
def config():
    source = 'hcp'

@unmask.command
def test(source):
    if source == 'hcp':
        dataset = fetch_hcp()
        smoothing_fwhm = 4
    elif source == 'adhd':
        dataset = fetch_adhd()
        smoothing_fwhm = 6
    else:
        raise ValueError('Wrong source.')
    imgs_list = dataset.rest
    root = dataset.root
    mask_img = dataset.mask
    n_jobs = 36
    print('masked len %i' % len(imgs_list))

    raw_dir = join(get_data_dirs()[0], 'raw', source, str(smoothing_fwhm))

    masker, imgs = get_raw_data(imgs_list, raw_dir)
    masker.fit()
    imgs = imgs['filename'].values
    print('unmasked len %i' % len(imgs))
    for i, img in enumerate(imgs):
        data = masker.transform(img, mmap_mode='r')
        del data

@unmask.automain
def run(source):
    if source == 'hcp':
        dataset = fetch_hcp()
        smoothing_fwhm = 4
    elif source == 'adhd':
        dataset = fetch_adhd()
        smoothing_fwhm = 6
    else:
        raise ValueError('Wrong source.')
    imgs_list = dataset.rest
    root = dataset.root
    mask_img = dataset.mask
    n_jobs = 36

    raw_dir = join(get_data_dirs()[0], 'raw', source, str(smoothing_fwhm))

    create_raw_data(imgs_list,
                    root=root,
                    raw_dir=raw_dir,
                    masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                       detrend=True,
                                       standardize=True,
                                       mask_img=mask_img),
                    n_jobs=n_jobs)