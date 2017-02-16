import os
from os.path import join

from sacred import Experiment
from sacred.observers import MongoObserver

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sklearn.externals.joblib import Memory

from modl.datasets import fetch_adhd
from modl.datasets import fetch_hcp, get_data_dirs
from modl.input_data.fmri.unmask import create_raw_rest_data, get_raw_rest_data
from modl.utils.system import get_cache_dirs

unmask_rest = Experiment('unmask_rest')
observer = MongoObserver.create(db_name='amensch', collection='runs')
unmask_rest.observers.append(observer)


@unmask_rest.config
def config():
    source = 'adhd'
    smoothing_fwhm = 6
    n_jobs = 3


@unmask_rest.named_config
def hcp():
    source = 'hcp'
    smoothing_fwhm = 4
    n_jobs = 36
    n_jobs = 3


@unmask_rest.command
def _test(source):
    if source == 'hcp':
        dataset = fetch_hcp()
        smoothing_fwhm = 4
    elif source == 'adhd':
        dataset = fetch_adhd()
        smoothing_fwhm = 6
    else:
        raise ValueError('Wrong source.')
    imgs_list = dataset.rest
    print('masked len %i' % len(imgs_list))

    raw_dir = join(get_data_dirs()[0], 'pipeline', 'unmask',
                   'rest', source, str(smoothing_fwhm))

    masker, imgs = get_raw_rest_data(imgs_list, raw_dir)
    masker.fit()
    imgs = imgs['filename'].values
    for i, img in enumerate(imgs):
        data = masker.transform(img, mmap_mode='r')
        del data


@unmask_rest.automain
def run(source, smoothing_fwhm, n_jobs,
        _run):
    if source == 'hcp':
        dataset = fetch_hcp()
    elif source == 'adhd':
        dataset = fetch_adhd()
    else:
        raise ValueError('Wrong source.')

    memory = Memory(cachedir=get_cache_dirs()[0])
    imgs_list = dataset.rest
    root = dataset.root
    mask_img = dataset.mask

    artifact_dir = join(get_data_dirs()[0], 'pipeline',
                        'unmask', 'rest', source, str(smoothing_fwhm))
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    _run.info['artifact_dir'] = artifact_dir

    create_raw_rest_data(imgs_list,
                         root=root,
                         raw_dir=artifact_dir,
                         masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                            detrend=True,
                                            standardize=True,
                                            mask_img=mask_img),
                         memory=memory,
                         n_jobs=n_jobs)

    with open(join(artifact_dir, 'exp_id'), 'w+') as f:
        f.write(str(_run._id))
