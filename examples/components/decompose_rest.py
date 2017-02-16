import os
import shutil
import warnings
from os.path import join

from matplotlib.cbook import MatplotlibDeprecationWarning

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from modl.input_data.fmri.unmask import get_raw_rest_data

from modl.input_data.fmri.base import safe_to_filename

from modl.datasets import fetch_adhd, fetch_hcp, get_data_dirs
from modl.decomposition.fmri import rfMRIDictionaryScorer, fMRIDictFact
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

decompose_rest = Experiment('decompose_rest')
observer = MongoObserver.create(db_name='amensch', collection='runs')
decompose_rest.observers.append(observer)


# noinspection PyUnusedLocal
@decompose_rest.config
def config():
    batch_size = 100
    learning_rate = 0.92
    method = 'gram'
    reduction = 12
    alpha = 1e-4
    n_epochs = 1
    smoothing_fwhm = 6
    n_components = 200
    n_jobs = 1
    verbose = 15
    seed = 2
    use_resource = True
    callback = True
    # Data
    source = 'adhd'
    n_subjects = 40
    train_size = 36
    test_size = 4
    seed = 2


# noinspection PyUnusedLocal
@decompose_rest.named_config
def hcp(rest_data):
    batch_size = 100
    smoothing_fwhm = 4
    # Data
    source = 'hcp'
    n_subjects = 788
    test_size = 1
    train_size = 787


@decompose_rest.capture
def get_rest_data(source, test_size, train_size, _run, _seed,
                  # Optional arguments
                  n_subjects,
                  use_resource,
                  smoothing_fwhm,
                  train_subjects=None,
                  test_subjects=None
                  ):
    if source == 'hcp':
        data = fetch_hcp(n_subjects=n_subjects)
    elif source == 'adhd':
        data = fetch_adhd(n_subjects=n_subjects)
    else:
        raise ValueError('Wrong resting-state source')
    imgs = data.rest
    mask = data.mask
    subjects = imgs.index.get_level_values('subject').unique().values

    if use_resource is True:
        # WARNING: this is a hack to use unmasked time series without
        # touching the core code
        unmasking_rest_dir = join(get_data_dirs()[0], 'pipeline', 'unmask',
                                  'rest',
                                  source,
                                  str(smoothing_fwhm))
        with open(join(unmasking_rest_dir, 'exp_id'), 'r') as f:
            exp_id = f.read()
        # We won't record resources as this is too much data
        _run.info['resource_dir'] = {'unmasking_rest': unmasking_rest_dir,
                                     'unmasking_rest_id': exp_id}
        mask, unmasked_imgs = get_raw_rest_data(unmasking_rest_dir)
        if source == 'adhd':
            unmasked_imgs.set_index('subject', inplace=True)
        else:
            unmasked_imgs.set_index(['subject', 'session', 'direction'],
                                    inplace=True)
        # TODO should assert that all record of imgs are in unmasked_dir
        imgs = unmasked_imgs

    if train_subjects is None and test_subjects is None:
        train_subjects, test_subjects = train_test_split(
            subjects, random_state=_seed, test_size=test_size)
        train_subjects = train_subjects.tolist()
        test_subjects = test_subjects.tolist()
    train_subjects = train_subjects[:train_size]
    test_subjects = test_subjects[:test_size]

    imgs = pd.concat([imgs.loc[train_subjects],
                      imgs.loc[test_subjects]], keys=['train', 'test'])

    _run.info['train_subjects'] = train_subjects
    _run.info['test_subjects'] = test_subjects
    # noinspection PyUnboundLocalVariable
    return imgs, mask


class CapturedfMRIDictionaryScorer(rfMRIDictionaryScorer):
    def __init__(self, test_imgs, test_confounds=None,
                 intermediary_dir=None):

        rfMRIDictionaryScorer.__init__(self, test_imgs,
                                       test_confounds=test_confounds)
        self.intermediary_dir = intermediary_dir
        if self.intermediary_dir is not None:
            if not os.path.exists(self.intermediary_dir):
                os.makedirs(self.intermediary_dir)
            self.intermediary_components = []
            self.call_count = 0

    @decompose_rest.capture
    def __call__(self, masker, dict_fact, _run=None):
        rfMRIDictionaryScorer.__call__(self, masker, dict_fact)
        _run.info['score'] = self.score
        _run.info['time'] = self.time
        _run.info['iter'] = self.iter
        if self.intermediary_dir is not None:
            call_count = len(self.intermediary_components)
            components_img = masker.inverse_transform(dict_fact.components_)
            components_name = 'components_%i.nii.gz' % call_count
            self.intermediary_components.append(components_name)
            components_img.to_filename(join(self.intermediary_dir,
                                            components_name))
            # Disable artifacts as it is too heavy
            # _run.add_artifact(join(self.intermediary_dir, components_name),
            #                   name=components_name)
            _run.info['intermediary_components'] = self.intermediary_components


@decompose_rest.capture
def compute_decomposition(alpha, batch_size, learning_rate,
                          n_components,
                          n_epochs,
                          n_jobs,
                          reduction,
                          smoothing_fwhm,
                          method,
                          verbose,
                          source,
                          callback,
                          _run,
                          _seed,
                          train_subjects=None,
                          test_subjects=None,
                          ):
    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'components',
                        source, str(n_components), str(alpha))
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    _run.info['artifact_dir'] = artifact_dir

    memory = Memory(cachedir=get_cache_dirs()[0])

    print('Retrieve resting-state data')
    imgs_list, mask = get_rest_data(train_subjects=train_subjects,
                                    test_subjects=test_subjects)
    print('Run dictionary learning')
    train_imgs, test_imgs = imgs_list.loc['train'], imgs_list.loc['test']

    if callback:
        callback = CapturedfMRIDictionaryScorer(test_imgs['filename'],
                                                test_confounds
                                                =test_imgs['confounds'],
                                                intermediary_dir=
                                                join(artifact_dir,
                                                     'intermediary'))
    else:
        callback = None

    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             method=method,
                             mask=mask,
                             memory=memory,
                             memory_level=2,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=_seed,
                             n_components=n_components,
                             dict_init=None,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=callback,
                             )
    dict_fact.fit(train_imgs['filename'], confounds=train_imgs['confounds'])

    final_score = dict_fact.score(test_imgs['filename'],
                                  confounds=test_imgs['confounds'])

    _run.info['score'] = callback.score
    _run.info['time'] = callback.time
    _run.info['iter'] = callback.iter
    _run.info['final_score'] = final_score

    print('Write components artifacts')
    safe_to_filename(dict_fact.components_img_,
                     join(artifact_dir, 'components.nii.gz'))
    _run.add_artifact(join(artifact_dir, 'components.nii.gz'),
                      name='components.nii.gz')

    safe_to_filename(dict_fact.mask_img_,
                     join(artifact_dir, 'mask_img.nii.gz'))
    _run.add_artifact(join(artifact_dir, 'mask_img.nii.gz'),
                      name='mask_img.nii.gz')

    fig = plt.figure()
    display_maps(fig, dict_fact.components_img_)
    plt.savefig(join(artifact_dir, 'components.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'components.png'),
                      name='components.png')
    fig, ax = plt.subplots(1, 1)
    ax.plot(callback.time, callback.score, marker='o')
    plt.savefig(join(artifact_dir, 'learning_curve.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'learning_curve.png'),
                      name='learning_curve.png')

    with open(join(artifact_dir, 'exp_id'), 'w+') as f:
        f.write(str(_run._id))

    return dict_fact, final_score


@decompose_rest.automain
def run_decomposition_ex(_run):
    compute_decomposition()
