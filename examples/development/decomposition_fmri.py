import copy
import shutil
import warnings
from os.path import expanduser, join
from tempfile import mkdtemp, mktemp

import matplotlib.pyplot as plt
from matplotlib.cbook import MatplotlibDeprecationWarning
from modl.datasets import fetch_adhd
from modl.datasets.hcp import fetch_hcp
from modl.fmri import rfMRIDictionaryScorer, fmri_dict_learning
from modl.plotting.fmri import display_maps
from modl.utils.nifti import monkey_patch_nifti_image
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred import Ingredient
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

monkey_patch_nifti_image()

import pandas as pd
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
rest_data_ing = Ingredient('rest_data')
decomposition_ex = Experiment('decomposition', ingredients=[rest_data_ing])

observer = FileStorageObserver.create(expanduser('~/runs'))
decomposition_ex.observers.append(observer)
observer = MongoObserver.create(db_name='amensch', collection='runs')
decomposition_ex.observers.append(observer)


@decomposition_ex.config
def config():
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    smoothing_fwhm = 6
    n_components = 40
    n_jobs = 20
    verbose = 15
    seed = 2

@rest_data_ing.config
def config():
    source = 'adhd'
    n_subjects = 40
    test_size = 2


@rest_data_ing.capture
def get_rest_data(source, n_subjects, test_size, _run, _seed):
    if source == 'hcp':
        data = fetch_hcp(n_subjects=n_subjects)
        imgs = data.rest
        mask_img = data.mask
        subjects = imgs.index.levels[0].values
    elif source == 'adhd':
        data = fetch_adhd(n_subjects=40)
        imgs = data.func
        # TODO merge this in adhd fetcher
        imgs = pd.DataFrame(columns=['filename'],
                            data=pd.Series(imgs, index=pd.Index(
                                np.arange(len(imgs)), name='subject')))
        mask_img = data.mask
        subjects = imgs.index.values
    else:
        raise ValueError('Wrong resting-state source')
    train_subjects, test_subjects = train_test_split(
        subjects, random_state=_seed, test_size=test_size)
    train_subjects = train_subjects.tolist()
    test_subjects = test_subjects.tolist()
    train_imgs = imgs.loc[train_subjects, 'filename'].values
    test_imgs = imgs.loc[test_subjects, 'filename'].values
    _run.info['train_subjects'] = 'train_subjects'
    _run.info['test_subjects'] = 'test_subjects'
    return train_imgs, test_imgs, mask_img


class CapturedfMRIDictionaryScorer(rfMRIDictionaryScorer):
    @decomposition_ex.capture
    def __call__(self, dict_fact, _run=None):
        rfMRIDictionaryScorer.__call__(self, dict_fact)
        _run.info['score'] = self.score
        _run.info['time'] = self.time
        _run.info['iter'] = self.iter


@decomposition_ex.capture
def compute_decomposition(alpha, batch_size, learning_rate,
                          n_components,
                          n_epochs,
                          n_jobs,
                          reduction,
                          smoothing_fwhm,
                          method,
                          verbose,
                          _run, _seed):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=0)
    print('Retrieve resting-state data')
    train_imgs, test_imgs, mask_img = get_rest_data()
    callback = CapturedfMRIDictionaryScorer(test_imgs)
    print('Run dictionary learning')
    components, mask_img, callback = memory.cache(
        fmri_dict_learning,
        ignore=['verbose', 'n_jobs',
                'memory',
                'memory_level',
                'callback'])(
        train_imgs,
        mask=mask_img,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_components=n_components,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        reduction=reduction,
        memory=memory,
        memory_level=2,
        smoothing_fwhm=smoothing_fwhm,
        method=method,
        callback=callback,
        random_state=_seed,
        verbose=verbose)

    print('Dump results')
    _run.info['score'] = callback.score
    _run.info['time'] = callback.time
    _run.info['iter'] = callback.iter
    _run.info['final_score'] = callback.score[-1]
    artifact_dir = mkdtemp()
    # Avoid trashing cache
    components_copy = copy.deepcopy(components)
    components_copy.to_filename(join(artifact_dir, 'components.nii.gz'))
    mask_img_copy = copy.deepcopy(mask_img)
    mask_img_copy.to_filename(join(artifact_dir, 'mask_img.nii.gz'))
    _run.add_artifact(join(artifact_dir, 'components.nii.gz'),
                      name='components.nii.gz')
    fig = plt.figure()
    display_maps(fig, components)
    plt.savefig(join(artifact_dir, 'components.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'components.png'),
                      name='components.png')
    fig, ax = plt.subplots(1, 1)
    ax.plot(_run.info['time'], _run.info['score'], marker='o')
    plt.savefig(join(artifact_dir, 'learning_curve.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'learning_curve.png'),
                      name='learning_curve.png')
    try:
        shutil.rmtree(artifact_dir)
    except FileNotFoundError:
        pass
    return components, mask_img

@decomposition_ex.automain
def run_decomposition_ex(_run):
    compute_decomposition()
    return _run.info['final_score']
