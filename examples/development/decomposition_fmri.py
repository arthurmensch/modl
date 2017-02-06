import copy
import shutil
import warnings
from os.path import expanduser, join
from tempfile import mkdtemp

from matplotlib.cbook import MatplotlibDeprecationWarning
from sacred.observers import FileStorageObserver

from modl.datasets import fetch_adhd
from modl.datasets.hcp import fetch_hcp
from modl.fmri import rfMRIDictionaryScorer, fmri_dict_learning, fMRICoder
from modl.plotting.fmri import display_maps
from modl.utils.nifti import monkey_patch_nifti_image
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred import Ingredient
from sacred.observers import MongoObserver

monkey_patch_nifti_image()

from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
rest_data_ing = Ingredient('rest_data')
decomposition_ex = Experiment('decomposition', ingredients=[rest_data_ing])

observer = MongoObserver.create(db_name='amensch', collection='runs')
decomposition_ex.observers.append(observer)

observer = FileStorageObserver.create(expanduser('~/output/runs'))
decomposition_ex.observers.append(observer)


@decomposition_ex.config
def config():
    batch_size = 50
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-4
    n_epochs = 1
    smoothing_fwhm = 4
    n_components = 40
    n_jobs = 1
    verbose = 15
    seed = 2


@rest_data_ing.config
def config():
    source = 'adhd'
    train_size = 36
    test_size = 4
    seed = 2


@decomposition_ex.named_config
def hcp():
    batch_size = 100


@rest_data_ing.named_config
def hcp():
    source = 'hcp'
    test_size = 1
    train_size = 10


@rest_data_ing.capture
def get_rest_data(source, test_size, train_size, _run, _seed,
                  # Optional arguments
                  train_subjects=None,
                  test_subjects=None
                  ):
    if source == 'hcp':
        data = fetch_hcp()
    elif source == 'adhd':
        data = fetch_adhd()
    else:
        raise ValueError('Wrong resting-state source')
    imgs = data.rest
    mask_img = data.mask
    subjects = imgs.index.get_level_values('subject').unique().values

    if train_subjects is None and test_subjects is None:
        train_subjects, test_subjects = train_test_split(
            subjects, random_state=_seed, test_size=test_size)
        train_subjects = train_subjects.tolist()
        test_subjects = test_subjects.tolist()
    train_subjects = train_subjects[:train_size]
    test_subjects = test_subjects[:test_size]

    train_imgs = imgs.loc[train_subjects, 'filename'].values
    test_imgs = imgs.loc[test_subjects, 'filename'].values
    train_confounds = imgs.loc[train_subjects, 'confounds'].values
    test_confounds = imgs.loc[test_subjects, 'confounds'].values

    _run.info['dec_train_subjects'] = train_subjects
    _run.info['dec_test_subjects'] = test_subjects
    # noinspection PyUnboundLocalVariable
    return train_imgs, train_confounds, test_imgs, test_confounds, mask_img


class CapturedfMRIDictionaryScorer(rfMRIDictionaryScorer):
    @decomposition_ex.capture
    def __call__(self, dict_fact, _run=None):
        rfMRIDictionaryScorer.__call__(self, dict_fact)
        _run.info['dec_score'] = self.score
        _run.info['dec_time'] = self.time
        _run.info['dec_iter'] = self.iter


@decomposition_ex.capture
def compute_decomposition(alpha, batch_size, learning_rate,
                          n_components,
                          n_epochs,
                          n_jobs,
                          reduction,
                          smoothing_fwhm,
                          method,
                          verbose,
                          _run, _seed,
                          # Optional arguments, to be passed by higher level experiments
                          train_subjects=None,
                          test_subjects=None,
                          observe=True,
                          return_score=True
                          ):
    observe = observe and not _run.unobserved

    memory = Memory(cachedir=get_cache_dirs()[0],
                    mmap_mode=None, verbose=0)
    print('Retrieve resting-state data')
    train_imgs, train_confounds, \
    test_imgs, test_confounds, mask_img = get_rest_data(
        train_subjects=train_subjects,
        test_subjects=test_subjects)
    if observe:
        callback = CapturedfMRIDictionaryScorer(test_imgs,
                                                test_confounds=test_confounds)
    else:
        callback = None
    print('Run dictionary learning')
    components, mask_img, callback = memory.cache(
        fmri_dict_learning,
        ignore=['verbose', 'n_jobs',
                'memory',
                'memory_level'])(
        train_imgs,
        confounds=train_confounds,
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

    if return_score:
        if observe:
            _run.info['dec_score'] = callback.score
            _run.info['dec_time'] = callback.time
            _run.info['dec_iter'] = callback.iter
        coder = fMRICoder(standardize=True, detrend=True,
                          smoothing_fwhm=smoothing_fwhm,
                          dictionary=components,
                          memory=memory,
                          alpha=alpha,
                          memory_level=2,
                          mask=mask_img).fit()
        final_score = coder.score(test_imgs, confounds=test_confounds)
    elif observe:
        raise ValueError('return_score should be True when observing')

    if observe:
        _run.info['dec_final_score'] = final_score
        print('Write decomposition artifacts')
        artifact_dir = mkdtemp()
        # Avoid trashing cache
        components_copy = copy.deepcopy(components)
        components_copy.to_filename(
            join(artifact_dir, 'dec_components.nii.gz'))
        mask_img_copy = copy.deepcopy(mask_img)
        mask_img_copy.to_filename(join(artifact_dir, 'dec_mask_img.nii.gz'))
        _run.add_artifact(join(artifact_dir, 'dec_components.nii.gz'),
                          name='dec_components.nii.gz')
        fig = plt.figure()
        display_maps(fig, components)
        plt.savefig(join(artifact_dir, 'dec_components.png'))
        plt.close(fig)
        _run.add_artifact(join(artifact_dir, 'dec_components.png'),
                          name='dec_components.png')
        fig, ax = plt.subplots(1, 1)
        ax.plot(callback.time, callback.score, marker='o')
        plt.savefig(join(artifact_dir, 'dec_learning_curve.png'))
        plt.close(fig)
        _run.add_artifact(join(artifact_dir, 'dec_learning_curve.png'),
                          name='dec_learning_curve.png')
        try:
            shutil.rmtree(artifact_dir)
        except FileNotFoundError:
            pass
    if return_score:
        return components, mask_img, final_score
    else:
        return components, mask_img


@decomposition_ex.automain
def run_decomposition_ex(_run):
    compute_decomposition()
