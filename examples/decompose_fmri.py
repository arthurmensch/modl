import shutil
import warnings
from os.path import join
from tempfile import mkdtemp

from matplotlib.cbook import MatplotlibDeprecationWarning

from modl.input_data.fmri.monkey import monkey_patch_nifti_image
from modl.input_data.fmri.raw_masker import get_raw_data, create_raw_data

monkey_patch_nifti_image()

from modl.input_data.fmri.base import safe_to_filename

from modl.datasets import fetch_adhd, fetch_hcp, get_data_dirs
from modl.decomposition.fmri import rfMRIDictionaryScorer, fMRIDictFact
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred import Ingredient
from sacred.observers import MongoObserver

from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
rest_data_ing = Ingredient('rest_data')
decomposition_ex = Experiment('decomposition', ingredients=[rest_data_ing])

observer = MongoObserver.create(db_name='amensch', collection='runs')
decomposition_ex.observers.append(observer)


@decomposition_ex.config
def config(rest_data):
    batch_size = 50
    learning_rate = 0.92
    method = 'gram'
    reduction = 10
    alpha = 1e-4
    n_epochs = 3
    smoothing_fwhm = 4
    n_components = 200
    n_jobs = 1
    verbose = 15
    seed = 2
    raw_dir = join(get_data_dirs()[0], 'raw', rest_data['source'],
                   str(smoothing_fwhm))


@rest_data_ing.config
def config():
    source = 'hcp'
    n_subjects = 788
    train_size = 787
    test_size = 1
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
                  n_subjects,
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
    mask_img = data.mask
    root = data.root
    subjects = imgs.index.get_level_values('subject').unique().values

    if train_subjects is None and test_subjects is None:
        train_subjects, test_subjects = train_test_split(
            subjects, random_state=_seed, test_size=test_size)
        train_subjects = train_subjects.tolist()
        test_subjects = test_subjects.tolist()
    train_subjects = train_subjects[:train_size]
    test_subjects = test_subjects[:test_size]

    imgs_list = pd.concat([imgs.loc[train_subjects],
                           imgs.loc[test_subjects]], keys=['train', 'test'])

    _run.info['dec_train_subjects'] = train_subjects
    _run.info['dec_test_subjects'] = test_subjects
    # noinspection PyUnboundLocalVariable
    return imgs_list, mask_img, root


@decomposition_ex.command
def unmask(smoothing_fwhm, raw_dir, n_jobs):
    imgs_list, mask_img, root = get_rest_data()
    create_raw_data(imgs_list,
                    root=root,
                    raw_dir=raw_dir,
                    masker_params=dict(smoothing_fwhm=smoothing_fwhm,
                                       detrend=True,
                                       standardize=True,
                                       mask_img=mask_img),
                    n_jobs=n_jobs)


class CapturedfMRIDictionaryScorer(rfMRIDictionaryScorer):
    @decomposition_ex.capture
    def __call__(self, masker, dict_fact, _run=None):
        rfMRIDictionaryScorer.__call__(self, masker, dict_fact)
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
                          raw_dir,
                          rest_data,
                          _run,
                          _seed,
                          train_subjects=None,
                          test_subjects=None,
                          observe=True,
                          ):
    observe = observe and not _run.unobserved

    memory = Memory(cachedir=get_cache_dirs()[0],
                    mmap_mode=None)
    print('Retrieve resting-state data')
    imgs_list, mask_img, _ = get_rest_data(
        train_subjects=train_subjects,
        test_subjects=test_subjects)
    print('Run dictionary learning')
    if raw_dir is not None:
        raw_dir = join(get_data_dirs()[0], 'raw', rest_data['source'],
                       str(smoothing_fwhm))
        mask, imgs_list = get_raw_data(imgs_list, raw_dir=raw_dir)
    else:
        mask = mask_img
    train_imgs, test_imgs = imgs_list.loc['train'], imgs_list.loc['test']

    if observe:
        callback = CapturedfMRIDictionaryScorer(test_imgs['filename'],
                                                test_confounds
                                                =test_imgs['confounds'])
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
    if observe:
        _run.info['dec_score'] = callback.score
        _run.info['dec_time'] = callback.time
        _run.info['dec_iter'] = callback.iter
        _run.info['dec_final_score'] = final_score
        print('Write decomposition artifacts')
        artifact_dir = mkdtemp()
        safe_to_filename(dict_fact.components_img_,
                         join(artifact_dir, 'dec_components.nii.gz'))
        _run.add_artifact(join(artifact_dir, 'dec_components.nii.gz'),
                          name='dec_components.nii.gz')

        safe_to_filename(dict_fact.mask_img_,
                         join(artifact_dir, 'dec_mask_img.nii.gz'))
        _run.add_artifact(join(artifact_dir, 'dec_mask_img.nii.gz'),
                          name='dec_mask_img.nii.gz')

        fig = plt.figure()
        display_maps(fig, dict_fact.components_img_)
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
    return dict_fact, final_score


@decomposition_ex.automain
def run_decomposition_ex(_run):
    compute_decomposition()
