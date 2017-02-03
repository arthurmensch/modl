import os
import copy
import time
from os.path import expanduser, join
from tempfile import mkdtemp

import shutil
from sacred.observers import TinyDbObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from modl.datasets import fetch_adhd
from modl.plotting.fmri import display_maps
from modl.utils.nifti import monkey_patch_nifti_image

monkey_patch_nifti_image()

import numpy as np
from sacred import Experiment
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl import fMRIDictFact
from modl.datasets.hcp import fetch_hcp
from modl.utils.system import get_cache_dirs

experiment = Experiment('hierachical_decomposition')
observer = TinyDbObserver.create(expanduser('~/runs'))
experiment.observers.append(observer)


class rfMRIDictionaryScorer:
    def __init__(self, test_data):
        self.start_time = time.perf_counter()
        self.test_data = test_data
        self.test_time = 0
        self.score = []
        self.iter = []
        self.time = []

    @experiment.capture
    def __call__(self, dict_fact, _run):
        test_time = time.perf_counter()
        score = dict_fact.score(self.test_data)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.iter.append(dict_fact.n_iter_)
        _run.info['score'] = self.score
        _run.info['time'] = self.time
        _run.info['iter'] = self.iter


@experiment.config
def config():
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 5
    verbose = 15
    n_jobs = 3
    smoothing_fwhm = 6
    n_components_list = [20, 40, 80, 120, 160]
    seed = 20
    n_subjects = 500
    test_size = 4
    rest_source = 'adhd'


def get_components(data, *,
                   alpha=1,
                   batch_size=20,
                   learning_rate=1,
                   mask=None,
                   n_components=20,
                   n_epochs=1,
                   n_jobs=1,
                   reduction=1,
                   smoothing_fwhm=4,
                   method='masked',
                   random_state=None,
                   test_data=None,
                   verbose=0):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=10)
    if test_data is not None:
        cb = rfMRIDictionaryScorer(test_data)
    else:
        cb = None
    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             mask=mask,
                             memory=memory,
                             method=method,
                             memory_level=2,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=random_state,
                             n_components=n_components,
                             dict_init=None,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=cb,
                             )
    dict_fact.fit(data)
    score = dict_fact.score(test_data)
    return dict_fact.components_, dict_fact.masker_, score, cb


def get_encodings(data, components, *, mask=None, n_jobs=1, verbose=0):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    dict_fact = fMRIDictFact(smoothing_fwhm=0,
                             mask=mask,
                             detrend=False,
                             standardize=False,
                             memory_level=2,
                             memory=memory,
                             n_jobs=n_jobs,
                             dict_init=components,
                             verbose=verbose - 1,
                             ).fit()
    encodings = np.concatenate(dict_fact.transform(data))
    return encodings


def logistic_regression(X_train, y_train):
    lr = LogisticRegression(multi_class='multinomial',
                            solver='sag', max_iter=1000, verbose=2)
    lr.fit(X_train, y_train)
    return lr


@experiment.automain
def run(alpha, batch_size, learning_rate, n_components_list, n_epochs, n_jobs,
        reduction, smoothing_fwhm, method, verbose, rest_source, _run, _seed):
    # Fold preparation

    # Rest data
    if rest_source == 'hcp':
        data = fetch_hcp(n_subjects=10)
        rest_data = data.rest
        rest_mask = data.mask
        subjects = rest_data.index.get_level_values(0).values
        train_subjects, test_subjects = \
            train_test_split(subjects, random_state=_seed, test_size=2)
        train_rest_data = rest_data.loc[train_subjects.tolist(), 'filename']
        test_rest_data = rest_data.loc[test_subjects.tolist(), 'filename']

    elif rest_source == 'adhd':
        adhd_data = fetch_adhd(n_subjects=40)
        rest_data = adhd_data.func
        rest_mask = adhd_data.mask
        train_rest_data, test_rest_data = train_test_split(rest_data,
                                                           random_state=_seed,
                                                           test_size=2)
    else:
        raise ValueError('Wrong resting-state source')

    memory = Memory(cachedir=get_cache_dirs()[0], verbose=verbose)

    res = Parallel(n_jobs=n_jobs)(delayed(memory.cache(get_components))(
        train_rest_data,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask=rest_mask,
        n_components=n_components,
        n_epochs=n_epochs,
        n_jobs=1,
        reduction=reduction,
        smoothing_fwhm=smoothing_fwhm,
        method=method,
        test_data=test_rest_data,
        verbose=verbose) for n_components in n_components_list)
    # Reporting
    _run.info['score'] = {}
    _run.info['components'] = {}
    _run.info['components_png'] = {}
    _run.info['learning_curve'] = {}
    _id = _run._id
    artifact_dir = join(expanduser('~/runs/artifacts'), _id)
    os.makedirs(artifact_dir)
    for i, ((components, masker, score, cb),
            n_components) in enumerate(zip(res, n_components_list)):
        components_name = join(artifact_dir, 'components_%i.nii.gz' % i)
        components_path = join(artifact_dir, components_name)
        components.to_filename(components_path)
        _run.add_artifact(components_path,
                          name=components_name)
        fig = plt.figure()
        display_maps(fig, components)
        components_png_name = 'components_%i.png' % i
        components_png_path = join(artifact_dir, components_png_name)
        plt.savefig(components_png_path)
        _run.add_artifact(components_png_path, name=components_png_name)
        fig, ax = plt.subplots(1, 1)
        ax.plot(cb.time, cb.score, marker='o')
        lc_name = 'learning_curve_%i.png' % i
        lc_path = join(artifact_dir, 'learning_curve_%i.png' % i)
        plt.savefig(lc_path)
        _run.add_artifact(lc_path, name=lc_name)
        _run.info['score'][n_components] = score
        _run.info['components'][n_components] = components_name
        _run.info['components_png'][n_components] = components_png_path
        _run.info['learning_curve'][n_components] = lc_name

    return score
