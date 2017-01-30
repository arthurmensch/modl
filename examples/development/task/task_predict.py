import os
import copy
import time
from os.path import expanduser

from sacred.observers import TinyDbObserver
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

experiment = Experiment('task_predict')
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
    n_epochs = 3
    verbose = 15
    n_jobs = 2
    smoothing_fwhm = 6
    n_components = 40
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
                             warmup=True,
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
def run(alpha, batch_size, learning_rate, n_components, n_epochs, n_jobs,
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

    # Unsupervised training
    components, masker, score, cb = memory.cache(get_components)(train_rest_data,
                                             alpha=alpha,
                                             batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             mask=rest_mask,
                                             n_components=n_components,
                                             n_epochs=n_epochs,
                                             n_jobs=n_jobs,
                                             reduction=reduction,
                                             smoothing_fwhm=smoothing_fwhm,
                                             method=method,
                                             test_data=test_rest_data,
                                             verbose=verbose)
    # Reporting for unsupervised training
    components_copy = copy.deepcopy(components)
    components_copy.to_filename('components.nii.gz')
    _run.add_artifact('components.nii.gz')
    fig = plt.figure()
    display_maps(fig, components)
    plt.savefig('components.png')
    _run.add_artifact('components.png')
    # os.unlink('components.nii.gz')
    # os.unlink('components.png')
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    plt.savefig('learning_curve.png')
    _run.add_artifact('learning_curve.png')
    # os.unlink('learning_curve.png')
    _run.info['unsupervised_score'] = score
    # End reporting

    # Task data
    if rest_source == 'adhd':
        data = fetch_hcp(n_subjects=10)
    task_data = data.task
    task_mask = data.mask
    subjects = task_data.index.get_level_values(0).values
    if rest_source == 'adhd':
        train_subjects, test_subjects = \
            train_test_split(subjects, random_state=_seed, test_size=2)
    z_maps = task_data.loc[:, 'filename']
    z_map_labels = task_data.index.get_level_values(2).values
    z_map_labels = LabelEncoder().fit_transform(z_map_labels)

    # Supervised validation
    encoded_z_maps = memory.cache(get_encodings)(z_maps, components,
                                                 mask=task_mask,
                                                 n_jobs=n_jobs)
    z_maps.loc[:, 'encoded'] = encoded_z_maps
    X_train, y_train = encoded_z_maps[train_subjects, 'encoded'], \
                       z_map_labels[train_subjects]
    X_test, y_test = encoded_z_maps[test_subjects, 'encoded'], z_map_labels[test_subjects]

    lr_estimator = memory.cache(logistic_regression)(X_train, y_train)
    y_pred = lr_estimator.predict(X_test)
    score = np.sum(y_pred == y_test) / y_test.shape[0]

    return score
