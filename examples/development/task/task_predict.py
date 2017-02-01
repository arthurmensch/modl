import os
import copy
import time
import warnings
from os.path import expanduser, join

import shutil
from matplotlib.cbook import MatplotlibDeprecationWarning
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import pandas as pd
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

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore',
                        category=UserWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore',
                        category=MatplotlibDeprecationWarning,
                        module='matplotlib')
warnings.filterwarnings('ignore',
                        category=UserWarning,
                        module='numpy')

experiment = Experiment('task_predict')
observer = FileStorageObserver.create(expanduser('~/runs'))
experiment.observers.append(observer)
observer = MongoObserver.create(db_name='amensch', collection='runs')
experiment.observers.append(observer)

@experiment.config
def config():
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    verbose = 15
    n_jobs = 20
    smoothing_fwhm = 6
    n_components = 40
    seed = 20
    n_subjects = 100
    test_size = 2
    rest_source = 'adhd'


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
        n_subjects, reduction, smoothing_fwhm, method, verbose, rest_source,
        test_size,
        _run, _seed):
    # Fold preparation

    # Rest data
    if rest_source == 'hcp':
        data = fetch_hcp(n_subjects=n_subjects)
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
    components, masker, unsupervised_test_score, cb = memory.cache(get_components,
                                                 ignore=['verbose', 'n_jobs'])(
        train_rest_data,
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
    _id = _run._id
    artifact_dir = join(expanduser('~/runs/artifacts'), str(_id))
    os.makedirs(artifact_dir)
    # Avoid trashing cache
    components_copy = copy.deepcopy(components)
    components_copy.to_filename(join(artifact_dir, 'components.nii.gz'))
    _run.add_artifact(join(artifact_dir, 'components.nii.gz'),
                      name='components.nii.gz')
    fig = plt.figure()
    display_maps(fig, components)
    plt.savefig(join(artifact_dir, 'components.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'components.png'),
                      name='components.png')
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    plt.savefig(join(artifact_dir, 'learning_curve.png'))
    plt.close(fig)
    _run.add_artifact(join(artifact_dir, 'learning_curve.png'),
                      name='learning_curve.png')
    _run.info['unsupervised_test_score'] = unsupervised_test_score
    # End reporting

    # Task data
    if rest_source == 'adhd':
        data = fetch_hcp(n_subjects=n_subjects)
    task_data = data.task
    task_mask = data.mask
    subjects = task_data.index.levels[0].values
    if rest_source == 'adhd':
        train_subjects, test_subjects = \
            train_test_split(subjects, random_state=_seed, test_size=test_size)
        train_subjects = train_subjects.tolist()
        test_subjects = test_subjects.tolist()
    _run.info['train_subjects'] = train_subjects
    _run.info['test_subjects'] = test_subjects

    # Selection of contrasts
    interesting_con = ['FACES', 'SHAPES', '']
    task_data = task_data.loc[(slice(None), slice(None), interesting_con), :]

    z_maps = task_data.loc[:, 'filename']
    z_map_labels = z_maps.index.get_level_values(2).values
    z_maps = z_maps.values
    label_encoder = LabelEncoder()
    z_map_labels = label_encoder.fit_transform(z_map_labels)
    task_data = task_data.assign(label=z_map_labels)
    # Supervised validation
    loadings = memory.cache(get_encodings, ignore=['n_jobs'])(z_maps,
                                                              components,
                                                              mask=task_mask,
                                                              n_jobs=n_jobs)
    # Ugly pandas <-> numpy
    loadings = [loading for loading in loadings]
    task_data = task_data.assign(loadings=loadings)
    X = task_data.loc[:, 'loadings'].values
    X_train = task_data.loc[train_subjects, 'loadings'].values
    y_train = task_data.loc[train_subjects, 'label'].values
    X_train = np.vstack(X_train)
    X = np.vstack(X)
    lr_estimator = memory.cache(logistic_regression)(X_train, y_train)

    y_pred = lr_estimator.predict(X)
    predicted_labels = label_encoder.inverse_transform(y_pred)
    task_data = task_data.assign(predicted_labels=predicted_labels,
                                 true_labels=task_data.index.
                                 get_level_values(2).values)

    res_df = task_data.loc[:, ['predicted_labels', 'true_labels']]
    res_df.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'),
                      name='prediction.csv')

    train_score = np.sum(res_df.loc[train_subjects,
                                    'predicted_labels']
                         == res_df.loc[train_subjects,
                                       'true_labels']) / task_data.loc[train_subjects].shape[0]

    test_score = np.sum(res_df.loc[test_subjects, 'predicted_labels']
                        == res_df.loc[
                            test_subjects, 'true_labels']) / task_data.loc[test_subjects].shape[0]
    _run.info['train_score'] = train_score
    _run.info['test_score'] = test_score

    shutil.rmtree(artifact_dir)

    return train_score, test_score
