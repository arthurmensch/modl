# Author: Arthur Mensch
# License: BSD
import os
import time

from modl.utils.nifti import monkey_patch_nifti_image
monkey_patch_nifti_image()

from modl import fMRIDictFact
from modl.datasets.fmri import load_atlas_init
from modl.plotting.fmri import display_maps

from modl.utils.system import get_cache_dirs
from sklearn.externals.joblib import Memory


from sklearn.model_selection import train_test_split
from modl.datasets import fetch_adhd
from sacred import Experiment

import matplotlib.pyplot as plt

experiment = Experiment('fMRI decomposition')


@experiment.config
def config():
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    verbose = 15
    n_jobs = 3
    smoothing_fwhm = 6
    dataset = 'adhd'
    raw = False
    n_subjects = 40
    test_size = 2
    n_components = 20
    source = 'smith'
    warmup = False
    seed = 0


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
        test_imgs, test_confounds = zip(*self.test_data)
        score = dict_fact.score(test_imgs, confounds=test_confounds)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.iter.append(dict_fact.n_iter_)
        _run.info['score'] = self.score
        _run.info['time'] = self.time
        _run.info['iter'] = self.iter


@experiment.automain
def main(n_components,
         batch_size,
         learning_rate,
         method,
         reduction,
         alpha,
         n_epochs,
         verbose,
         n_jobs,
         warmup,
         smoothing_fwhm,
         source,
         n_subjects,
         test_size,
         _run, _seed):
    dict_init = load_atlas_init(source, n_components=n_components)

    dataset = fetch_adhd(n_subjects=n_subjects)
    data = list(zip(dataset.func, dataset.confounds))
    train_data, test_data = train_test_split(data, test_size=test_size,
                                             random_state=0)
    train_imgs, train_confounds = zip(*train_data)
    mask = dataset.mask
    memory = Memory(cachedir=get_cache_dirs()[0],
                    verbose=2)

    cb = rfMRIDictionaryScorer(test_data)
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
                             dict_init=dict_init,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=cb,
                             warmup=warmup,
                             )
    dict_fact.fit(train_imgs, train_confounds)
    test_imgs, test_confounds = zip(*test_data)
    score = dict_fact.score(test_imgs, test_confounds)

    dict_fact.components_.to_filename('components.nii.gz')
    _run.add_artifact('components.nii.gz')
    os.unlink('components.nii.gz')
    fig = plt.figure()
    display_maps(fig, dict_fact.components_)
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    plt.savefig('learning_curve.png')
    _run.add_artifact('learning_curve.png')
    os.unlink('learning_curve.png')
    plt.show()

    return score