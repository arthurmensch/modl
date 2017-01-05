# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time

import matplotlib.pyplot as plt
from sacred import Experiment
from sacred import Ingredient
from sklearn.externals.joblib import Memory

from modl.datasets.fmri import load_rest_func, load_atlas_init
from modl.fmri import rfMRIDictFact
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs

data_ing = Ingredient('data')
init_ing = Ingredient('init')

decompose_ex = Experiment('decompose_fmri', ingredients=[data_ing, init_ing])

@init_ing.config
def config():
    n_components = 20
    source = None

@data_ing.config
def config():
    dataset = 'adhd'
    raw = False
    n_subjects = 40
    test_size = 1

@decompose_ex.config
def config():
    batch_size = 200
    learning_rate = 0.92
    offset = 0
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    verbose = 15
    n_jobs = 3
    smoothing_fwhm = 6
    buffer_size = 1200
    subset_sampling = 'random'



@data_ing.capture
def load_data(dataset, n_subjects, test_size, raw, _seed):
    return load_rest_func(dataset, n_subjects=n_subjects,
                          test_size=test_size,
                          raw=raw,
                          random_state=_seed)

@init_ing.capture
def load_init(source, n_components):
    return load_atlas_init(source, n_components=n_components)


class rfMRIDictionaryScorer:
    def __init__(self, test_data, raw):
        self.start_time = time.perf_counter()
        self.test_data = test_data
        self.test_time = 0
        self.raw = raw
        self.score = []
        self.iter = []
        self.time = []

    def __call__(self, dict_fact):
        test_time = time.perf_counter()
        score = dict_fact.score(self.test_data, raw=self.raw)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.iter.append(dict_fact.n_iter_)

@decompose_ex.automain
def decompose_run(smoothing_fwhm,
                  batch_size,
                  learning_rate,
                  verbose,
                  reduction,
                  alpha,
                  n_jobs,
                  n_epochs,
                  buffer_size,
                  data,
                  init,
                  _seed,
                  ):
    raw = data['raw']
    n_components = init['n_components']
    dict_init = load_init()
    train_data, test_data, mask = load_data()

    if raw:
        memory = None
    else:
        memory = Memory(cachedir=get_cache_dirs()[0],
                        verbose=2)

    cb = rfMRIDictionaryScorer(test_data, raw=raw)
    dict_fact = rfMRIDictFact(smoothing_fwhm=smoothing_fwhm,
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
                              buffer_size=buffer_size,
                              callback=cb,
                              )
    dict_fact.fit(train_data, raw=raw)

    dict_fact.components_.to_filename('components.nii.gz')
    fig = plt.figure()
    display_maps(fig, dict_fact.components_)
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score)
    plt.show()
