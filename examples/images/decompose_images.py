# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import join
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from data import data_ing, patch_ing, make_patches
from joblib import Memory
from modl._utils.system import get_cache_dirs
from modl.dict_fact import DictFact
from modl.plotting.images import plot_patches
from sacred import Experiment
from sacred.observers import MongoObserver

decompose_ex = Experiment('decompose_images',
                          ingredients=[patch_ing])
decompose_ex.observers.append(MongoObserver.create())


@decompose_ex.config
def config():
    batch_size = 100
    learning_rate = 0.9
    offset = 0
    AB_agg = 'full'
    G_agg = 'average'
    Dx_agg = 'average'
    reduction = 10
    alpha = 0.1
    l1_ratio = 0
    pen_l1_ratio = 0.9
    n_jobs = 1
    n_epochs = 2
    verbose = 30
    n_components = 100
    n_threads = 3
    subset_sampling = 'random'
    temp_dir = '/tmp'


@data_ing.config
def config():
    source = 'aviris'
    gray = False
    scale = 1
    in_memory = False


@patch_ing.config
def config():
    patch_size = (8, 8)
    max_patches = 100000
    test_size = 2000
    normalize_per_channel = True
    pickle = True


class ImageScorer():
    @decompose_ex.capture
    def __init__(self, test_data, _run):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        for info_key in ['score', 'time',
                         'iter', 'profiling',
                         'components',
                         'filename']:
            _run.info[info_key] = []

    @decompose_ex.capture
    def __call__(self, dict_fact, _run):
        test_time = time.clock()

        print(dict_fact.feature_counter_)

        filename = 'record_%s.npy' % dict_fact.n_iter_

        # with TemporaryDirectory() as dir:
        #     filename = join(dir, filename)
        #     np.save(filename, dict_fact.components_)
        #     _run.add_artifact(filename)

        score = dict_fact.score(self.test_data)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time

        test_time = time.clock()

        _run.info['time'].append(this_time)
        _run.info['score'].append(score)
        _run.info['profiling'].append(dict_fact.profiling_.tolist())
        _run.info['iter'].append(dict_fact.n_iter_)
        _run.info['components'].append(filename)

        self.test_time += time.clock() - test_time


@decompose_ex.automain
def decompose_run(batch_size,
                  learning_rate,
                  offset,
                  verbose,
                  AB_agg, G_agg, Dx_agg,
                  reduction,
                  alpha,
                  l1_ratio,
                  pen_l1_ratio,
                  n_components,
                  n_threads,
                  subset_sampling,
                  n_epochs,
                  temp_dir,
                  _seed,
                  _run
                  ):
    train_data, test_data = make_patches()
    if _run.observers:
        cb = ImageScorer(test_data)
    else:
        cb = None

    dict_fact = DictFact(verbose=verbose,
                         n_epochs=n_epochs,
                         random_state=_seed,
                         n_components=n_components,
                         n_threads=n_threads,
                         pen_l1_ratio=pen_l1_ratio,
                         learning_rate=learning_rate,
                         offset=offset,
                         batch_size=batch_size,
                         subset_sampling=subset_sampling,
                         temp_dir=temp_dir,
                         AB_agg=AB_agg,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         alpha=alpha,
                         l1_ratio=l1_ratio,
                         callback=cb,
                         )
    dict_fact.fit(train_data)

    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.npy')
        np.save(filename, dict_fact.components_)
        _run.add_artifact(filename)

    fig = plot_patches(dict_fact.components_, _run.info['data_shape'])
    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.png')
        plt.savefig(filename)
        _run.add_artifact(filename)
    fig, ax = plt.subplots(1, 1)
    ax.plot(_run.info['time'], _run.info['score'])
    plt.show()

