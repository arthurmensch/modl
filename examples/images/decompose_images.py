# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import join, expanduser
from tempfile import TemporaryDirectory

import itertools
import matplotlib.pyplot as plt
import numpy as np
from data import load_data, data_ing
from modl.datasets.images import Batcher
from modl.dict_fact import DictFact
from modl.plotting.images import plot_patches
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import check_random_state

decompose_ex = Experiment('decompose_images',
                          ingredients=[data_ing])
decompose_ex.observers.append(MongoObserver.create())


@decompose_ex.config
def config():
    batch_size = 100
    learning_rate = 0.9
    offset = 0
    AB_agg = 'masked'
    G_agg = 'full'
    Dx_agg = 'masked'
    reduction = 10
    alpha = 1e-1
    l1_ratio = 0
    pen_l1_ratio = 0.9
    n_epochs = 2000
    verbose = 50
    n_components = 100
    n_threads = 3
    subset_sampling = 'random'
    dict_reduction = 'follow'
    temp_dir = None
    buffer_size = 10000
    test_size = 4000
    max_patches = 10000
    patch_shape = (8, 8)


@data_ing.config
def config():
    source = 'lisboa'
    gray = False
    scale = 1


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

        filename = 'record_%s.npy' % dict_fact.n_iter_

        with TemporaryDirectory() as dir:
            filename = join(dir, filename)
            np.save(filename, dict_fact.components_)
            _run.add_artifact(filename)

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
                  dict_reduction,
                  n_epochs,
                  temp_dir,
                  patch_shape,
                  test_size,
                  buffer_size,
                  max_patches,
                  data,
                  _seed,
                  _run
                  ):
    image = load_data()
    width, height, n_channel = image.shape
    batcher = Batcher(patch_shape=patch_shape,
                      batch_size=test_size,
                      clean=data['source'] == 'aviris',
                      random_state=_seed)
    batcher.prepare(image[:height // 2, :])
    test_data, _ = batcher.generate_one()
    batcher = Batcher(patch_shape=patch_shape,
                      batch_size=buffer_size,
                      max_samples=max_patches,
                      clean=data['source'] == 'aviris',
                      random_state=_seed)
    batcher.prepare(image[height // 2:, :])
    n_samples = batcher.n_samples_
    _run.info['data_shape'] = (test_data.shape[1],
                               test_data.shape[2],
                               test_data.shape[3])
    test_data = test_data.reshape((test_data.shape[0], -1))
    test_data -= np.mean(test_data, axis=1)[:, np.newaxis]
    std = np.sqrt(np.sum(test_data ** 2, axis=1))
    std[std == 0] = 1
    test_data /= std[:, np.newaxis]

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
                         dict_reduction=dict_reduction,
                         temp_dir=temp_dir,
                         AB_agg=AB_agg,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         alpha=alpha,
                         l1_ratio=l1_ratio,
                         lasso_tol=1e-2,
                         callback=cb,
                         buffer_size=buffer_size,
                         n_samples=n_samples
                         )
    for batch, indices in batcher.generate(n_epochs=n_epochs):
        batch = batch.reshape((batch.shape[0], -1))
        batch -= np.mean(batch, axis=1)[:, np.newaxis]
        std = np.sqrt(np.sum(batch ** 2, axis=1))
        std[std == 0] = 1
        batch /= std[:, np.newaxis]
        dict_fact.partial_fit(batch, indices, check_input=False)

    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.npy')
        np.save(filename, dict_fact.components_)
        _run.add_artifact(filename)

    fig = plt.figure()
    plot_patches(fig, dict_fact.components_, _run.info['data_shape'])
    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.png')
        plt.savefig(filename)
        _run.add_artifact(filename)
    fig, ax = plt.subplots(1, 1)
    ax.plot(_run.info['time'], _run.info['score'])
    plt.show()
