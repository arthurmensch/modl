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

decompose_ex = Experiment('decompose_images',
                          ingredients=[data_ing])
decompose_ex.observers.append(MongoObserver.create(url='172.18.0.16'))


@decompose_ex.config
def config():
    batch_size = 200
    learning_rate = 0.9
    offset = 0
    AB_agg = 'full'
    G_agg = 'full'
    Dx_agg = 'full'
    reduction = 1
    alpha = 1e-1
    l1_ratio = 0
    pen_l1_ratio = 1
    n_epochs = 10
    verbose = 50
    verbose_offset = 100
    n_components = 100
    n_threads = 3
    subset_sampling = 'random'
    dict_reduction = 'follow'
    temp_dir = expanduser('~/tmp')
    buffer_size = 5000
    test_size = 2000
    max_patches = None
    patch_shape = (16, 16)


@data_ing.config
def config():
    source = 'aviris'
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
        self.call_counter = 0

    @decompose_ex.capture
    def __call__(self, dict_fact, _run):
        test_time = time.clock()

        filename = 'record_%s.npy' % dict_fact.n_iter_
        if not self.call_counter % 10:
            with TemporaryDirectory() as dir:
                filename = join(dir, filename)
                np.save(filename, dict_fact.components_)
                _run.add_artifact(filename)
        self.call_counter += 1

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
                  verbose_offset,
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
    batcher.prepare(image[:width // 5, :, :])
    test_data, _ = batcher.generate_one()
    batcher = Batcher(patch_shape=patch_shape,
                      batch_size=buffer_size,
                      max_samples=max_patches,
                      clean=data['source'] == 'aviris',
                      random_state=_seed)
    batcher.prepare(image[width // 5:, :, :])
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
                         verbose_offset=verbose_offset,
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
    patches = dict_fact.components_.reshape((dict_fact.components_.shape[0],
                                             * _run.info['data_shape']))
    plot_patches(fig, patches)
    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.png')
        plt.savefig(filename)
        _run.add_artifact(filename)
    fig, ax = plt.subplots(1, 1)
    ax.plot(_run.info['time'], _run.info['score'])
    plt.show()
