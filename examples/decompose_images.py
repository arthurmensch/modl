# Author: Arthur Mensch
# License: BSD
import time

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.ingredient import Ingredient

from modl.datasets.images import load_images
from modl.dict_fact import DictFact
from modl.plotting.images import plot_patches
from modl.feature_extraction.images import ImageBatcher

from math import sqrt

data_ing = Ingredient('data')
decompose_ex = Experiment('decompose_images',
                          ingredients=[data_ing])


@decompose_ex.config
def config():
    batch_size = 200
    learning_rate = 0.92
    G_agg = 'average'
    Dx_agg = 'average'
    reduction = 10
    code_alpha = 0.8e-1
    code_l1_ratio = 1
    comp_l1_ratio = 0
    n_epochs = 2
    n_components = 200
    code_pos = False
    comp_pos = False
    normalize = True
    center = True
    test_size = 4000
    buffer_size = 5000
    max_patches = 50000
    patch_shape = (16, 16)
    n_threads = 3
    verbose = 10


@data_ing.config
def config():
    source = 'lisboa'
    gray = False
    scale = 1
    center = False
    normalize = False


@data_ing.capture
def load_data(source, scale, gray, center, normalize):
    return load_images(source, scale=scale,
                       gray=gray,
                       center=center,
                       normalize=normalize)


class ImageScorer():
    def __init__(self, test_data):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        self.time = []
        self.score = []
        self.iter = []

    def __call__(self, dict_fact):
        test_time = time.clock()

        score = dict_fact.score(self.test_data)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time

        test_time = time.clock()

        self.time.append(this_time)
        self.score.append(score)
        self.iter.append(dict_fact.n_iter_)

        self.test_time += time.clock() - test_time


@decompose_ex.automain
def decompose_run(batch_size,
                  learning_rate,
                  G_agg, Dx_agg,
                  code_pos,
                  comp_pos,
                  center,
                  normalize,
                  reduction,
                  code_alpha,
                  code_l1_ratio,
                  comp_l1_ratio,
                  n_components,
                  n_epochs,
                  patch_shape,
                  test_size,
                  buffer_size,
                  max_patches,
                  data,
                  n_threads,
                  verbose,
                  _seed,
                  ):
    clean = data['source'] == 'aviris'
    image = load_data()
    width, height, n_channel = image.shape
    data_shape = (patch_shape[0], patch_shape[1], n_channel)
    batcher = ImageBatcher(patch_shape=patch_shape,
                           batch_size=test_size,
                           clean=clean,
                           normalize=normalize,
                           center=center,
                           random_state=_seed)
    batcher.prepare(image[:, :height // 2, :])
    test_data, _ = batcher.generate_single()

    batcher = ImageBatcher(patch_shape=patch_shape,
                           batch_size=buffer_size,
                           max_samples=max_patches,
                           clean=clean,
                           normalize=normalize,
                           center=center,
                           random_state=_seed)
    batcher.prepare(image[:, height // 2:, :])
    cb = ImageScorer(test_data)
    dict_fact = DictFact(n_epochs=n_epochs,
                         random_state=_seed,
                         n_components=n_components,
                         comp_l1_ratio=comp_l1_ratio,
                         learning_rate=learning_rate,
                         comp_pos=comp_pos,
                         code_pos=code_pos,
                         batch_size=batch_size,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         code_alpha=code_alpha,
                         code_l1_ratio=code_l1_ratio,
                         callback=cb,
                         verbose=verbose,
                         n_threads=n_threads,
                         )
    dict_fact.connect(batcher)

    fig = plt.figure()
    patches = dict_fact.components_.reshape((dict_fact.components_.shape[0],
                                             data_shape[0],
                                             data_shape[1], data_shape[2]))
    plot_patches(fig, patches)
    fig.suptitle('Dictionary components')
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test objective value')
    plt.show()
