# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import join, expanduser
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from modl.datasets.images import load_images
from modl.dict_fact import DictFact
from modl.dict_fact_slow import DictFactSlow
from modl.plotting.images import plot_patches
from sacred import Experiment
from sacred.ingredient import Ingredient

from modl.streaming.batcher import ImageBatcher

data_ing = Ingredient('data')
decompose_ex = Experiment('decompose_images',
                          ingredients=[data_ing])


@decompose_ex.config
def config():
    batch_size = 200
    learning_rate = 0.92
    BC_agg = 'async'
    G_agg = 'average'
    Dx_agg = 'average'
    reduction = 10
    code_alpha = 1e-1
    code_l1_ratio = 1
    comp_l1_ratio = 0
    n_epochs = 1
    n_components = 50
    code_pos = False
    comp_pos = False
    normalize = True
    center = True
    mask_sampling = 'random'
    test_size = 4000
    buffer_size = 5000
    max_patches = 100000
    patch_shape = (16, 16)


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
                  BC_agg, G_agg, Dx_agg,
                  code_pos,
                  comp_pos,
                  center,
                  normalize,
                  reduction,
                  code_alpha,
                  code_l1_ratio,
                  comp_l1_ratio,
                  n_components,
                  mask_sampling,
                  n_epochs,
                  patch_shape,
                  test_size,
                  buffer_size,
                  max_patches,
                  data,
                  _seed,
                  ):
    clean = data['source'] == 'aviris'
    image = load_data()
    width, height, n_channel = image.shape
    data_shape = (patch_shape[0], patch_shape[1], n_channel)
    batcher = ImageBatcher(patch_shape=patch_shape,
                           batch_size=test_size,
                           clean=clean,
                           random_state=_seed)
    batcher.prepare(image[:, :height // 2, :])
    test_data, _ = batcher.generate_single()
    if center:
        test_data -= np.mean(test_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
    if normalize:
        std = np.sqrt(np.sum(test_data ** 2, axis=(1, 2)))
        std[std == 0] = 1
        test_data /= std[:, np.newaxis, np.newaxis, :]
    test_data = test_data.reshape((test_data.shape[0], -1))
    n_features = test_data.shape[1]

    batcher = ImageBatcher(patch_shape=patch_shape,
                           batch_size=buffer_size,
                           max_samples=max_patches,
                           clean=clean,
                           random_state=_seed)
    batcher.prepare(image[:, height // 2:, :])
    n_samples = batcher.n_samples_

    cb = ImageScorer(test_data)
    cb = None
    dict_fact = DictFactSlow(
                         n_epochs=n_epochs,
                         random_state=_seed,
                         n_components=n_components,
                         comp_l1_ratio=comp_l1_ratio,
                         learning_rate=learning_rate,
                         comp_pos=comp_pos,
                         code_pos=code_pos,
                         batch_size=batch_size,
                         mask_sampling=mask_sampling,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         code_alpha=code_alpha,
                         code_l1_ratio=code_l1_ratio,
                         callback=cb,
                         verbose=1,
                         n_threads=1,
                         )
    first_batch = True
    for _ in range(n_epochs):
        for batch, indices in batcher.generate_once():
            if center:
                batch -= np.mean(batch, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
            if normalize:
                std = np.sqrt(np.sum(batch ** 2, axis=(1, 2)))
                std[std == 0] = 1
                batch /= std[:, np.newaxis, np.newaxis, :]
            batch = batch.reshape((batch.shape[0], -1))
            if first_batch:
                dict_fact.prepare(n_samples=n_samples, n_features=n_features, X=batch)
                first_batch = False
            dict_fact.partial_fit(batch, indices)

    # fig = plt.figure()
    # patches = dict_fact.components_.reshape((dict_fact.components_.shape[0],
    #                                          data_shape[0],
    #                                          data_shape[1], data_shape[2]))
    # plot_patches(fig, patches)
    # fig.suptitle('Dictionary components')
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(cb.iter, cb.score)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Test objective value')
    # plt.show()
