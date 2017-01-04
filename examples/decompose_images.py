# Author: Arthur Mensch
# License: BSD
import time

import matplotlib.pyplot as plt
from modl.feature_extraction.image import LazyCleanPatchExtractor
from modl.image import ImageDictFact
from sacred import Experiment
from sacred.ingredient import Ingredient

from modl.datasets.images import load_images
from modl.plotting.images import plot_patches

data_ing = Ingredient('data')
decompose_ex = Experiment('decompose_images', ingredients=[data_ing])


@decompose_ex.config
def config():
    batch_size = 400
    learning_rate = 0.92
    reduction = 10
    alpha = 0.1
    n_epochs = 1
    n_components = 200
    test_size = 1000
    max_patches = 100000
    patch_size = (16, 16)
    n_threads = 1
    verbose = 20
    method = 'masked'
    setting = 'dictionary learning'


@data_ing.config
def config():
    source = 'lisboa'
    gray = False
    scale = 1


@data_ing.capture
def load_data(source, scale, gray):
    return load_images(source, scale=scale,
                       gray=gray)


class DictionaryScorer:
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
        self.time.append(this_time)
        self.score.append(score)
        self.iter.append(dict_fact.n_iter_)


@decompose_ex.automain
def decompose_run(batch_size,
                  learning_rate,
                  reduction,
                  n_components,
                  n_epochs,
                  patch_size,
                  test_size,
                  method,
                  alpha,
                  setting,
                  n_threads,
                  verbose,
                  max_patches,
                  _seed,
                  ):
    image = load_data()
    width, height, n_channel = image.shape
    patch_extractor = LazyCleanPatchExtractor(patch_size=patch_size,
                                              max_patches=test_size,
                                              random_state=_seed)
    test_data = patch_extractor.transform(image[:, :height // 2, :])
    cb = DictionaryScorer(test_data)
    dict_fact = ImageDictFact(method=method,
                              setting=setting,
                              alpha=alpha,
                              n_epochs=n_epochs,
                              random_state=_seed,
                              n_components=n_components,
                              learning_rate=learning_rate,
                              max_patches=max_patches,
                              batch_size=batch_size,
                              patch_size=patch_size,
                              reduction=reduction,
                              callback=cb,
                              verbose=verbose,
                              n_threads=n_threads,
                              )
    dict_fact.fit(image[:, height // 2:, :])

    fig = plt.figure()

    patches = dict_fact.components_
    plot_patches(fig, patches)
    fig.suptitle('Dictionary components')
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test objective value')
    plt.show()
