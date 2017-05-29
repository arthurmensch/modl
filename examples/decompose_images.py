# Author: Arthur Mensch
# License: BSD
import os
import time
from os.path import expanduser

import matplotlib.pyplot as plt
from modl.datasets.image import load_image
from modl.decomposition.image import ImageDictFact
from modl.feature_extraction.image import LazyCleanPatchExtractor
from modl.plotting.image import plot_patches
from sacred import Experiment
from sacred.observers import TinyDbObserver

decompose_ex = Experiment('decompose_images')
observer = TinyDbObserver.create(expanduser('~/runs'))
decompose_ex.observers.append(observer)


@decompose_ex.config
def config():
    batch_size = 400
    learning_rate = 0.92
    reduction = 10
    alpha = 0.08
    n_epochs = 10
    n_components = 50
    test_size = 4000
    max_patches = 10000
    patch_size = (16, 16)
    n_threads = 3
    verbose = 20
    method = 'gram'
    setting = 'dictionary learning'
    source = 'lisboa'
    gray = False
    scale = 1


class DictionaryScorer:
    def __init__(self, test_data):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        self.time = []
        self.score = []
        self.iter = []

    @decompose_ex.capture
    def __call__(self, dict_fact, _run):
        test_time = time.clock()
        score = dict_fact.score(self.test_data)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        self.time.append(this_time)
        self.score.append(score)
        self.iter.append(dict_fact.n_iter_)
        _run.info['score'] = self.score
        _run.info['time'] = self.time
        _run.info['iter'] = self.iter


@decompose_ex.automain
def decompose_run(batch_size,
                  learning_rate,
                  reduction,
                  n_components,
                  n_epochs,
                  patch_size,
                  test_size,
                  alpha,
                  setting,
                  n_threads,
                  verbose,
                  max_patches,
                  method,
                  source,
                  scale, gray,
                  _run,
                  _seed,
                  ):
    print('Loading data')
    image = load_image(source, scale=scale, gray=gray)
    print('Done')
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
    score = dict_fact.score(test_data)

    fig = plt.figure()
    patches = dict_fact.components_
    plot_patches(fig, patches)
    fig.suptitle('Dictionary raw')
    plt.savefig('raw.png')
    plt.close(fig)
    _run.add_artifact('raw.png')
    os.unlink('raw.png')

    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test objective value')
    plt.savefig('training_curve.png')
    plt.close(fig)
    _run.add_artifact('training_curve.png')
    os.unlink('training_curve.png')

    return score
