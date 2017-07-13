# Author: Arthur Mensch
# License: BSD
import os
from os.path import join

import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver

from modl.datasets.image import load_image
from modl.decomposition.image import ImageDictFact, DictionaryScorer
from modl.feature_extraction.image import LazyCleanPatchExtractor
from modl.plotting.image import plot_patches
from modl.utils.system import get_output_dir

exp = Experiment('decompose_images')
base_artifact_dir = join(get_output_dir(), 'decompose_images')
exp.observers.append(FileStorageObserver.create(basedir=base_artifact_dir))


@exp.config
def config():
    batch_size = 400
    learning_rate = 0.92
    reduction = 10
    alpha = 0.08
    n_epochs = 12
    n_components = 100
    test_size = 4000
    max_patches = 10000
    patch_size = (32, 32)
    n_threads = 2
    verbose = 10
    method = 'gram'
    step_size = 0.1
    setting = 'dictionary learning'
    optimizer = 'variational'
    source = 'lisboa'
    gray = False
    scale = 1


@exp.automain
def decompose_images(batch_size,
                     learning_rate,
                     reduction,
                     alpha,
                     n_epochs,
                     n_components,
                     test_size,
                     max_patches,
                     patch_size,
                     n_threads,
                     verbose,
                     optimizer,
                     method,
                     step_size,
                     setting,
                     source,
                     gray,
                     scale,
                     _run):
    basedir = join(_run.observers[0].basedir, str(_run._id))
    artifact_dir = join(basedir, 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    print('Loading data')
    image = load_image(source, scale=scale, gray=gray)
    print('Done')
    width, height, n_channel = image.shape
    patch_extractor = LazyCleanPatchExtractor(patch_size=patch_size,
                                              max_patches=test_size,
                                              random_state=1)
    test_data = patch_extractor.transform(image[:, :height // 2, :])
    cb = DictionaryScorer(test_data, info=_run.info)
    dict_fact = ImageDictFact(method=method,
                              setting=setting,
                              alpha=alpha,
                              step_size=step_size,
                              n_epochs=n_epochs,
                              random_state=1,
                              n_components=n_components,
                              learning_rate=learning_rate,
                              max_patches=max_patches,
                              optimizer=optimizer,
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
    fig.suptitle('Dictionary')
    fig.savefig(join(artifact_dir, 'dictionary.png'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test objective value')
    fig.savefig(join(artifact_dir, 'score.png'))
