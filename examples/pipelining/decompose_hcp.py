# Author: Arthur Mensch
# License: BSD
import os
from os.path import join

import matplotlib.pyplot as plt
from sacred.observers import FileStorageObserver

from modl.input_data.fmri.fixes import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sklearn.model_selection import train_test_split

from modl.input_data.fmri.rest import get_raw_rest_data
from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_output_dir

from sacred import Experiment

exp = Experiment('decompose_rest')
base_artifact_dir = join(get_output_dir(), 'components', 'hcp')
exp.observers.append(FileStorageObserver.create(basedir=base_artifact_dir))

@exp.config
def config():
    n_components = 20
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 1
    verbose = 15
    n_jobs = 5
    smoothing_fwhm = 4
    optimizer = 'variational'
    positive = True


@exp.automain
def compute_components(n_components,
                       batch_size,
                       learning_rate,
                       method,
                       positive,
                       reduction,
                       alpha,
                       optimizer,
                       n_epochs,
                       verbose,
                       n_jobs,
                       _run):
    artifact_dir = join(base_artifact_dir, str(_run._id), 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    raw_res_dir = join(get_output_dir(), 'unmasked', 'hcp')
    masker, data = get_raw_rest_data(raw_res_dir)

    train_imgs, test_imgs = train_test_split(data, test_size=1, random_state=0)
    train_imgs = train_imgs['filename'].values
    test_imgs = test_imgs['filename'].values

    cb = rfMRIDictionaryScorer(test_imgs)
    dict_fact = fMRIDictFact(method=method,
                             mask=masker,
                             verbose=verbose,
                             optimizer=optimizer,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=1,
                             n_components=n_components,
                             positive=positive,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=cb,
                             )
    dict_fact.fit(train_imgs)
    dict_fact.components_img_.to_filename(join(artifact_dir, 'components.nii.gz'))
    fig = plt.figure()
    display_maps(fig, dict_fact.components_img_)
    plt.savefig(join(artifact_dir, 'components.png'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    plt.savefig(join(artifact_dir, 'score.png'))
