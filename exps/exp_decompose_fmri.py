# Author: Arthur Mensch
# License: BSD
import json
import os
from os.path import join

import matplotlib.pyplot as plt
from hcp_builder.dataset import fetch_hcp_mask
from sacred.observers import FileStorageObserver

from modl.datasets import get_data_dirs
from modl.input_data.fmri.fixes import monkey_patch_nifti_image
from modl.input_data.fmri.unmask import MultiRawMasker

monkey_patch_nifti_image()

from sklearn.model_selection import train_test_split

from modl.input_data.fmri.rest import get_raw_rest_data
from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_output_dir

from sacred import Experiment

import pandas as pd

exp = Experiment('decomppose_fmri')
base_artifact_dir = join(get_output_dir(), 'decompose_fmri')
exp.observers.append(FileStorageObserver.create(basedir=base_artifact_dir))

@exp.config
def config():
    n_components = 70
    batch_size = 100
    learning_rate = 0.92
    method = 'dictionary only'
    reduction = 1
    alpha = 1e-4
    n_epochs = 100
    verbose = 30
    n_jobs = 5
    step_size = 1e-5
    source = 'adhd_4'
    seed = 1


@exp.automain
def compute_components(n_components,
                       batch_size,
                       learning_rate,
                       method,
                       reduction,
                       alpha,
                       step_size,
                       n_jobs,
                       n_epochs,
                       verbose,
                       source,
                       _run):
    basedir = join(_run.observers[0].basedir, str(_run._id))
    artifact_dir = join(basedir, 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if source == 'hcp':
        # Hack to recover data from TSP
        train_size = None
        smoothing_fwhm = 3
        test_size = 2
        data_dir = get_data_dirs()[0]
        mask = fetch_hcp_mask()
        masker = MultiRawMasker(mask_img=mask, smoothing_fwhm=smoothing_fwhm,
                                detrend=True, standardize=True)
        mapping = json.load(
            open(join(data_dir, 'HCP_unmasked/mapping.json'), 'r'))
        data = sorted(list(mapping.values()))
        data = list(map(lambda x: join(data_dir, x), data))
        data = pd.DataFrame(data, columns=['filename'])
    else:
        smoothing_fwhm = 6
        train_size = 4
        test_size = 4
        raw_res_dir = join(get_output_dir(), 'unmasked', source)
        try:
            masker, data = get_raw_rest_data(raw_res_dir)
        except ValueError:  # On local machine:
            raw_res_dir = join(get_output_dir(), 'unmask', source)
            masker, data = get_raw_rest_data(raw_res_dir)



    train_imgs, test_imgs = train_test_split(data, test_size=test_size,
                                             random_state=0,
                                             train_size=train_size)
    train_imgs = train_imgs['filename'].values
    test_imgs = test_imgs['filename'].values

    cb = rfMRIDictionaryScorer(test_imgs, info=_run.info)
    dict_fact = fMRIDictFact(method=method,
                             mask=masker,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=1,
                             n_components=n_components,
                             smoothing_fwhm=smoothing_fwhm,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             step_size=step_size,
                             alpha=alpha,
                             callback=cb,
                             )
    dict_fact.fit(train_imgs)
    dict_fact.components_img_.to_filename(join(artifact_dir, 'components.nii.gz'))
    fig = plt.figure()
    display_maps(fig, dict_fact.components_img_)
    plt.savefig(join(artifact_dir, 'components.png'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.cpu_time, cb.score, marker='o')
    _run.info['time'] = cb.cpu_time
    _run.info['score'] = cb.score
    _run.info['iter'] = cb.iter
    plt.savefig(join(artifact_dir, 'score.png'))
