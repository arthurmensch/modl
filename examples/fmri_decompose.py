# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import expanduser, join
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from modl.datasets.fmri import load_data, fmri_data_ingredient, data_path_ingredient
from modl.plotting.fmri import display_maps
from modl.spca_fmri import SpcaFmri
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory

fmri_decompose = Experiment('fmri_decompose',
                            ingredients=[fmri_data_ingredient])
fmri_decompose.observers.append(MongoObserver.create())

# noinspection PyUnusedLocal
@fmri_decompose.config
def config():
    batch_size = 50
    learning_rate = 0.8
    offset = 0
    AB_agg = 'full'
    G_agg = 'full'
    Dx_agg = 'average'
    reduction = 3
    alpha = 1e-4
    l1_ratio = 0.5
    smoothing_fwhm = 6
    n_jobs = 1
    n_epochs = 1
    verbose = 5

@fmri_data_ingredient.config
def config():
    dataset = 'adhd'

@data_path_ingredient.config
def config():
    raw = False


class SpcaFmriScorer():
    @fmri_decompose.capture
    def __init__(self, test_data, _run):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        for info_key in ['score', 'time',
                         'iter', 'profiling',
                         'components',
                         'filename']:
            _run.info[info_key] = []

    @fmri_decompose.capture
    def __call__(self, spca_fmri, _run):
        test_time = time.clock()

        filename = 'record_%s.nii.gz' % spca_fmri.n_iter_

        with TemporaryDirectory() as dir:
            filename = join(dir, filename)
            spca_fmri.components_.to_filename(filename)
            _run.add_artifact(filename)

        score = spca_fmri.score(self.test_data)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time

        test_time = time.clock()

        _run.info['time'].append(this_time)
        _run.info['score'].append(score)
        _run.info['profiling'].append(spca_fmri.profiling_.tolist())
        _run.info['iter'].append(spca_fmri.n_iter_)
        _run.info['components'].append(filename)

        self.test_time += time.clock() - test_time


@fmri_decompose.automain
def fmri_decompose_run(smoothing_fwhm,
                       batch_size,
                       learning_rate,
                       offset,
                       verbose,
                       AB_agg, G_agg, Dx_agg,
                       reduction,
                       alpha,
                       l1_ratio,
                       n_jobs,
                       n_epochs,
                       _seed,
                       _run
                       ):
    train_data, test_data, mask = load_data()
    print('seed: ', _seed)
    if _run.observers:
        cb = SpcaFmriScorer(test_data)
    else:
        cb = None

    spca_fmri = SpcaFmri(smoothing_fwhm=smoothing_fwhm,
                         mask=mask,
                         memory=Memory(cachedir=expanduser("~/nilearn_cache"),
                                       verbose=0),
                         memory_level=2,
                         verbose=verbose,
                         n_epochs=n_epochs,
                         n_jobs=n_jobs,
                         random_state=_seed,
                         learning_rate=learning_rate,
                         offset=offset,
                         batch_size=batch_size,
                         AB_agg=AB_agg,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         alpha=alpha,
                         l1_ratio=l1_ratio,
                         callback=cb,
                         )
    spca_fmri.fit(train_data)

    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.nii.gz')
        spca_fmri.components_.to_filename(filename)
        _run.add_artifact(filename)

    fig = display_maps(spca_fmri.components_)
    plt.show(fig)
