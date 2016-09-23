# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import expanduser, join
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
from modl.spca_fmri import SpcaFmri
from nilearn.image import index_img
from nilearn.plotting import plot_prob_atlas
from nilearn.plotting import plot_stat_map
from sacred import Experiment
from sacred.observers import MongoObserver

from modl.datasets.fmri import load_data, fmri_data_ingredient

ex = Experiment('fMRI_decompose', ingredients=[fmri_data_ingredient])
ex.observers.append(MongoObserver.create())

# noinspection PyUnusedLocal
@ex.config
def config():
    batch_size = 50
    learning_rate = 1
    offset = 0
    AB_agg = 'full'
    G_agg = 'full'
    Dx_agg = 'average'
    reduction = 3
    alpha = 1e-4
    l1_ratio = 0.5
    smoothing_fwhm = 6


class SpcaFmriScorer():
    @ex.capture
    def __init__(self, test_data):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        for info_key in ['score', 'time',
                         'iter', 'profiling',
                         'components',
                         'filename']:
            ex.info[info_key] = []

    @ex.capture
    def __call__(self, spca_fmri):
        test_time = time.clock()
        ex.info['score'].append(spca_fmri.score(self.test_data))
        ex.info['profiling'].append(spca_fmri.profiling_.tolist())
        ex.info['iter'].append(spca_fmri.n_iter_)

        filename = 'record_%s.nii.gz' % spca_fmri.n_iter_
        ex.info['components'].append(filename)

        with TemporaryDirectory() as dir:
            filename = join(dir, filename)
            spca_fmri.components_.to_filename(filename)
            ex.add_artifact(filename)

        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        ex.info['time'].append(this_time)


def display_maps(components, index=0):
    fig, axes = plt.subplots(2, 1)
    plot_prob_atlas(components, view_type="filled_contours",
                    axes=axes[0])
    plot_stat_map(index_img(components, index),
                  axes=axes[1],
                  colorbar=False,
                  threshold=0)
    return fig


@ex.automain
def run(smoothing_fwhm, batch_size, learning_rate,
        offset,
        AB_agg, G_agg, Dx_agg,
        reduction,
        alpha,
        l1_ratio,
        _seed
        ):
    train_data, test_data, mask = load_data()

    if ex.current_run.observers:
        cb = SpcaFmriScorer(test_data)
    else:
        cb = None

    spca_fmri = SpcaFmri(smoothing_fwhm=smoothing_fwhm,
                         mask=mask,
                         memory=Memory(cachedir=expanduser("~/nilearn_cache"),
                                       verbose=0),
                         memory_level=2,
                         verbose=5,
                         n_epochs=1,
                         n_jobs=1,
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
        ex.add_artifact(filename)

    fig = display_maps(spca_fmri.components_)
    plt.show(fig)
