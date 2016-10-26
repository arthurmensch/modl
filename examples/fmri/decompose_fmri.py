# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import time
from os.path import join
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from data import load_data, data_ing, load_init, init_ing
from modl._utils.system import get_cache_dirs
from modl.plotting.fmri import display_maps
from modl.spca_fmri import SpcaFmri
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory

decompose_ex = Experiment('decompose_fmri',
                          ingredients=[data_ing, init_ing])
decompose_ex.observers.append(MongoObserver.create())


@data_ing.config
def config():
    dataset = 'adhd'
    raw = True
    n_subjects = 40


@init_ing.config
def config():
    source = None
    n_components = 70


@decompose_ex.config
def config():
    batch_size = 200
    learning_rate = 0.9
    offset = 0
    AB_agg = 'full'
    G_agg = 'full'
    Dx_agg = 'full'
    reduction = 10
    alpha = 1e-3
    l1_ratio = 1
    n_epochs = 3
    verbose = 15
    n_jobs = 3
    smoothing_fwhm = 6
    buffer_size = 1200
    temp_dir = '/tmp'
    subset_sampling = 'random'


class SpcaFmriScorer():
    @decompose_ex.capture
    def __init__(self, test_data, raw, _run):
        self.start_time = time.perf_counter()
        self.test_data = test_data
        self.test_time = 0
        self.raw = raw
        for info_key in ['score', 'time',
                         'iter', 'profiling',
                         'components',
                         'filename']:
            _run.info[info_key] = []
        self.call_count = 0

    @decompose_ex.capture
    def __call__(self, spca_fmri, _run):
        test_time = time.perf_counter()
        self.call_count += 1
        filename = 'record_%s.nii.gz' % spca_fmri.n_iter_

        if not self.call_count % 5:
            with TemporaryDirectory() as dir:
                filename = join(dir, filename)
                spca_fmri.components_.to_filename(filename)
                _run.add_artifact(filename)

        score = spca_fmri.score(self.test_data, raw=self.raw)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time

        test_time = time.perf_counter()

        _run.info['time'].append(this_time)
        _run.info['score'].append(score)
        _run.info['profiling'].append(spca_fmri.profiling_.tolist())
        _run.info['iter'].append(spca_fmri.n_iter_)
        # _run.info['components'].append(filename)

        self.test_time += time.perf_counter() - test_time


@decompose_ex.automain
def decompose_run(smoothing_fwhm,
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
                  buffer_size,
                  temp_dir,
                  subset_sampling,
                  _seed,
                  _run
                  ):
    train_data, test_data, mask, raw = load_data()
    print('seed: ', _seed)
    if _run.observers:
        cb = SpcaFmriScorer(test_data, raw=raw)
    else:
        cb = None

    n_components, init = load_init()

    if raw:
        memory = None
    else:
        memory = Memory(cachedir=get_cache_dirs()[0],
                        verbose=0)

    spca_fmri = SpcaFmri(smoothing_fwhm=smoothing_fwhm,
                         mask=mask,
                         memory=memory,
                         memory_level=2,
                         verbose=verbose,
                         n_epochs=n_epochs,
                         n_jobs=n_jobs,
                         random_state=_seed,
                         n_components=n_components,
                         dict_init=init,
                         learning_rate=learning_rate,
                         offset=offset,
                         batch_size=batch_size,
                         AB_agg=AB_agg,
                         G_agg=G_agg,
                         Dx_agg=Dx_agg,
                         reduction=reduction,
                         alpha=alpha,
                         l1_ratio=l1_ratio,
                         subset_sampling=subset_sampling,
                         buffer_size=buffer_size,
                         temp_dir=temp_dir,
                         callback=cb,
                         )
    spca_fmri.fit(train_data, raw=raw)

    try:
        with TemporaryDirectory() as dir:
            filename = join(dir, 'components.nii.gz')
            spca_fmri.components_.to_filename(filename)
            _run.add_artifact(filename)
    except KeyError:
        pass
    fig = display_maps(spca_fmri.components_)

    with TemporaryDirectory() as dir:
        filename = join(dir, 'components.png')
        plt.savefig(filename)
        _run.add_artifact(filename)
    plt.savefig('test.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(_run.info['time'], _run.info['score'])
    plt.savefig('score.png')
    plt.close(fig)
