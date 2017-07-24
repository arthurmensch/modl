import os
import sys
from os import path
from os.path import join

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

from modl.utils.system import get_output_dir

# Add examples to known modules
sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))
from exps.exp_decompose_images import exp as single_exp

exp = Experiment('multi_decompose_images')
basedir = join(get_output_dir(), 'multi_decompose_images')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 15
    n_seeds = 1
    seed = 1


@single_exp.config
def config():
    batch_size = 200
    learning_rate = 0.92
    reduction = 10
    alpha = 0.1
    n_epochs = 40
    n_components = 256
    test_size = 4000
    max_patches = 100000
    patch_size = (16, 16)
    n_threads = 2
    verbose = 100
    method = 'gram'
    step_size = 0.1
    setting = 'NMF'
    source = 'aviris'
    gray = False
    scale = 1


def single_run(config_updates, rundir, _id):
    for i in range(3):
        try:
            run = single_exp._create_run(config_updates=config_updates)
            observer = FileStorageObserver.create(basedir=rundir)
            run._id = _id
            run.observers = [observer]
            run()
            break
        except TypeError:
            if i < 2:
                print("Run %i failed at start, retrying..." % _id)
            else:
                print("Giving up %i" % _id)
            continue


@exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    exps += [{'method': 'sgd',
              'step_size': step_size}
             for step_size in np.logspace(-3, 3, 7)]
    exps += [{'method': 'gram',
             'reduction': reduction}
            for reduction in [1, 4, 6, 8, 12, 24]]

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
