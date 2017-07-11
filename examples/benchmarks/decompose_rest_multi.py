import os
import sys
from os import path
from os.path import join

import numpy as np
from modl.utils.system import get_output_dir
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

# Add examples to known modules
sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))
from examples.benchmarks.decompose_rest import exp as single_exp

exp = Experiment('decompose_rest_multi')
basedir = join(get_output_dir(), 'decompose_rest_multi')
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
    n_components = 70
    batch_size = 200
    learning_rate = 0.92
    method = 'gram'
    reduction = 12
    alpha = 1e-4
    n_epochs = 30
    verbose = 100
    n_jobs = 2
    optimizer = 'variational'
    step_size = 1e-5
    source = 'adhd_4'
    seed = 1


def single_run(config_updates, rundir, _id):
    run = single_exp._create_run(config_updates=config_updates)
    observer = FileStorageObserver.create(basedir=rundir)
    run._id = _id
    run.observers = [observer]
    try:
        run()
    except:
        print('Run %i failed' % _id)


@exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    exps += [{'optimizer': 'sgd',
              'step_size': step_size}
             for step_size in np.logspace(-7, 0, 15)]
    exps += [{'optimizer': 'variational',
              'reduction': reduction}
             for reduction in [1, 4, 6, 8, 12, 24]]

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
