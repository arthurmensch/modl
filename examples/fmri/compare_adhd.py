"""Does not work with NamedConfig for unknown reason"""
from functools import partial
from multiprocessing import pool, get_context

from copy import copy

from decompose_fmri import decompose_ex, decompose_run
from data import data_ing, init_ing
from joblib import Parallel
from joblib import delayed
from sacred import Experiment
from sacred.observers import MongoObserver

compare_ex = Experiment('compare_adhd', ingredients=[decompose_ex])
observer = MongoObserver.create()
compare_ex.observers.append(observer)


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
    reduction = 3
    batch_size = 50
    learning_rate = 0.9
    offset = 0
    alpha = 1e-3
    l1_ratio = 1
    n_epochs = 30
    verbose = 150
    n_jobs = 2
    buffer_size = 1200
    temp_dir = '/tmp'


@compare_ex.config
def config():
    n_jobs = 10
    param_updates_list = [
        # Reduction on BCD only
        {'G_agg': 'average', 'Dx_agg': 'average', 'AB_agg': 'async'},
        # TSP with full parameter update
        {'G_agg': 'full', 'Dx_agg': 'full', 'AB_agg': 'async'},
        # TSP full Gram with full parameter update
        {'G_agg': 'full', 'Dx_agg': 'average', 'AB_agg': 'async'},
        # ICML with full parameter update
        # {'G_agg': 'masked', 'Dx_agg': 'masked', 'AB_agg': 'full'},
        # ICML
        # {'G_agg': 'masked', 'Dx_agg': 'masked', 'AB_agg': 'async'}]
    ]
    config_updates_list = []
    reductions = [6, 12, 24]
    for param in param_updates_list:
        for reduction in reductions:
            config_updates_list.append(dict(reduction=reduction,
                                            **param))
    # Reference
    config_updates_list.append({'G_agg': 'full',
                                'Dx_agg': 'full', 'AB_agg': 'async',
                                'reduction': 1})
    del param_updates_list, reductions  # , param


# Cannot capture in joblib
def single_run(our_config_updates, config, parent_id):
    @decompose_ex.capture
    def pre_run_hook(_run):
        _run.info['parent_id'] = parent_id
        _run.info['updated_params'] = our_config_updates

    single_observer = MongoObserver.create()
    decompose_ex.pre_run_hooks = [pre_run_hook]
    decompose_ex.observers = [single_observer]

    config_updates = config['decompose_fmri'].copy()
    config_updates['seed'] = config['seed']
    for key, value in our_config_updates.items():
        config_updates[key] = value
    for ingredient in decompose_ex.ingredients:
        path = ingredient.path
        config_updates[path] = {}
        ingredient_config_update = config[path]
        config_updates[path]['seed'] = config['seed']
        for key, value in ingredient_config_update.items():
            config_updates[path][key] = value

    decompose_ex.run(config_updates=config_updates)


@compare_ex.automain
def compare_run(config_updates_list, n_jobs, _run):
    parent_id = _run.observers[0].run_entry['_id']
    config = _run.config
    Parallel(n_jobs=n_jobs,
             backend='multiprocessing')(
        delayed(single_run)(our_config_updates=our_config_updates,
                            parent_id=parent_id, config=config)
        for our_config_updates in config_updates_list)


