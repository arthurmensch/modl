"""Does not work with NamedConfig for unknown reason"""

from copy import copy

from decompose_images import decompose_ex, decompose_run
from data import data_ing, patch_ing
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Parallel, delayed

compare_ex = Experiment('compare_hyperspectral', ingredients=[decompose_ex])
observer = MongoObserver.create()
compare_ex.observers.append(observer)


@decompose_ex.config
def config():
    batch_size = 100
    learning_rate = 0.9
    offset = 0
    AB_agg = 'full'
    G_agg = 'full'
    Dx_agg = 'full'
    reduction = 1
    alpha = 0.6
    l1_ratio = 0
    pen_l1_ratio = 0.9
    n_jobs = 1
    n_epochs = 20
    verbose = 50
    n_components = 100
    n_threads = 1


@data_ing.config
def config():
    source = 'aviris'
    gray = False
    scale = 1


@patch_ing.config
def config():
    patch_size = (16, 16)
    max_patches = 2000
    test_size = 2000
    in_memory = False
    normalize_per_channel = True


@compare_ex.config
def config():
    n_jobs = 1
    param_updates_list = [
        # Reduction on BCD only
        {'G_agg': 'full', 'Dx_agg': 'full', 'AB_agg': 'full'},
        # TSP
        {'G_agg': 'full', 'Dx_agg': 'average', 'AB_agg': 'async'},
        # TSP with full parameter update
        {'G_agg': 'full', 'Dx_agg': 'average', 'AB_agg': 'full'},
        # ICML with full parameter update
        {'G_agg': 'masked', 'Dx_agg': 'masked', 'AB_agg': 'full'},
        # ICML
        {'G_agg': 'masked', 'Dx_agg': 'masked', 'AB_agg': 'async'}]
    config_updates_list = []
    reductions = [2, 4, 8, 12]
    for param in param_updates_list:
        for reduction in reductions:
            config_updates_list.append(dict(reduction=reduction,
                                            **param))
    # Reference
    config_updates_list.append({'G_agg': 'full',
                               'Dx_agg': 'full', 'AB_agg': 'full',
                               'reduction': 1})
    del param_updates_list, reductions, param

# Cannot capture in joblib
def single_run(our_config_updates=None):
    @decompose_ex.capture
    def pre_run_hook(_run):
        _run.info['parent_id'] = compare_ex.observers[0].run_entry['_id']
        _run.info['updated_params'] = our_config_updates

    single_observer = MongoObserver.create()
    decompose_ex.pre_run_hooks = [pre_run_hook]
    decompose_ex.observers = [single_observer]

    config_updates = compare_ex.current_run.config['decompose_images'].copy()
    config_updates['seed'] = compare_ex.current_run.config['seed']
    for key, value in our_config_updates.items():
        config_updates[key] = value
    for ingredient in decompose_ex.ingredients:
        path = ingredient.path
        config_updates[path] = {}
        ingredient_config_update = compare_ex.current_run.config[path]
        config_updates[path]['seed'] = compare_ex.current_run.config['seed']
        for key, value in ingredient_config_update.items():
            config_updates[path][key] = value

    decompose_ex.run(config_updates=config_updates)


@compare_ex.automain
def compare_run(config_updates_list, n_jobs):
    Parallel(n_jobs=n_jobs,
             backend='multiprocessing')(
        delayed(single_run)(our_config_updates=our_config_updates)
        for our_config_updates in config_updates_list)
