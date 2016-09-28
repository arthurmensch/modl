from copy import copy

from fmri_decompose import fmri_decompose, fmri_decompose_run
from sacred import Experiment
from sacred.initialize import create_run
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Parallel, delayed

fmri_compare = Experiment('fmri_compare', ingredients=[fmri_decompose])
observer = MongoObserver.create(url='mongo')
fmri_compare.observers.append(observer)


@fmri_decompose.config
def config():
    reduction = 3
    n_epochs = 2
    verbose = 4


@fmri_compare.config
def config():
    config_updates_list = []
    param_updates_list = [{'AB_agg': 'full'},
                          {'AB_agg': 'async'}]
    reductions = [5, 7]
    for param in param_updates_list:
        for reduction in reductions:
            config_updates_list.append(dict(reduction=reduction, **param))


# Cannot capture in joblib
def single_run(config_updates=None, _seed=0):
    config_updates['seed'] = _seed

    @fmri_decompose.capture
    def pre_run_hook(_run):
        _run.info['parent_id'] = fmri_compare.observers[0].run_entry['_id']
        _run.info['updated_params'] = config_updates

    single_observer = MongoObserver.create(url='mongo')
    fmri_decompose.pre_run_hooks = [pre_run_hook]
    fmri_decompose.observers = [single_observer]
    run = create_run(fmri_decompose, fmri_decompose_run.__name__,
                     config_updates)
    run()


@fmri_compare.automain
def fmri_compare_run(config_updates_list, _run, _seed):
    _run.info['runs'] = []
    Parallel(n_jobs=2,
             backend='multiprocessing')(
        delayed(single_run)(config_updates=config_updates,
                            _seed=_seed)
        for config_updates in config_updates_list)
