import sys
from os import path

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.components.decompose_rest import decompose_rest

multi_decompose_rest = Experiment('multi_decompose_rest',
                                  ingredients=[decompose_rest])
observer = MongoObserver.create(db_name='amensch', collection='runs')
multi_decompose_rest.observers.append(observer)

@multi_decompose_rest.config
def config():
    n_jobs = 1
    n_components_list = [256]
    alpha_list = [1e-4]


@decompose_rest.config
def config():
    batch_size = 100
    learning_rate = 0.92
    method = 'gram'
    reduction = 12
    alpha = 1e-4  # Overriden
    n_components = 40  # Overriden
    n_epochs = 4
    smoothing_fwhm = 6
    n_jobs = 4
    verbose = 10
    seed = 2

    source = 'adhd'
    n_subjects = 40
    train_size = 39
    test_size = 1


def single_run(config_updates, _id):
    observer = MongoObserver.create(db_name='amensch',
                                    collection='runs')
    decompose_rest.observers = [observer]

    run = decompose_rest._create_run(config_updates=config_updates)
    run._id = _id
    run()


@multi_decompose_rest.automain
def run(n_components_list, alpha_list, n_jobs):
    update_list = []
    for n_components in n_components_list:
        for alpha in alpha_list:
            config_updates = {'n_components': n_components,
                               'alpha': alpha}
            update_list.append(config_updates)

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database.runs.find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i)
                            for i, config_updates in enumerate(update_list))
