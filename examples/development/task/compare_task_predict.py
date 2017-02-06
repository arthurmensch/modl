import sys
from os import path
from os.path import expanduser

import numpy as np

from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))


def single_run(config_updates, _id):
    from examples.development.task.task_predict import prediction_ex, \
        task_data_ing, rest_data_ing, decomposition_ex

    @prediction_ex.config
    def config():
        standardize = True
        C = np.logspace(-1, 1, 10)
        n_jobs = 1
        verbose = 2
        seed = 2
        max_iter = 10000
        tol = 1e-7
        hierachical = False

    @task_data_ing.config
    def config():
        train_size = 750
        test_size = 10
        seed = 2

    @rest_data_ing.config
    def config():
        source = 'hcp'
        train_size = None  # Overriden
        test_size = 1
        seed = 2
        # train and test are overriden

    @decomposition_ex.config
    def config():
        batch_size = 100
        learning_rate = 0.92
        method = 'masked'
        reduction = 10
        alpha = 1e-3 # Overriden
        n_components = 40 # Overriden
        n_epochs = 1
        smoothing_fwhm = 4
        n_jobs = 1
        verbose = 15
        seed = 2

    observer = MongoObserver.create(db_name='amensch', collection='runs')

    observer_file = FileStorageObserver.create(expanduser('~/output/runs'))
    prediction_ex.observers.append(observer_file)

    prediction_ex.observers = [observer, observer_file]

    run = prediction_ex._create_run(config_updates=config_updates)
    run._id = _id
    run()


def first_grid_search():
    n_jobs = 18
    train_size_list = [50, 200, 778]

    n_components_list = [20, 50, 100]

    alpha_list = [1e-4, 1e-5]

    update_list = []
    for train_size in train_size_list:
        for n_components in n_components_list:
            for alpha in alpha_list:
                config_updates = {'task_data': {'train_size': 778},
                                  'rest_data': {'train_size': train_size},
                                  'decomposition':
                                      {'n_components': n_components,
                                       'alpha': alpha},
                                  }
                update_list.append(config_updates)

    client = pymongo.MongoClient()
    database = client['amensch']
    c = database.runs.find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs)(delayed(single_run)(config_updates, c + i)
                            for i, config_updates in enumerate(update_list))


def second_grid_search():
    pass


if __name__ == '__main__':
    first_grid_search()
