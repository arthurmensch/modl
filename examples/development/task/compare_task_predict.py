import sys
from os import path

from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))


def single_run(config_updates, _id):
    from examples.development.task.task_predict import prediction_ex
    observer = MongoObserver.create(db_name='amensch', collection='runs')
    prediction_ex.observers = [observer]

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
