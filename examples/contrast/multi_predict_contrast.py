import sys
from os import path

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

import numpy as np
from sklearn.utils import check_random_state

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.predict_contrast import predict_contrast

multi_predict_task = Experiment('multi_predict_task',
                                ingredients=[predict_contrast])
observer = MongoObserver.create(db_name='amensch', collection='runs')
multi_predict_task.observers.append(observer)


@multi_predict_task.config
def config():
    n_jobs = 1
    loss_list = ['l1', 'l2']
    n_seeds = 5


@predict_contrast.config
def config():
    standardize = True
    Cs = np.logspace(-2, 2, 10).tolist()
    n_jobs = 30
    verbose = 2
    max_iter = 1000
    tol = 1e-5
    alpha = 1e-4
    identity = False
    refit = False
    n_components_list = [16, 64, 256]
    test_size = 0.1
    train_size = None
    n_subjects = 788
    loss = 'l1'  # Overriden


def single_run(config_updates, _id):
    observer = MongoObserver.create(db_name='amensch',
                                    collection='runs')
    predict_contrast.observers = [observer]

    run = predict_contrast._create_run(config_updates=config_updates)
    run._id = _id
    run()


@multi_predict_task.automain
def run(loss_list, n_seeds, n_jobs, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    update_list = []
    for seed in seed_list:
        for loss in loss_list:
            config_updates = {'loss': loss,
                              'seed': seed}
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
