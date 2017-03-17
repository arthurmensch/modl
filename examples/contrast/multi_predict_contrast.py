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
    penalty_list = ['l1', 'l2']
    dropout_list = [True, False]
    latent_dim_list = [30, 100, 200]
    alpha_list = np.logspace(-4, -1, 4)
    activation_list = ['linear', 'relu']
    n_seeds = 5


@predict_contrast.config
def config():
    n_subjects = 30
    n_jobs = 1


def single_run(config_updates, _id):
    observer = MongoObserver.create(db_name='amensch',
                                    collection='runs')
    predict_contrast.observers = [observer]

    run = predict_contrast._create_run(config_updates=config_updates)
    run._id = _id
    run()


@multi_predict_task.automain
def run(penalty_list,
        dropout_list,
        alpha_list,
        activation_list,
        latent_dim_list,
        n_seeds, n_jobs, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    update_list = []
    for seed in seed_list:
        for penalty in penalty_list:
            for dropout in dropout_list:
                for latent_dim in latent_dim_list:
                    for alpha in alpha_list:
                        for activation in activation_list:
                            config_updates = {'penalty': penalty,
                                              'dropout': dropout,
                                              'latent_dim': latent_dim,
                                              'alpha': alpha,
                                              'activation': activation,
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
