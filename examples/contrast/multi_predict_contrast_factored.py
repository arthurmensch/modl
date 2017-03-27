import sys
from itertools import chain
from os import path

from modl.datasets import get_data_dirs
from os.path import join
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.predict_contrast import predict_contrast

multi_predict_task = Experiment('multi_predict_contrast',
                                ingredients=[predict_contrast])
collection = multi_predict_task.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
multi_predict_task.observers.append(observer)


@multi_predict_task.config
def config():
    n_jobs = 30
    dropout_list = [0, 0.3, 0.6, 0.9]
    latent_dim_list = [30, 100, 200]
    alpha_list = [1e-4]
    beta_list = [0]
    activation_list = ['linear']
    n_seeds = 10

def single_run(config_updates, _id, master_id):
    observer = MongoObserver.create(db_name='amensch',
                                    collection=collection)
    predict_contrast.observers = [observer]

    @predict_contrast.config
    def config():
        n_jobs = 1
        from_loadings = True
        projected = True
        factored = True
        n_subjects = 788
        max_iter = 100
        loadings_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                            'reduced')
        verbose = 0

    run = predict_contrast._create_run(config_updates=config_updates)
    run._id = _id
    run.info['multi_predict_contrast_id'] = master_id
    run()


@multi_predict_task.automain
def run(dropout_list,
        alpha_list,
        beta_list,
        activation_list,
        latent_dim_list,
        n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    param_grid = ParameterGrid(
        {'datasets': [['archi'], ['hcp'], ['archi', 'hcp']],
         'dropout': dropout_list,
         'latent_dim': latent_dim_list,
         'alpha': alpha_list,
         'beta': beta_list,
         'activation': activation_list,
         'seed': seed_list})

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(param_grid))
