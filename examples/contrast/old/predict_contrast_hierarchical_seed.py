import sys
from os import path
from os.path import join

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.predict_contrast_hierarchical\
    import predict_contrast_hierarchical

multi_predict_task = Experiment('multi_predict_contrast_hierarchical',
                                ingredients=[predict_contrast_hierarchical])
collection = multi_predict_task.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
multi_predict_task.observers.append(observer)


@multi_predict_task.config
def config():
    n_jobs = 36
    dropout_latent_list = [0., 0.5]
    dropout_input_list = [0., 0.25]
    latent_dim_list = [25, 50, 100, 200]
    shared_supervised_list = [False, True]
    task_prob_list = [0., 0.5, 1]
    alpha_list = [1e-4]
    n_seeds = 10
    verbose = 0
    seed = 10


def single_run(config_updates, _id, master_id):
    config_updates['seed'] = config_updates.pop('aseed', None)
    observer = MongoObserver.create(db_name='amensch', collection=collection)
    predict_contrast_hierarchical.observers = [observer]

    @predict_contrast_hierarchical.config
    def config():
        n_jobs = 1
        epochs = 50
        verbose = 0

    run = predict_contrast_hierarchical._create_run(config_updates=config_updates)
    run._id = _id
    run.info['master_id'] = master_id
    try:
        run()
    except:
        pass


@multi_predict_task.automain
def run(dropout_latent_list,
        dropout_input_list,
        latent_dim_list,
        shared_supervised_list,
        task_prob_list,
        n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    param_grid = ParameterGrid(
        {'datasets': [['archi', 'hcp', 'brainomics', 'la5c']],
         'shared_supervised': shared_supervised_list,
         'task_prob': task_prob_list,
         'dropout_latent': dropout_latent_list,
         'dropout_input': dropout_input_list,
        'latent_dim': latent_dim_list,
         # Hack to iterate over seed first'
         'aseed': seed_list})

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(param_grid))
