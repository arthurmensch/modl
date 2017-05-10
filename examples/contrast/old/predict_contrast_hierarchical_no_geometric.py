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
    n_jobs = 10
    alpha_list = np.logspace(-2, 1, 10).tolist()
    n_seeds = 10
    verbose = 0
    seed = 2


def single_run(config_updates, _id, master_id):
    config_updates['seed'] = config_updates.pop('aseed', None)
    observer = MongoObserver.create(db_name='amensch', collection=collection)
    predict_contrast_hierarchical.observers = [observer]

    @predict_contrast_hierarchical.config
    def config():
        datasets = ['archi']
        validation = False
        geometric_reduction = False
        alpha = 10
        latent_dim = None
        activation = 'linear'
        dropout_input = 0.
        dropout_latent = 0.
        batch_size = 300
        per_dataset_std = False
        joint_training = True
        optimizer = 'sgd'
        epochs = 15
        depth_weight = [0., 1., 0.]
        n_jobs = 2
        verbose = 2
        seed = 10
        shared_supervised = False
        mix_batch = False
        steps_per_epoch = None
        _seed = 0

    run = predict_contrast_hierarchical._create_run(config_updates=config_updates)
    run._id = _id
    run.info['master_id'] = master_id
    try:
        run()
    except:
        pass


@multi_predict_task.automain
def run(alpha_list,
        n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    param_grid = ParameterGrid(
        {'alpha': alpha_list,
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
