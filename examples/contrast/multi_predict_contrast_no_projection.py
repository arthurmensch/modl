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

multi_predict_task = Experiment('multi_predict_contrast_no_projection',
                                ingredients=[predict_contrast])
collection = multi_predict_task.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
multi_predict_task.observers.append(observer)



@multi_predict_task.config
def config():
    n_jobs = 6 
    penalty_list = ['trace']
    alpha_list = np.logspace(-6, -1, 3).tolist()
    n_seeds = 1


def single_run(config_updates, _id, master_id):
    observer = MongoObserver.create(db_name='amensch',
                                    collection=collection)

    @predict_contrast.config
    def config():
        n_jobs = 1
        from_loadings = True
        projection = False
        factored = False
        loadings_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                            'reduced')
        verbose = 2
        max_iter = 50

    predict_contrast.observers = [observer]

    run = predict_contrast._create_run(config_updates=config_updates)
    run._id = _id
    run.info['multi_predict_contrast_id'] = master_id
    run()


@multi_predict_task.automain
def run(penalty_list,
        alpha_list,
        n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    param_grid = ParameterGrid({'datasets': [['archi', 'hcp'], 'archi'],
                                'penalty': penalty_list,
                                'alpha': alpha_list,
                                'seed': seed_list})
    # Robust labelling of experim   ents
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(param_grid))
