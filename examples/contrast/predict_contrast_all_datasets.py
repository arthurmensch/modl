import sys
from os import path
from os.path import join

import numpy as np
from modl.datasets import get_data_dirs
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.predict_contrast import predict_contrast_exp

predict_contrast_multi_exp = Experiment('predict_contrast_all_datasets',
                                        ingredients=[predict_contrast_exp])
collection = predict_contrast_multi_exp.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
predict_contrast_multi_exp.observers.append(observer)


@predict_contrast_multi_exp.config
def config():
    n_jobs = 24
    n_seeds = 10
    seed = 2


def single_run(config_updates, _id, master_id):
    observer = MongoObserver.create(db_name='amensch', collection=collection)
    predict_contrast_exp.observers = [observer]

    @predict_contrast_exp.config
    def config():
        n_jobs = 1
        epochs = 400
        artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                            'all_datasets')
        steps_per_epoch = 200
        dropout_input = 0.25
        dropout_latent = 0.5
        source = 'hcp_rs_concat'
        depth_prob = [0, 1., 0]
        shared_supervised = False
        batch_size = 100
        alpha = 1e-5
        validation = False
        mix_batch = False
        verbose = 2
        train_size = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                          camcan=None,
                          human_voice=None)

    run = predict_contrast_exp._create_run(
        config_updates=config_updates)
    run._id = _id
    run.info['master_id'] = master_id
    try:
        run()
    except:
        pass


@predict_contrast_multi_exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    transfer = [{'datasets': ['archi', 'brainomics', 'camcan', 'hcp'],
                 'geometric_reduction': True,
                 'latent_dim': 50,
                 'dropout_input': 0.25,
                 'dropout_latent': 0.5,
                 'optimizer': 'adam',
                 'seed': seed} for seed in seed_list]
    exps += transfer

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(exps))
