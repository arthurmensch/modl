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
from sklearn.utils import check_random_state, shuffle

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.predict_contrast import predict_contrast_exp

predict_contrast_multi_exp = Experiment('predict_contrast_multinomial',
                                        ingredients=[predict_contrast_exp])
collection = predict_contrast_multi_exp.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
predict_contrast_multi_exp.observers.append(observer)


@predict_contrast_multi_exp.config
def config():
    n_jobs = 10
    n_seeds = 10
    seed = 2


def single_run(config_updates, _id, master_id):
    observer = MongoObserver.create(db_name='amensch', collection=collection)
    predict_contrast_exp.observers = [observer]

    @predict_contrast_exp.config
    def config():
        n_jobs = 1
        epochs = 300
        steps_per_epoch = 200
        dropout_input = 0.25
        dropout_latent = 0.5
        source = 'hcp_rs'
        depth_prob = [0, 1., 0]
        shared_supervised = False
        batch_size = 256
        alpha = 1e-5
        validation = False
        mix_batch = False
        verbose = 1
        train_size = dict(hcp=None, archi=30, la5c=50, brainomics=30,
                          camcan=100,
                          human_voice=None)
        unmask_dir = '/storage/data/contrast'
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
    for seed in seed_list:
        for dataset in ['la5c']:
            multinomial = [{'datasets': [dataset],
                            'geometric_reduction': False,
                            'latent_dim': None,
                            'dropout_input': 0.,
                            'dropout_latent': 0.,
                            'epochs': 50,
                            'steps_per_epoch': None,
                            'batch_size': 300,
                            'alpha': alpha,
                            # 'lr': lr,
                            'optimizer': 'sgd',
                            'seed': seed}
                           # for source in ['hcp_rs_concat', 'hcp_rs']
                           for alpha in np.logspace(-4, 1, 6)
                           ]
            geometric_reduction = [{'datasets': [dataset],
                                    'geometric_reduction': True,
                                    'latent_dim': None,
                                    'dropout_input': 0.,
                                    'dropout_latent': 0.,
                                    'source': source,
                                    'alpha': alpha,
                                    'optimizer': 'adam',
                                    'seed': seed} for seed in seed_list
                                   for alpha in [1e-5, 1e-4]
                                   for source in ['hcp_rs', 'hcp_rs_concat']]
            latent_dropout = [{'datasets': [dataset],
                               'geometric_reduction': True,
                               'latent_dim': 50,
                               'dropout_input': 0.25,
                               'dropout_latent': 0.5,
                               'optimizer': 'adam',
                               'seed': seed} for seed in seed_list]
            transfer = [{'datasets': [dataset, 'hcp'],
                         'geometric_reduction': True,
                         'latent_dim': 50,
                         'dropout_input': 0.25,
                         'dropout_latent': 0.5,
                         'optimizer': 'adam',
                         'seed': seed} for seed in seed_list]
            exps += multinomial
            # exps += geometric_reduction
            # exps += latent_dropout
            # exps += transfer

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1
    exps = shuffle(exps)
    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(exps))
