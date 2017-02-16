import sys
from os import path
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))



def single_run(config_updates, _id):
    from examples.components.decompose_rest import decompose_rest

    @decompose_rest.config
    def config():
        batch_size = 100
        learning_rate = 0.92
        method = 'gram'
        reduction = 12
        alpha = 1e-4  # Overriden
        n_components = 40  # Overriden
        n_epochs = 2
        smoothing_fwhm = 4
        n_jobs = 1
        verbose = 10
        seed = 2

        source = 'hcp'
        n_subjects = 788
        train_size = 787
        test_size = 1
        seed = 2

    observer = MongoObserver.create(db_name='amensch',
                                    collection='runs')

    decompose_rest.observers = [observer]

    run = decompose_rest._create_run(config_updates=config_updates)
    run._id = _id
    run()


@multi_decompose_rest.automain
def run():
    n_jobs = 6
    n_components_list = [16, 64, 256]

    alpha_list = [1e-4, 1e-5]

    update_list = []
    for n_components in n_components_list:
        for alpha in alpha_list:
            config_updates = {'n_components': n_components,
                               'alpha': alpha}
            update_list.append(config_updates)

    client = pymongo.MongoClient()
    database = client['amensch']
    c = database.runs.find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1

    Parallel(n_jobs=n_jobs)(delayed(single_run)(config_updates, c + i)
                            for i, config_updates in enumerate(update_list))
