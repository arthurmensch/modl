import sys
from os import path

from sacred.observers import MongoObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))


def single_run(config_updates):
    from examples.development.task.task_predict import prediction_ex
    observer = MongoObserver.create(db_name='amensch', collection='runs')
    prediction_ex.observers = [observer]

    # config_updates = config['decompose_fmri'].copy()
    # config_updates['seed'] = config['seed']
    # for key, value in our_config_updates.items():
    #     config_updates[key] = value
    # for ingredient in decompose_ex.ingredients:
    #     path = ingredient.path
    #     config_updates[path] = {}
    #     ingredient_config_update = config[path]
    #     config_updates[path]['seed'] = config['seed']
    #     for key, value in ingredient_config_update.items():
    #         config_updates[path][key] = value

    prediction_ex.run(config_updates=config_updates)

def main():
    n_jobs = 2
    n_subjects_list = [50, 100, 200, 500, 788]
    n_subjects_list = [50]

    n_components_list = [20, 50, 100]
    n_components_list = [20]

    alpha_list = [1e-4, 1e-4]

    update_list = []
    for n_subjects in n_subjects_list:
        for n_components in n_components_list:
            for alpha in alpha_list:
                config_updates = {'task_data.n_subjects': n_subjects,
                                  'decomposition.n_components': n_components,
                                  'decomposition.alpha': alpha
                                  }
                update_list.append(config_updates)

    Parallel(n_jobs=n_jobs)(delayed(single_run)(config_updates)
                            for config_updates in update_list)

if __name__ == '__main__':
    main()