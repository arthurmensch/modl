import sys
from os import path

from sacred.observers import MongoObserver

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.development.task.task_predict import prediction_ex

observer = MongoObserver.create(db_name='amensch', collection='runs')

n_subjects_list = [50, 100, 200, 500, 788]

n_components_list = [20, 50, 100]

alpha_list = [1e-4, 1e-4]

update_list = []
for n_subjects in n_subjects_list:
    for n_components in n_components_list:
        for alpha in alpha_list:
            config_updates = {'task_data.n_subjects': n_subjects,
                              'decomposition.n_components': n_components,
                              'decomposition.alpha': alpha
                              }
            prediction_ex.run()
