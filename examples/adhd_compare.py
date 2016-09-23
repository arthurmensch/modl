import json
import subprocess
from sklearn.externals.joblib import Parallel, delayed


def single_run(index, config_updates=None):
    json_name = 'test_%i.json' % index
    with open(json_name, 'w+') as f:
        json.dump(config_updates, f)
    subprocess.run(args=['python',  'adhd_decompose.py', 'with %s' % json_name])


config_updates_list = [{'AB_agg': 'full'}, {'AB_agg': 'async'}]

Parallel(n_jobs=2, backend='multiprocessing')(delayed(single_run)(index=i,
                                                                  config_updates=config_updates)
                   for i, config_updates in enumerate(config_updates_list))
