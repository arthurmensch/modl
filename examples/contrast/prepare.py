import sys
from os import path
from os.path import join

from sklearn.externals.joblib import Parallel, delayed

from modl.datasets import get_data_dirs

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.unmask.unmask_contrast import unmask_contrast
from examples.contrast.reduce_contrast import reduce_contrast


def launch_dataset(dataset):
    # run = unmask_contrast._create_run(config_updates=dict(dataset=dataset))
    run()
    run = reduce_contrast._create_run(
        config_updates=dict(dataset=dataset,
                            n_jobs=3,
                            output_dir=join(get_data_dirs()[0], 'pipeline',
                                            'contrast',
                                            'reduced',
                                            'non_standardized')))
    run()

Parallel(n_jobs=4)(delayed(launch_dataset)(dataset) for dataset in
                   ['la5c', 'archi', 'brainomics'])
