import sys
from os import path

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.contrast.reduce_contrast import reduce_contrast
from examples.unmask.unmask_contrast import unmask_contrast

for dataset in ['la5c', 'brainomics', 'archi', 'hcp']:
    run = unmask_contrast._create_run(config_updates=dict(dataset=dataset))
    run()
    run = reduce_contrast._create_run(config_updates=dict(dataset=dataset))
    run()
