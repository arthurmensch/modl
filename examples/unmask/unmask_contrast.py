from os.path import join

from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from modl.datasets import get_data_dirs
from modl.datasets.hcp import fetch_hcp, INTERESTING_CONTRASTS_EXTENDED
from modl.input_data.fmri.unmask import create_raw_contrast_data

unmask_task = Experiment('unmask_task')
observer = MongoObserver.create(db_name='amensch', collection='runs')
unmask_task.observers.append(observer)


@unmask_task.config
def config():
    n_jobs = 10
    batch_size = 1200


@unmask_task.automain
def run(n_jobs, batch_size, _run):
    dataset = fetch_hcp()
    imgs = dataset.contrasts
    interesting_con = INTERESTING_CONTRASTS_EXTENDED
    imgs = imgs.loc[(slice(None), slice(None), interesting_con), :]
    mask = dataset.mask

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'unmask',
                        'contrast', 'hcp', '23')
    _run.info['artifact_dir'] = artifact_dir

    memory = Memory(cachedir=None)

    create_raw_contrast_data(imgs, mask, artifact_dir, n_jobs=n_jobs,
                             memory=memory,
                             batch_size=batch_size)

    with open(join(artifact_dir, 'exp_id'), 'w+') as f:
        f.write(str(_run._id))
