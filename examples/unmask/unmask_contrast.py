from os.path import join

from modl.datasets.human_voice import fetch_human_voice
from nilearn.datasets import load_mni152_brain_mask
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory

from modl.datasets.archi import fetch_archi
from modl.datasets.brainomics import fetch_brainomics
from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from modl.datasets import get_data_dirs, fetch_hcp
from modl.datasets.la5c import fetch_la5c
from modl.input_data.fmri.unmask import create_raw_contrast_data

unmask_contrast = Experiment('unmask_contrast')
observer = MongoObserver.create(db_name='amensch', collection='runs')
unmask_contrast.observers.append(observer)

output_dir = join(get_data_dirs()[0], 'pipeline', 'unmask', 'contrast')



@unmask_contrast.config
def config():
    n_jobs = 10
    batch_size = 1200
    dataset = 'human_voice'


@unmask_contrast.automain
def run(n_jobs, batch_size, dataset,
        _run):
    if dataset == 'hcp':
        fetch_data = fetch_hcp
    elif dataset == 'archi':
        fetch_data = fetch_archi
    elif dataset == 'brainomics':
        fetch_data = fetch_brainomics
    elif dataset == 'la5c':
        fetch_data = fetch_la5c
    elif dataset == 'human_voice':
        fetch_data = fetch_human_voice
    else:
        raise ValueError

    imgs = fetch_data()
    if dataset == 'hcp':
        imgs = imgs.contrasts
    mask = fetch_hcp(n_subjects=1).mask

    artifact_dir = join(output_dir, dataset)
    _run.info['artifact_dir'] = artifact_dir

    memory = Memory(cachedir=None)

    create_raw_contrast_data(imgs, mask, artifact_dir, n_jobs=n_jobs,
                             memory=memory,
                             batch_size=batch_size)

    with open(join(artifact_dir, 'exp_id'), 'w+') as f:
        f.write(str(_run._id))
