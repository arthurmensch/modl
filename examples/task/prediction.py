from os.path import join

import numpy as np
from hcp_builder.dataset import fetch_hcp_mask
from nilearn.input_data import MultiNiftiMasker
from sacred import Experiment
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.classification.fmri import fMRITaskClassifier, MultiProjectionTransformer
from modl.datasets import get_data_dirs
from modl.input_data.fmri.raw_masker import get_raw_contrast_data
from modl.utils.system import get_cache_dirs

prediction_ex = Experiment('task_predict')


@prediction_ex.config
def config():
    standardize = True
    C = np.logspace(-1, 2, 15)
    n_jobs = 1
    verbose = 10
    seed = 2
    max_iter = 10000
    tol = 1e-7
    dump_dir = join(get_data_dirs()[0], 'raw', 'hcp', 'task')
    n_components_list = [16, 64, 256]
    test_size = 0.1
    n_subjects = 100


@prediction_ex.automain
def get_raw_task(standardize, C, tol, n_components_list, max_iter, n_jobs,
                 test_size,
                 train_size,
                 n_subjects,
                 _run,
                 _seed):
    memory = Memory(cachedir=get_cache_dirs()[0])
    imgs = get_raw_contrast_data()

    subjects = imgs.index.get_level_values('subject').unique().values.tolist()
    subjects = subjects[:n_subjects]
    imgs = imgs.loc[subjects]

    train_subjects, test_subjects = \
        train_test_split(subjects, random_state=_seed, test_size=test_size)
    train_subjects = train_subjects[:train_size]
    _run.info['pred_train_subject'] = train_subjects
    _run.info['pred_test_subjects'] = test_subjects

    components_dir = '_'.join(map(str, n_components_list))
    components_dir = join(get_data_dirs()[0], 'hierarchical', components_dir)
    components_imgs = [join(components_dir,
                            'components_%i.nii.gz' % this_n_components)
                       for this_n_components in n_components_list]
    mask_img = fetch_hcp_mask(data_dir=join(get_data_dirs()[0], 'HCP900'))
    masker = MultiNiftiMasker(smoothing_fwhm=0,
                              mask_img=mask_img).fit()
    components = masker.transform(components_imgs)
    transformer = MultiProjectionTransformer(identity=True,
                                             bases=components)
    classifier = fMRITaskClassifier(transformer=transformer,
                                    memory=memory,
                                    memory_level=2,
                                    C=C,
                                    standardize=standardize,
                                    random_state=_seed,
                                    tol=tol,
                                    max_iter=max_iter,
                                    n_jobs=n_jobs,
                                    )

    train_imgs = imgs.loc[train_subjects]
    train_labels = train_imgs.get_level_values('contrast').values
    train_imgs = train_imgs.values
    classifier.fit(train_imgs, train_labels)
