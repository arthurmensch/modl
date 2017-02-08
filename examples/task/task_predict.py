import shutil
from os.path import join
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from sacred import Ingredient, Experiment
from sacred.observers import MongoObserver

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

monkey_patch_nifti_image()

from modl.input_data.fmri.base import safe_to_filename

from modl.classification.fmri import fMRITaskClassifier
from modl.plotting.fmri import display_maps

from sklearn.preprocessing import LabelEncoder

from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.datasets.hcp import fetch_hcp, contrasts_description
from modl.utils.system import get_cache_dirs

import matplotlib.pyplot as plt

import sys
from os import path

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.decomposition_fmri \
    import decomposition_ex, compute_decomposition, rest_data_ing

task_data_ing = Ingredient('task_data')
prediction_ex = Experiment('task_predict', ingredients=[task_data_ing,
                                                        decomposition_ex])

observer = MongoObserver.create(db_name='amensch', collection='runs')
prediction_ex.observers.append(observer)

@prediction_ex.config
def config():
    standardize = True
    C = np.logspace(-1, 2, 15)
    n_jobs = 1
    verbose = 10
    seed = 2
    max_iter = 10000
    tol = 1e-7


@task_data_ing.config
def config():
    n_subjects = 8
    train_size = 5
    test_size = 3
    seed = 2


@rest_data_ing.config
def config():
    source = 'hcp'
    n_subjects = 3 # Overriden
    train_size = 2  # Overriden
    test_size = 1
    seed = 2
    # train and test are overriden


@decomposition_ex.config
def config():
    batch_size = 100
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-4
    n_epochs = 1
    smoothing_fwhm = 4
    n_components = 40
    n_jobs = 1
    verbose = 15
    seed = 2
    raw = None


@task_data_ing.capture
def get_task_data(n_subjects, train_size, test_size, _run, _seed):
    print('Retrieve task data')
    data = fetch_hcp(n_subjects=n_subjects)
    imgs = data.task
    mask_img = data.mask
    subjects = imgs.index.get_level_values('subject').unique().values.tolist()
    train_subjects, test_subjects = \
        train_test_split(subjects, random_state=_seed, test_size=test_size)
    train_subjects = train_subjects[:train_size]
    _run.info['pred_train_subject'] = train_subjects
    _run.info['pred_test_subjects'] = test_subjects

    # Selection of contrasts
    interesting_con = list(contrasts_description.keys())
    imgs = imgs.loc[(slice(None), slice(None), interesting_con), :]

    contrast_labels = imgs.index.get_level_values(2).values
    label_encoder = LabelEncoder()
    contrast_labels = label_encoder.fit_transform(contrast_labels)
    imgs = imgs.assign(label=contrast_labels)

    imgs_list = pd.concat([imgs.loc[train_subjects],
                      imgs.loc[test_subjects]], keys=['train', 'test'])

    return imgs_list, mask_img, label_encoder


@prediction_ex.automain
def run(C,
        standardize,
        tol, max_iter,
        n_jobs,
        decomposition,
        _run,
        _seed):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=0)

    imgs_list, mask_img, label_encoder = get_task_data()

    train_subjects = imgs_list.loc['train'].index.get_level_values('subject').\
        unique().values
    test_subjects = imgs_list.loc['test'].index.get_level_values('subject').\
        unique().values

    if not decomposition['rest_data']['source'] == 'hcp':
        train_subjects = None
        test_subjects = None

    print('Compute components')
    dict_fact, final_score = compute_decomposition(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        observe=False)

    if _run.unobserved:
        _run.info['unsupervised_score'] = final_score
        artifact_dir = mkdtemp()
        safe_to_filename(dict_fact.components_img_,
            join(artifact_dir, 'components.nii.gz'))
        safe_to_filename(mask_img, join(artifact_dir, 'mask_img.nii.gz'))
        _run.add_artifact(join(artifact_dir, 'components.nii.gz'),
                          name='components.nii.gz')
        fig = plt.figure()
        display_maps(fig, dict_fact.components_img_)
        plt.savefig(join(artifact_dir, 'components.png'))
        plt.close(fig)
        _run.add_artifact(join(artifact_dir, 'components.png'),
                          name='components.png')

    print('Compute loadings')
    classifier = fMRITaskClassifier(transformer=dict_fact,
                                    memory=memory,
                                    memory_level=2,
                                    C=C,
                                    standardize=standardize,
                                    random_state=_seed,
                                    tol=tol,
                                    max_iter=max_iter,
                                    n_jobs=n_jobs,
                                    )
    classifier.fit(imgs_list.loc['train', 'filename'].values)

    true_labels = imgs_list.index.get_level_values('contrast').values

    predicted_labels = classifier.predict(imgs_list.loc[:, 'filename'].values)
    prediction = pd.DataFrame(data=list(zip(true_labels, predicted_labels)),
                              columns=['true_label', 'predicted_label'],
                              index=imgs_list.index)

    train_score = np.sum(prediction.loc['train', 'predicted_label']
                         == prediction.loc['train', 'true_label'])
    train_score /= prediction.loc['train'].shape[0]

    test_score = np.sum(prediction.loc['test', 'predicted_label']
                        == prediction.loc['test', 'true_label'])
    test_score /= prediction.loc['test'].shape[0]

    if not _run.unobserved:
        _run.info['pred_train_score'] = train_score
        _run.info['pred_test_score'] = test_score
        print('Write task prediction artifacts')
        mask_img = check_niimg(mask_img)
        mask_img.to_filename(join(artifact_dir, 'pred_mask_img.nii.gz'))
        prediction.to_csv(join(artifact_dir, 'pred_prediction.csv'))
        _run.add_artifact(join(artifact_dir, 'pred_prediction.csv'),
                          name='pred_prediction.csv')
        try:
            shutil.rmtree(artifact_dir)
        except FileNotFoundError:
            pass

    return test_score
