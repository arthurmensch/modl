import copy
import shutil
from os.path import expanduser, join
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from sacred import Ingredient, Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from modl.utils.nifti import monkey_patch_nifti_image

monkey_patch_nifti_image()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.datasets.hcp import fetch_hcp, contrasts_description, fetch_hcp_task
from modl.utils.system import get_cache_dirs
from modl.fmri import compute_loadings

import sys
from os import path

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.development.decomposition_fmri \
    import decomposition_ex, compute_decomposition, rest_data_ing

task_data_ing = Ingredient('task_data')
prediction_ex = Experiment('task_predict', ingredients=[task_data_ing,
                                                        decomposition_ex])

observer = MongoObserver.create(db_name='amensch', collection='runs')
prediction_ex.observers.append(observer)

observer = FileStorageObserver.create(expanduser('~/output/runs'))
prediction_ex.observers.append(observer)


@prediction_ex.config
def config():
    test_size = 0.1
    standardize = True
    cross_val = True
    n_jobs = 1
    verbose = 2
    seed = 2


@task_data_ing.config
def config():
    train_size = 20
    test_size = 10
    seed = 2


@rest_data_ing.config
def config():
    source = 'hcp'
    seed = 2
    # train and test are overriden


@decomposition_ex.config
def config():
    batch_size = 100
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    smoothing_fwhm = 4
    n_components = 40
    n_jobs = 1
    verbose = 15
    seed = 2


@task_data_ing.capture
def get_task_data(train_size, test_size, _run, _seed):
    print('Retrieve task data')
    data = fetch_hcp()
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

    train_imgs = imgs.loc[train_subjects, :]
    test_imgs = imgs.loc[test_subjects, :]

    return train_imgs, train_subjects, test_imgs, test_subjects, \
           mask_img, label_encoder


@prediction_ex.capture
def logistic_regression(X_train, y_train,
                        # Injected parameters
                        standardize,
                        cross_val,
                        n_jobs,
                        _run,
                        _seed):
    memory = Memory(cachedir=get_cache_dirs()[0])
    lr = memory.cache(_logistic_regression)(X_train, y_train,
                                            standardize=standardize,
                                            cross_val=cross_val,
                                            n_jobs=n_jobs,
                                            random_state=_seed)
    if cross_val:
        best_C = lr.best_params_["logistic_regression__C"]
        print('Best C %.3f' % best_C)
        _run.info['pred_best_C'] = best_C
    return lr


def _logistic_regression(X_train, y_train, standardize=False, cross_val=False,
                         n_jobs=1, random_state=None):
    """Function to be cached"""
    lr = LogisticRegression(multi_class='multinomial',
                            solver='sag', tol=1e-4, max_iter=1000, verbose=2,
                            random_state=random_state)
    if standardize:
        sc = StandardScaler()
        lr = Pipeline([('standard_scaler', sc), ('logistic_regression', lr)])
    if cross_val:
        lr = GridSearchCV(lr,
                          {'logistic_regression__C': np.logspace(-1, 1, 10)},
                          cv=ShuffleSplit(test_size=0.1),
                          n_jobs=n_jobs)
    lr.fit(X_train, y_train)
    return lr


@prediction_ex.automain
def run(n_jobs,
        decomposition,
        _run):
    memory = Memory(cachedir=get_cache_dirs()[0])

    train_data, train_subjects, test_data,\
    test_subjects, mask_img, label_encoder = get_task_data()

    if not decomposition['rest_data']['source'] == 'hcp':
        train_subjects = None
        test_subjects = None
    else:
        # Reduce the test size for efficient debug control
        test_subjects = test_subjects[:1]
    print('Compute components')
    components, mask_img = compute_decomposition(train_subjects=train_subjects,
                                                 test_subjects=test_subjects)

    data = pd.concat([train_data, test_data], keys=['train', 'test'],
                     names=['fold'])
    print('Compute loadings')
    loadings = memory.cache(compute_loadings,
                            ignore=['n_jobs'])(
        data.loc[:, 'filename'].values,
        components,
        mask=mask_img,
        n_jobs=n_jobs)
    data = data.assign(loadings=loadings)

    X = data.loc[:, 'loadings']
    y = data.loc[:, 'label']
    X_train = np.vstack(X.loc['train'].values)
    y_train = y.loc['train'].values
    X = np.vstack(X)

    print('Fit logistic regression')
    lr_estimator = logistic_regression(X_train, y_train)

    print('Dump results')
    y_pred = lr_estimator.predict(X)
    true_labels = data.index.get_level_values('contrast').values
    predicted_labels = label_encoder.inverse_transform(y_pred)
    prediction = pd.DataFrame(data=list(zip(true_labels, predicted_labels)),
                              columns=['true_label', 'predicted_label'],
                              index=data.index)

    train_score = np.sum(prediction.loc['train', 'predicted_label']
                         == prediction.loc['train', 'true_label'])
    train_score /= prediction.loc['train'].shape[0]

    test_score = np.sum(prediction.loc['test', 'predicted_label']
                        == prediction.loc['test', 'true_label'])
    test_score /= prediction.loc['test'].shape[0]

    _run.info['pred_train_score'] = train_score
    _run.info['pred_test_score'] = test_score
    if not _run.unobserved:
        print('Write task prediction artifacts')
        artifact_dir = mkdtemp()
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
