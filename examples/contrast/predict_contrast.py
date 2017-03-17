import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from modl.utils import concatenated_cv
from modl.classification import make_loadings_extractor, make_classifier
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design, retrieve_components
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

predict_contrast = Experiment('predict_contrast')

global_artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                           'prediction')

observer = MongoObserver.create(db_name='amensch', collection='runs')
predict_contrast.observers.append(observer)

observer = FileStorageObserver.create(basedir=global_artifact_dir)
predict_contrast.observers.append(observer)


@predict_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    datasets = ['hcp', 'archi']
    test_size = 0.1
    train_size = None
    n_subjects = 30

    factored = True

    alpha = 0.0001
    latent_dim = 100  # Factored only
    activation = 'linear'  # Factored only
    dropout = False
    penalty = 'l1'

    max_iter = 200
    tol = 1e-7  # Non-factored only

    standardize = True
    scale_importance = 'sqrt'
    multi_class = 'multinomial'  # Non-factored only

    fit_intercept = True
    identity = False
    refit = False  # Non-factored only

    n_jobs = 24
    verbose = 2
    seed = 2

    hcp_unmask_contrast_dir = join(get_data_dirs()[0], 'pipeline',
                                   'unmask', 'contrast', 'hcp', '23')
    archi_unmask_contrast_dir = join(get_data_dirs()[0], 'pipeline',
                                     'unmask', 'contrast', 'archi', '30')
    datasets_dir = {'archi': archi_unmask_contrast_dir,
                    'hcp': hcp_unmask_contrast_dir}

    del hcp_unmask_contrast_dir
    del archi_unmask_contrast_dir


@predict_contrast.automain
def run(dictionary_penalty,
        alpha,
        latent_dim,
        n_components_list,
        max_iter, n_jobs,
        test_size,
        train_size,
        tol,
        dropout,
        identity,
        fit_intercept,
        multi_class,
        n_subjects,
        scale_importance,
        standardize,
        penalty,
        datasets,
        datasets_dir,
        factored,
        activation,
        _run,
        _seed):
    artifact_dir = join(global_artifact_dir, str(_run._id), '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    X, masker = memory.cache(build_design)(datasets,
                                           datasets_dir,
                                           n_subjects)
    if factored:
        # Add a dataset column to the X matrix
        datasets = X.index.get_level_values('dataset').values
        dataset_encoder = LabelEncoder()
        datasets = dataset_encoder.fit_transform(datasets)
        datasets = pd.Series(index=X.index, data=datasets, name='dataset')
        X = pd.concat([X, datasets], axis=1)

    print('Split data')
    single_X = X.iloc[:, 0].reset_index()
    single_X.iloc[:, 0][0] = np.arange(single_X.shape[0])
    train = []
    test = []
    for idx, df in single_X.groupby(by='dataset'):
        cv = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               train_size=train_size, random_state=0)
        splitter = cv.split(df.index.tolist(), groups=df['subject'].values)
        group_train, group_test = next(splitter)
        this_train, this_test = df[0].iloc[group_train], df[0].iloc[group_test]
        train.append(this_train)
        test.append(this_test)
    train = np.concatenate(train)
    test = np.concatenate(test)

    train_samples = len(train)

    X_train = X.iloc[train]
    X_test = X.iloc[test]

    _run.info['train'] = train
    _run.info['test'] = test

    print('Retrieve components')
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)

    print('Transform and fit data')
    labels = X.index.get_level_values('contrast').values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    y = pd.Series(index=X.index, data=labels, name='label')

    y_train = y.iloc[train]
    y_test = y.iloc[test]

    pipeline = make_loadings_extractor(components,
                                       standardize=standardize,
                                       scale_importance=scale_importance,
                                       identity=identity,
                                       factored=factored,
                                       scale_bases=True,
                                       n_jobs=n_jobs,
                                       memory=memory)
    classifier = make_classifier(alpha, latent_dim,
                                 factored=factored,
                                 fit_intercept=fit_intercept,
                                 activation=activation,
                                 max_iter=max_iter,
                                 multi_class=multi_class,
                                 dropout=dropout,
                                 n_jobs=n_jobs,
                                 penalty=penalty,
                                 tol=tol,
                                 train_samples=train_samples,
                                 random_state=_seed)
    estimator = Pipeline([('transformer', pipeline),
                          ('classifier', classifier)],
                         memory=memory)

    if factored:
        Xt_test = pipeline.fit_transform(X_test, y_test)
        estimator.fit(X_train, y_train, classifier__validation_data=
        (Xt_test, y_test))
    else:
        sample_weight = 1 / X_train[0].groupby(
            level=['dataset', 'contrast']).transform('count')
        sample_weight /= np.min(sample_weight)
        estimator.fit(X_train, y_train,
                      classifier__sample_weight=sample_weight)

    predicted_labels = estimator.predict(X)
    predicted_labels = label_encoder.inverse_transform(predicted_labels)
    labels = label_encoder.inverse_transform(labels)

    prediction = pd.DataFrame({'true_label': labels,
                               'predicted_label': predicted_labels},
                              index=X.index)

    prediction = pd.concat([prediction.iloc[train], prediction.iloc[test]],
                           names=['fold'], keys=['train', 'test'])
    prediction.sort_index()

    print('Compute score')
    train_score = np.sum(prediction.loc['train']['predicted_label']
                         == prediction.loc['train']['true_label'])
    train_score /= prediction.loc['train'].shape[0]

    _run.info['train_score'] = float(train_score)

    test_score = np.sum(prediction.loc['test']['predicted_label']
                        == prediction.loc['test']['true_label'])
    test_score /= prediction.loc['test'].shape[0]

    _run.info['test_score'] = float(test_score)

    print('Write task prediction artifacts')
    prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'),
                      name='prediction.csv')

    dump(label_encoder, join(artifact_dir, 'label_encoder.pkl'))
    dump(estimator, join(artifact_dir, 'estimator.pkl'))
