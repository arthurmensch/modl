import os
from os.path import join

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from modl.classification import make_loadings_extractor, make_classifier
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design, retrieve_components
from modl.model_selection import StratifiedGroupShuffleSplit
from modl.utils.system import get_cache_dirs

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

    from_loadings = True
    loadings_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')

    datasets = ['hcp', 'archi']
    n_subjects = 30

    test_size = 0.1
    train_size = None

    factored = True

    alpha = 0.0001
    latent_dim = 100  # Factored only
    activation = 'linear'  # Factored only
    dropout = False
    penalty = 'l1'

    max_iter = 50
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
        from_loadings,
        loadings_dir,
        verbose,
        _run,
        _seed):
    artifact_dir = join(global_artifact_dir, str(_run._id), '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

    if verbose:
        print('Fetch data')
    if not from_loadings:
        X, masker = memory.cache(build_design)(datasets,
                                               datasets_dir,
                                               n_subjects)
        # Add a dataset column to the X matrix
        datasets = X.index.get_level_values('dataset').values
        dataset_encoder = LabelEncoder()
        datasets = dataset_encoder.fit_transform(datasets)
        datasets = pd.Series(index=X.index, data=datasets, name='dataset')
        X = pd.concat([X, datasets], axis=1)

        labels = X.index.get_level_values('contrast').values
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        y = pd.Series(index=X.index, data=labels, name='label')
    else:
        masker = load(join(loadings_dir, 'masker.pkl'))
        X = load(join(loadings_dir, 'Xt.pkl'))
        y = load(join(loadings_dir, 'y.pkl'))
        label_encoder = load(join(loadings_dir, 'label_encoder.pkl'))

    if verbose:
        print('Split data')
    cv = StratifiedGroupShuffleSplit(stratify_name='dataset',
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=train_size,
                                     n_splits=1,
                                     random_state=_seed)
    train, test = next(cv.split(X))

    y_train = y.iloc[train]
    train_samples = len(train)

    cv = StratifiedGroupShuffleSplit(stratify_name='dataset',
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=None,
                                     n_splits=1,
                                     random_state=_seed)
    sub_train, val = next(cv.split(y_train))

    sub_train = train[sub_train]
    val = train[val]

    X_train = X.iloc[sub_train]
    y_train = y.iloc[sub_train]
    X_val = X.iloc[val]
    y_val = y.iloc[val]

    train = sub_train

    if verbose:
        print('Transform and fit data')
    pipeline = []
    if not from_loadings:
        if verbose:
            print('Retrieve components')
        components = memory.cache(retrieve_components)(dictionary_penalty,
                                                       masker,
                                                       n_components_list)
        transformer = make_loadings_extractor(components,
                                              standardize=standardize,
                                              scale_importance=scale_importance,
                                              identity=identity,
                                              scale_bases=True,
                                              n_jobs=n_jobs,
                                              memory=memory)
        pipeline.append(('transformer', transformer))
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
                                 random_state=_seed,
                                 verbose=verbose)
    pipeline.append(('classifier', classifier))
    estimator = Pipeline(pipeline, memory=memory)

    if factored:
        if not from_loadings:
            Xt_val = transformer.fit_transform(X_val, y_val)
        else:
            Xt_val = X_val
        estimator.fit(X_train, y_train,
                      classifier__validation_data=(Xt_val, y_val))
    else:
        sample_weight = 1 / X_train[0].groupby(
            level=['dataset', 'contrast']).transform('count')
        sample_weight /= np.min(sample_weight)
        estimator.fit(X_train, y_train,
                      classifier__sample_weight=sample_weight)

    predicted_labels = estimator.predict(X)
    predicted_labels = label_encoder.inverse_transform(predicted_labels)
    labels = label_encoder.inverse_transform(y)

    prediction = pd.DataFrame({'true_label': labels,
                               'predicted_label': predicted_labels},
                              index=X.index)


    prediction = pd.concat([prediction.iloc[train],
                            prediction.iloc[val],
                            prediction.iloc[test]],
                           names=['fold'], keys=['train', 'val', 'test'])
    prediction.sort_index()
    match = prediction['true_label'] == prediction['predicted_label']

    if verbose:
        print('Compute score')
    for fold, sub_match in match.groupby(level='fold'):
        _run.info['%s_score' % fold] = np.mean(sub_match)
    for (fold, dataset), sub_match in match.groupby(level=['fold', 'dataset']):
        _run.info['%s_%s_score' % (fold, dataset)] = np.mean(sub_match)
    if verbose:
        print('Write task prediction artifacts')
    prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'),
                      name='prediction.csv')

    dump(label_encoder, join(artifact_dir, 'label_encoder.pkl'))
    dump(estimator, join(artifact_dir, 'estimator.pkl'))
