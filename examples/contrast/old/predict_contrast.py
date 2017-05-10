import os
import time
from os.path import join

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory, dump, load
from sklearn.preprocessing import LabelEncoder

from modl.classification import FactoredLogistic
from modl.datasets import get_data_dirs
from modl.model_selection import StratifiedGroupShuffleSplit

idx = pd.IndexSlice

predict_contrast = Experiment('predict_contrast')
collection = predict_contrast.path

global_artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                           'prediction')

observer = MongoObserver.create(db_name='amensch', collection=collection)
predict_contrast.observers.append(observer)


@predict_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    from_loadings = True
    reduced_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')

    datasets = ['archi']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1)
    train_size = None

    validation = True

    max_samples = int(1e6)
    alpha = 0.0001
    beta = 0.0
    latent_dim = 30
    activation = 'linear'
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 100
    early_stop = False
    optimizer = 'adam'
    fine_tune = 0.1

    # projection = True

    fit_intercept = True
    identity = False

    model_indexing = 'dataset'

    n_jobs = 24
    verbose = 2
    seed = 10


@predict_contrast.automain
def run(alpha,
        beta,
        latent_dim,
        model_indexing,
        dataset_weight,
        batch_size,
        max_samples,
        n_jobs,
        n_subjects,
        test_size,
        train_size,
        dropout_input,
        dropout_latent,
        early_stop,
        fit_intercept,
        optimizer,
        datasets,
        activation,
        reduced_dir,
        validation,
        fine_tune,
        verbose,
        _run,
        _seed):
    artifact_dir = join(global_artifact_dir,
                        str(_run._id), '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if verbose:
        print('Fetch data')

    X = []
    for dataset in datasets:
        this_reduced_dir = join(reduced_dir, dataset)
        this_X = load(join(this_reduced_dir, 'Xt.pkl'), mmap_mode='r')
        subjects = this_X.index.get_level_values('subject'). \
            unique().values.tolist()
        subjects = subjects[:n_subjects[dataset]]
        this_X = this_X.loc[idx[subjects]]
        X.append(this_X)
    X = pd.concat(X, keys=datasets, names=['dataset'])
    X.sort_index(inplace=True)

    datasets = X.index.get_level_values('dataset').values
    tasks = X.index.get_level_values('task')
    contrasts = X.index.get_level_values('contrast')

    dataset_tasks = list(map(lambda x: '_'.join(x), zip(datasets, tasks)))
    labels = list(map(lambda x: '_'.join(x), zip(datasets, tasks, contrasts)))

    unique_datasets = np.unique(datasets)
    if not isinstance(train_size, dict):
        new_train_size = {}
        for dataset in unique_datasets:
            new_train_size[dataset] = train_size
        train_size = new_train_size
    if not isinstance(test_size, dict):
        new_test_size = {}
        for dataset in unique_datasets:
            new_test_size[dataset] = test_size
        test_size = new_test_size

    if model_indexing == 'dataset':
        model_indices = pd.Series(data=datasets, index=X.index)
        stratify_levels = 'dataset'
        model_weight = dataset_weight
    elif model_indexing == 'dataset_task':
        model_indices = pd.Series(data=dataset_tasks, index=X.index)
        model_weight = {}
        new_test_size = {}
        new_train_size = {}
        for dataset, dataset_X in X.groupby(level='dataset'):
            n_tasks = dataset_X.index.get_level_values('task').unique().shape[0]
            for task, task_X in dataset_X.groupby(level='task'):
                dataset_task = dataset + '_' + task
                model_weight[dataset_task] = dataset_weight[dataset]
                model_weight[dataset_task] *= task_X.shape[0] / dataset_X.shape[0]
                # model_weight[dataset_task] /= n_tasks
                new_test_size[dataset_task] = test_size[dataset]
                new_train_size[dataset_task] = train_size[dataset]
        test_size = new_test_size
        train_size = new_train_size
        stratify_levels = ['dataset', 'task']
    else:
        raise ValueError

    print('Model weight: ', model_weight)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = pd.Series(data=y, index=X.index)

    if verbose:
        print('Split data')
    cv = StratifiedGroupShuffleSplit(stratify_levels=stratify_levels,
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=train_size,
                                     n_splits=1,
                                     random_state=_seed)
    train, test = next(cv.split(X))
    y_train = y.iloc[train]

    if validation:
        cv = StratifiedGroupShuffleSplit(stratify_levels=stratify_levels,
                                         group_name='subject',
                                         test_size=.1,
                                         train_size=None,
                                         n_splits=1,
                                         random_state=_seed)
        sub_train, val = next(cv.split(y_train))
        sub_train = train[sub_train]
        val = train[val]
        X_train = X.iloc[sub_train]
        y_train = y.iloc[sub_train]
        model_indices_train = model_indices.iloc[sub_train]
        X_val = X.iloc[val]
        y_val = y.iloc[val]
        model_indices_val = model_indices.iloc[val]
        train = sub_train
        fit_kwargs = {'validation_data': (X_val, y_val, model_indices_val)}
    else:
        model_indices_train = model_indices.iloc[train]
        X_train = X.iloc[train]
        fit_kwargs = {}

    if verbose:
        print('Transform and fit data')
    classifier = FactoredLogistic(optimizer=optimizer,
                                  max_samples=max_samples,
                                  activation=activation,
                                  fit_intercept=fit_intercept,
                                  latent_dim=latent_dim,
                                  dropout_latent=dropout_latent,
                                  dropout_input=dropout_input,
                                  alpha=alpha,
                                  early_stop=early_stop,
                                  fine_tune=fine_tune,
                                  beta=beta,
                                  batch_size=batch_size,
                                  n_jobs=n_jobs,
                                  verbose=verbose)

    t0 = time.time()
    classifier.fit(X_train, y_train, model_weight=model_weight,
                   model_indices=model_indices_train,
                   **fit_kwargs)

    print('Fit time: %.2f' % (time.time() - t0))

    if model_indexing == 'dataset_task':
        print('Refitting')
        model_indices = pd.Series(data=datasets, index=X.index)
        model_indices_train = model_indices.iloc[train]
        model_indices_val = model_indices.iloc[val]
        model_weight = dataset_weight
        fit_kwargs = {'validation_data': (X_val, y_val, model_indices_val)}

        dump(classifier, join(artifact_dir, 'classifier.pkl'))

        new_classifier = FactoredLogistic(optimizer=optimizer,
                                          max_samples=max_samples,
                                          activation='linear',
                                          fit_intercept=True,
                                          latent_dim=latent_dim,
                                          dropout_latent=dropout_latent,
                                          dropout_input=dropout_input,
                                          alpha=alpha,
                                          beta=beta,
                                          early_stop=False,
                                          fine_tune=0,
                                          batch_size=batch_size,
                                          n_jobs=n_jobs,
                                          verbose=verbose)
        new_classifier.fit(X_train, y_train, model_weight=model_weight,
                           latent_weights=classifier.encoder_.get_layer('latent').get_weights(),
                           model_indices=model_indices_train,
                           **fit_kwargs)
        classifier = new_classifier

    predicted_labels = classifier.predict(X, model_indices=model_indices)
    predicted_labels = le.inverse_transform(predicted_labels)
    labels = le.inverse_transform(y)

    prediction = pd.DataFrame({'true_label': labels,
                               'predicted_label': predicted_labels},
                              index=X.index)

    if validation:
        prediction = pd.concat([prediction.iloc[train],
                                prediction.iloc[val],
                                prediction.iloc[test]],
                               names=['fold'], keys=['train', 'val', 'test'])
    else:
        prediction = pd.concat([prediction.iloc[train],
                                prediction.iloc[test]],
                               names=['fold'], keys=['train', 'test'])
    prediction.sort_index()
    match = prediction['true_label'] == prediction['predicted_label']

    _run.info['n_epochs'] = classifier.n_epochs_
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
    dump(le, join(artifact_dir, 'label_encoder.pkl'))
    if model_indexing == 'dataset_task':
        dump(classifier, join(artifact_dir, 'new_classifier.pkl'))
