import os
from os.path import join

import numpy as np
import pandas as pd
from modl.datasets import get_data_dirs
from modl.hierarchical import make_multi_model, init_tensorflow
from modl.model_selection import StratifiedGroupShuffleSplit
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import load
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import gen_batches, check_random_state, shuffle

idx = pd.IndexSlice

predict_contrast_exp = Experiment('predict_contrast_legacy')
collection = predict_contrast_exp.path

observer = MongoObserver.create(db_name='amensch', collection=collection)


def batch_generator_dataset(Xs, y_ohs, models, batch_size, random_state):
    batch_generators = {}
    while True:
        for dataset in Xs['dataset']:
            y_oh = y_ohs['dataset'][dataset]
            X = Xs['dataset'][dataset]
            model = models['dataset'][dataset]
            try:
                batch = next(batch_generators[dataset])
                new_epoch = False
            except (KeyError, StopIteration):
                len_dataset = X.shape[0]
                permutation = random_state.permutation(len_dataset)
                X = X.iloc[permutation]
                y_oh = y_oh.iloc[permutation]
                Xs['dataset'][dataset] = X
                y_ohs['dataset'][dataset] = y_oh
                batch_generators[dataset] = gen_batches(len_dataset,
                                                        batch_size)
                batch = next(batch_generators[dataset])
                new_epoch = True
            X_batch = X.iloc[batch].values
            y_oh_batch = y_oh.iloc[batch].values
            yield dataset, model, X_batch, y_oh_batch, new_epoch


def batch_generator_task(Xs, y_ohs, models, batch_size, seed,
                         dataset):
    batch_generators = {}
    random_state = check_random_state(seed)
    while True:
        for task in Xs['task'][dataset]:
            y_oh = y_ohs['task'][dataset][task]
            X = Xs['task'][dataset][task]
            model = models['task'][dataset][task]
            try:
                batch = next(batch_generators[task])
                new_epoch = False
            except (KeyError, StopIteration):
                len_task = X.shape[0]
                permutation = random_state.permutation(len_task)
                X = X.iloc[permutation]
                y_oh = y_oh.iloc[permutation]
                Xs['task'][dataset][task] = X
                y_ohs['task'][dataset][task] = y_oh
                batch_generators[task] = gen_batches(len_task,
                                                     batch_size)
                batch = next(batch_generators[task])
                new_epoch = True
            X_batch = X.iloc[batch].values
            y_oh_batch = y_oh.iloc[batch].values
            yield task, model, X_batch, y_oh_batch, new_epoch


def scale(X, train, per_dataset_std):
    X_train = X.iloc[train]
    if per_dataset_std:
        standard_scaler = {}
        for dataset, this_X_train in X_train.groupby(level='dataset'):
            this_standard_scaler = StandardScaler()
            this_standard_scaler.fit(this_X_train)
            standard_scaler[dataset] = this_standard_scaler
        for dataset, this_X in X.groupby(level='dataset'):
            this_X_new = standard_scaler[dataset].transform(this_X)
            X.loc[dataset] = this_X_new
    else:
        standard_scaler = StandardScaler(with_std=False)
        standard_scaler.fit(X_train)
        X_new = standard_scaler.transform(X)
        X = pd.DataFrame(X_new, index=X.index)
    return X, standard_scaler


@predict_contrast_exp.config
def config():
    reduced_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')
    unmask_dir = join(get_data_dirs()[0], 'pipeline', 'unmask',
                      'contrast')
    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical')
    datasets = ['archi', 'brainomics', 'hcp']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5,
                     camcan=.5,
                     human_voice=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1,
                          camcan=1,
                          human_voice=1)
    train_size = dict(hcp=None, archi=20, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    geometric_reduction = True
    alpha = 1e-4
    latent_dim = 50
    activation = 'linear'
    source = 'hcp_rs_concat'
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 100
    budget = 4e6
    n_jobs = 2
    verbose = 2
    seed = 10
    _seed = 0

    use_task_specific = False


def test_model(prediction):
    match = prediction.values == prediction.index.get_level_values(
        'contrast').values
    match = pd.Series(match, prediction.index)
    score = match.groupby(level=['fold', 'dataset']).aggregate('mean')
    res = {}
    for fold, sub_score in score.groupby(level='fold'):
        res[fold] = {}
        for (_, dataset), this_score in sub_score.iteritems():
            res[fold][dataset] = this_score
    return res


@predict_contrast_exp.automain
def train_model(alpha,
                latent_dim,
                n_subjects,
                unmask_dir,
                geometric_reduction,
                test_size,
                train_size,
                budget,
                dropout_input,
                source,
                dropout_latent,
                activation,
                datasets,
                dataset_weight,
                reduced_dir,
                batch_size,
                artifact_dir,
                use_task_specific,
                verbose,
                n_jobs,
                _run,
                _seed):
    artifact_dir = join(artifact_dir, str(_run._id))
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if verbose:
        print('Fetch data')
    X = []
    keys = []

    for dataset in datasets:
        if dataset_weight[dataset] != 0:
            this_reduced_dir = join(reduced_dir, source, dataset)
            if geometric_reduction:
                X_dataset = load(join(this_reduced_dir, 'Xt.pkl'))
            else:
                X_dataset = load(join(unmask_dir, dataset, 'imgs.pkl'))
            if dataset in ['archi', 'brainomics']:
                X_dataset = X_dataset.drop(['effects_of_interest'],
                                           level='contrast', )
            subjects = X_dataset.index.get_level_values('subject'). \
                unique().values.tolist()
            subjects = subjects[:n_subjects[dataset]]
            X_dataset = X_dataset.loc[idx[subjects]]
            X.append(X_dataset)
            keys.append(dataset)

    X = pd.concat(X, keys=keys, names=['dataset'])

    X = X.reset_index(level=['direction'], drop=True)
    X.sort_index(inplace=True)

    # Cross validation folds
    cv = StratifiedGroupShuffleSplit(stratify_levels='dataset',
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=train_size,
                                     n_splits=1,
                                     random_state=0)
    train, test = next(cv.split(X))

    X = pd.concat([X.iloc[train], X.iloc[test]], keys=['train', 'test'],
                  names=['fold'])
    X.sort_index(inplace=True)
    n_features = X.shape[1]

    lbins = {'dataset': {}, 'task': {}}
    Xs_train = {'dataset': {}, 'task': {}}
    Xs_test = {'dataset': {}, 'task': {}}
    y_ohs_train = {'dataset': {}, 'task': {}}
    y_ohs_test = {'dataset': {}, 'task': {}}
    pred = {'dataset': {}, 'task': {}}
    for dataset, X_dataset in X.groupby(level='dataset'):
        lbin = LabelBinarizer()
        this_y = X_dataset.index.get_level_values(level='contrast')
        this_y_oh = lbin.fit_transform(this_y)
        this_y_oh = pd.DataFrame(data=this_y_oh, index=X_dataset.index)
        lbins['dataset'][dataset] = lbin
        y_ohs_train['dataset'][dataset] = this_y_oh.query("fold == 'train'")
        y_ohs_test['dataset'][dataset] = this_y_oh.query("fold == 'test'")
        Xs_train['dataset'][dataset] = X_dataset.query("fold == 'train'")
        Xs_test['dataset'][dataset] = X_dataset.query("fold == 'test'")

        if use_task_specific:
            lbins['task'][dataset] = {}
            Xs_train['task'][dataset] = {}
            Xs_test['task'][dataset] = {}
            y_ohs_train['task'][dataset] = {}
            y_ohs_test['task'][dataset] = {}
            for task, X_task in X_dataset.groupby(level='task'):
                lbin = LabelBinarizer()
                this_y = X_task.index.get_level_values(level='contrast')
                this_y_oh = lbin.fit_transform(this_y)
                this_y_oh = pd.DataFrame(data=this_y_oh, index=X_task.index)
                lbins['task'][dataset][task] = lbin
                this_train = X_task.index.get_level_values(
                    'fold').values == 'train'
                this_test = X_task.index.get_level_values(
                    'fold').values == 'test'
                Xs_train['task'][dataset][task] = X_task.iloc[this_train]
                y_ohs_train['task'][dataset][task] = this_y_oh.iloc[this_train]
                Xs_test['task'][dataset][task] = X_task.iloc[this_test]
                y_ohs_test['task'][dataset][task] = this_y_oh.iloc[this_test]

    init_tensorflow(n_jobs=n_jobs, debug=False)

    models = make_multi_model(n_features=n_features,
                              lbins=lbins,
                              alpha=alpha,
                              latent_dim=latent_dim,
                              activation=activation,
                              dropout_input=dropout_input,
                              dropout_latent=dropout_latent,
                              use_task_specific=use_task_specific,
                              seed=_seed)

    random_state = check_random_state(_seed)
    our_batch_generator_dataset = batch_generator_dataset(Xs_train,
                                                          y_ohs_train, models,
                                                          batch_size,
                                                          random_state)
    if use_task_specific:
        our_batch_generator_task = {
            dataset: batch_generator_task(Xs_train, y_ohs_train, models,
                                          batch_size,
                                          _seed,
                                          dataset)
            for dataset in Xs_train['dataset']}

    n_samples = 0
    n_epochs = {dataset: 0 for dataset in datasets}
    while n_samples < budget:
        (dataset, model, X_batch,
         y_oh_batch, new_epoch) = next(our_batch_generator_dataset)
        n_samples += X_batch.shape[0]
        if not use_task_specific:
            model.train_on_batch(X_batch, y_oh_batch)
        if new_epoch:
            n_epochs[dataset] += 1
            X_test = Xs_test['dataset'][dataset]
            lbin = lbins['dataset'][dataset]
            contrast = lbin.inverse_transform(model.predict(X_test.values))
            true_contrast = X_test.index.get_level_values(
                level='contrast').values
            accuracy = np.mean(contrast == true_contrast)
            print(dataset, 'epoch %5i' % n_epochs[dataset],
                  'n_samples %6i' % n_samples,
                  'test accuracy', accuracy)

        if use_task_specific:
            task, model, X_batch, y_oh_batch, new_epoch = next(
                our_batch_generator_task[dataset])
            n_samples += X_batch.shape[0]
            model.train_on_batch(X_batch, y_oh_batch)
            if new_epoch:
                X_test = Xs_test['task'][dataset][task]
                lbin = lbins['task'][dataset][task]
                contrast = lbin.inverse_transform(model.predict(X_test.values))
                true_contrast = X_test.index.get_level_values(
                    level='contrast').values
                accuracy = np.mean(contrast == true_contrast)
                print(n_samples, 'condition vs task', dataset, task, 'acc',
                      accuracy)

    for dataset in Xs_test['dataset']:
        X_dataset = Xs_test['dataset'][dataset]
        model = models['dataset'][dataset]
        lbin = lbins['dataset'][dataset]
        contrast = lbin.inverse_transform(model.predict(X_dataset.values))
        pred['dataset'][dataset] = pd.Series(data=contrast,
                                             index=X_dataset.index)
        if use_task_specific:
            for task, X_task in X_dataset.groupby(level='task'):
                model = models['task'][dataset][task]
                lbin = lbins['task'][dataset][task]
                contrast = lbin.inverse_transform(model.predict(X_task.values))
                pred['task'][(dataset, task)] = pd.Series(data=contrast,
                                                          index=X_task.index)

    pred_dataset = pd.concat(pred['dataset'].values())
    pred_dataset.sort_index(inplace=True)
    pred_dataset.to_csv(join(artifact_dir, 'prediction_dataset.csv'))

    res_dataset = test_model(pred_dataset)
    _run.info['score'] = {'dataset': res_dataset}

    _run.add_artifact(join(artifact_dir,
                           'prediction_dataset.csv'), 'prediction_dataset')

    if use_task_specific:
        pred_task = pd.concat(pred['task'].values())
        pred_task.sort_index(inplace=True)
        pred_task.to_csv(join(artifact_dir, 'prediction_task.csv'))
        res_task = test_model(pred_task)
        _run.add_artifact(join(artifact_dir,
                               'prediction_task.csv'), 'prediction_task')
        _run.info['score']['task'] = res_task

    print(_run.info['score'])

    for dataset in models['dataset']:
        model = models['dataset'][dataset]
        model.save(join(artifact_dir, 'model_dataset_%s.keras' % dataset))
        if use_task_specific:
            for task in models['task'][dataset]:
                model = models['task'][dataset][task]
                model.save(join(artifact_dir, 'model_dataset_%s_%s.keras'
                                % (dataset, task)))
