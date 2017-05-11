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

predict_contrast_exp = Experiment('predict_contrast')
collection = predict_contrast_exp.path

observer = MongoObserver.create(db_name='amensch', collection=collection)


def batch_generator_dataset(Xs_train, lbins, models, batch_size, random_state):
    batch_generators = {}
    while True:
        for dataset in Xs_train['dataset']:
            lbin = lbins['dataset'][dataset]
            X_dataset = Xs_train['dataset'][dataset]
            model = models['dataset'][dataset]
            try:
                batch = next(batch_generators[dataset])
                new_epoch = False
            except (KeyError, StopIteration):
                len_dataset = X_dataset.shape[0]
                X_dataset = shuffle(X_dataset, random_state=random_state)
                Xs_train['dataset'][dataset] = X_dataset
                batch_generators[dataset] = gen_batches(len_dataset,
                                                        batch_size)
                batch = next(batch_generators[dataset])
                new_epoch = True
            X_batch = X_dataset.iloc[batch]
            y_oh_batch = X_batch.index.get_level_values(level='contrast')
            y_oh_batch = lbin.transform(y_oh_batch)
            X_batch = X_batch.values
            yield dataset, model, X_batch, y_oh_batch, new_epoch


def batch_generator_task(Xs_train, y_ohs_train, models, batch_size, seed,
                         dataset):
    batch_generators = {}
    random_state = check_random_state(seed)
    while True:
        for task in Xs_train['task'][dataset]:
            y_oh = y_ohs_train['task'][dataset][task]
            X_task = Xs_train['task'][dataset][task]
            model = models['task'][dataset][task]
            try:
                batch = next(batch_generators[task])
                new_epoch = False
            except (KeyError, StopIteration):
                len_task = X_task.shape[0]
                shuffle(X_task, random_state=random_state)
                batch_generators[task] = gen_batches(len_task,
                                                     batch_size)
                batch = next(batch_generators[task])
                new_epoch = True
            X_batch = X_task.iloc[batch]
            y_oh_batch = y_oh.iloc[batch]
            X_batch = X_batch.values
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
    datasets = ['archi', 'hcp']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5,
                     camcan=.5,
                     human_voice=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1,
                          camcan=1,
                          human_voice=1)
    train_size = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    validation = True
    geometric_reduction = True
    alpha = 1e-4
    latent_dim = 50
    activation = 'linear'
    source = 'hcp_rs_concat'
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 100
    n_jobs = 2
    verbose = 2
    seed = 10
    _seed = 0


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
                dropout_input,
                source,
                dropout_latent,
                activation,
                datasets,
                dataset_weight,
                reduced_dir,
                batch_size,
                artifact_dir,
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

    # dump(standard_scaler, join(artifact_dir, 'standard_scaler.pkl'))

    lbins = {'dataset': {}, 'task': {}}
    Xs_train = {'dataset': {}, 'task': {}}
    Xs_test = {'dataset': {}, 'task': {}}
    y_ohs_train = {'dataset': {}, 'task': {}}
    y_ohs_test = {'dataset': {}, 'task': {}}
    pred = {'dataset': {}, 'task': {}}
    for dataset, X_dataset in X.groupby(level='dataset'):
        lbin = LabelBinarizer()
        this_y = X_dataset.index.get_level_values(level='contrast')
        lbin.fit(this_y)
        lbins['dataset'][dataset] = lbin
        Xs_train['dataset'][dataset] = X_dataset.query("fold == 'train'")
        Xs_test['dataset'][dataset] = X_dataset.query("fold == 'test'")
        lbins['task'][dataset] = {}
        Xs_train['task'][dataset] = {}
        Xs_test['task'][dataset] = {}
        for task, X_task in X_dataset.groupby(level='task'):
            lbin = LabelBinarizer()
            this_y = X_task.index.get_level_values(level='contrast')
            lbin.fit(this_y)
            lbins['task'][dataset][task] = lbin
            Xs_train['task'][dataset][task] = X_task.query("fold == 'train'")
            Xs_test['task'][dataset][task] = X_task.query("fold == 'test'")

    init_tensorflow(n_jobs=n_jobs, debug=False)

    models = make_multi_model(n_features=n_features,
                              lbins=lbins,
                              alpha=alpha,
                              latent_dim=latent_dim,
                              activation=activation,
                              dropout_input=dropout_input,
                              dropout_latent=dropout_latent,
                              seed=_seed)

    random_state = check_random_state(_seed)
    our_batch_generator_dataset = batch_generator_dataset(Xs_train,
                                                          lbins, models,
                                                          batch_size,
                                                          random_state)
    our_batch_generator_task = {
        dataset: batch_generator_task(Xs_train, lbins, models, batch_size,
                                      _seed,
                                      dataset)
        for dataset in Xs_train['dataset']}

    n_samples = 0
    n_epochs = {dataset: 0 for dataset in datasets}
    while n_samples < 1e7:
        (dataset, model, X_batch,
         y_oh_batch, new_epoch) = next(our_batch_generator_dataset)
        n_samples += X_batch.shape[0]
        model.train_on_batch(X_batch, y_oh_batch)
        if new_epoch:
            n_epochs[dataset] += 1
            X_test = Xs_test['dataset'][dataset]
            lbin = lbins['dataset'][dataset]
            contrast = lbin.inverse_transform(model.predict(X_test.values))
            true_contrast = X_test.index.get_level_values(level='contrast').values
            accuracy = np.mean(contrast == true_contrast)
            print(dataset, 'epoch %5i' % n_epochs[dataset],
                  'n_samples %6i' % n_samples,
                  'acc', accuracy)
        # task, model, X_batch, y_oh_batch, new_epoch = next(
        #     our_batch_generator_task[dataset])
        # n_samples += X_batch.shape[0]
        # model.train_on_batch(X_batch, y_oh_batch)
        # if new_epoch:
        #     X_test = Xs_test['task'][dataset][task]
        #     lbin = lbins['task'][dataset][task]
        #     contrast = lbin.inverse_transform(model.predict(X_test.values))
        #     true_contrast = X_test.index.get_level_values(level='contrast').values
        #     accuracy = np.mean(contrast == true_contrast)
        #     print(n_samples, 'condition vs task', dataset, task, 'acc', accuracy)

    for dataset in Xs_test['dataset']:
        X_dataset = Xs_test['dataset'][dataset]
        model = models['dataset'][dataset]
        lbin = lbins['dataset'][dataset]
        contrast = lbin.inverse_transform(model.predict(X_dataset.values))
        pred['dataset'][dataset] = pd.Series(data=contrast,
                                             index=X_dataset.index)
        for task, X_task in X_dataset.groupby(level='task'):
            model = models['task'][dataset][task]
            lbin = lbins['task'][dataset][task]
            contrast = lbin.inverse_transform(model.predict(X_task.values))
            pred['task'][(dataset, task)] = pd.Series(data=contrast,
                                                      index=X_task.index)
    pred_dataset = pd.concat(pred['dataset'].values())
    pred_task = pd.concat(pred['task'].values())
    pred_dataset.sort_index(inplace=True)
    pred_task.sort_index(inplace=True)
    res_dataset = test_model(pred_dataset)
    res_task = test_model(pred_task)
    _run.info['score'] = {'dataset': res_dataset,
                          'task': res_task}
    print(_run.info['score'])
    #
    # model.save(join(artifact_dir, 'model.keras'))
    # _run.add_artifact(join(artifact_dir, 'model.keras'), 'model')
