import os
from os.path import join

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import load
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from modl.datasets import get_data_dirs
from modl.hierarchical import make_model, init_tensorflow
from modl.model_selection import StratifiedGroupShuffleSplit

idx = pd.IndexSlice

predict_contrast_hierarchical = Experiment('predict_contrast_hierarchical')
collection = predict_contrast_hierarchical.path

observer = MongoObserver.create(db_name='amensch', collection=collection)


@predict_contrast_hierarchical.config
def config():
    reduced_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')
    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical')
    datasets = ['hcp', 'la5c']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1)
    train_size = None
    validation = True
    alpha = 0.0001
    latent_dim = 50
    activation = 'linear'
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 200
    optimizer = 'adam'
    epochs = 30
    task_prob = 1.
    n_jobs = 2
    verbose = 2
    seed = 10
    shared_supervised = False


@predict_contrast_hierarchical.command
def test_model(prediction):
    match = prediction['true_label'] == prediction['predicted_label']
    prediction = prediction.assign(match=match)

    score = prediction['match'].groupby(level=['fold', 'dataset']).apply(
        np.mean)
    res = {}
    for fold, sub_score in score.groupby(level='fold'):
        res[fold] = {}
        for dataset, this_score in sub_score.iteritems():
            res[fold][dataset] = this_score
    return res


@predict_contrast_hierarchical.automain
def train_model(alpha,
                latent_dim,
                n_subjects,
                test_size,
                train_size,
                dropout_input,
                dropout_latent,
                activation,
                datasets,
                dataset_weight,
                task_prob,
                reduced_dir,
                optimizer,
                batch_size,
                artifact_dir,
                epochs,
                verbose,
                shared_supervised,
                n_jobs,
                _run,
                _seed):
    artifact_dir = join(artifact_dir, str(_run._id))
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if verbose:
        print('Fetch data')

    depth_probs = [0., 1 - task_prob, task_prob]

    X = []
    keys = []
    for dataset in datasets:
        if dataset_weight[dataset] != 0:
            this_reduced_dir = join(reduced_dir, dataset)
            this_X = load(join(this_reduced_dir, 'Xt.pkl'), mmap_mode='r')
            subjects = this_X.index.get_level_values('subject'). \
                unique().values.tolist()
            subjects = subjects[:n_subjects[dataset]]
            this_X = this_X.loc[idx[subjects]]
            X.append(this_X)
            keys.append(dataset)
    X = pd.concat(X, keys=keys, names=['dataset'])
    level_names = X.index.names
    X = X.reset_index()
    X['contrast'] = X.apply(lambda row: '_'.join([row['dataset'],
                                                 row['task'],
                                                 row['contrast']]),
                            axis=1)
    X = X.set_index(level_names)
    X.sort_index(inplace=True)

    # Building numpy array
    y = np.zeros((X.shape[0], 3), dtype=np.int32)
    le_dict = {}
    for i, label in enumerate(['dataset', 'task', 'contrast']):
        le = LabelEncoder()
        y[:, i] = le.fit_transform(X.index.get_level_values(label).values)
        le_dict[label] = le

    label_pool = np.vstack({tuple(row) for row in y})
    indices = np.argsort(label_pool[:, -1])
    label_pool = label_pool[indices]
    # d is only used at test time. 1: per dataset. 2: per task
    d = np.ones(y.shape[0], dtype=np.int32)

    lbin = LabelBinarizer()
    contrasts = y[:, -1]
    contrasts_oh = lbin.fit_transform(contrasts)
    n_features = X.shape[1]
    x = X.values

    sample_weight = []
    keys = []
    for dataset, sub_X in X.iloc[:, 0].groupby(level='dataset'):
        length = sub_X.shape[0]
        sample_weight.append(
            pd.Series(data=dataset_weight[dataset] / length,
                      index=sub_X.index))
        keys.append(dataset)
    sample_weight = pd.concat(sample_weight, names=['dataset'], keys=keys)
    sample_weight *= X.shape[0] / np.sum(sample_weight)

    # Cross validation folds
    cv = StratifiedGroupShuffleSplit(stratify_levels='dataset',
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=train_size,
                                     n_splits=1,
                                     random_state=0)
    train, test = next(cv.split(X))

    _run.info['train_fold'] = train.tolist()
    _run.info['test_fold'] = test.tolist()

    init_tensorflow(n_jobs=n_jobs, debug=False)

    model = make_model(n_features,
                       alpha=alpha,
                       latent_dim=latent_dim,
                       activation=activation,
                       dropout_input=dropout_input,
                       dropout_latent=dropout_latent,
                       label_pool=label_pool,
                       seed=_seed,
                       depth_weight=depth_probs,
                       shared_supervised=shared_supervised)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=[x[train], y[train], d[train]],
              y=contrasts_oh[train], sample_weight=sample_weight[train],
              batch_size=batch_size,
              epochs=epochs)
    # model.get_layer('pool').
    # use only y[:, :d] as input
    y_pred_oh = model.predict(x=[x, y, d])

    y_pred = lbin.inverse_transform(y_pred_oh)
    true_contrast = le_dict['contrast'].inverse_transform(contrasts)
    predicted_contrast = le_dict['contrast'].inverse_transform(y_pred)
    prediction = pd.DataFrame({'true_label': true_contrast,
                               'predicted_label': predicted_contrast},
                              index=X.index)
    prediction = pd.concat([prediction.iloc[train],
                            prediction.iloc[test]],
                           names=['fold'], keys=['train', 'test'])
    prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'), 'prediction')
    model.save(join(artifact_dir, 'model.keras'))
    _run.add_artifact(join(artifact_dir, 'model.keras'), 'model')

    res = test_model(prediction)
    print('Prediction score', res)
    _run.info['score'] = res
