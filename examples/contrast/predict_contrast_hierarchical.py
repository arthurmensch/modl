import os
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, Callback
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import load, dump
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils import gen_batches, check_random_state

from modl.datasets import get_data_dirs
from modl.hierarchical import make_model, init_tensorflow
from modl.model_selection import StratifiedGroupShuffleSplit

idx = pd.IndexSlice

predict_contrast_hierarchical = Experiment('predict_contrast_hierarchical')
collection = predict_contrast_hierarchical.path

observer = MongoObserver.create(db_name='amensch', collection=collection)


class MyCallback(Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epochs, logs={}):
        weights = self.model.get_layer('latent').get_weights()[0].flat[:10]
        print('latent', weights)
        self.weights.append(weights)


@predict_contrast_hierarchical.config
def config():
    reduced_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')
    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical')
    datasets = ['brainomics', 'la5c', 'hcp', 'archi']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5,
                     human_voice=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      human_voice=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1,
                          human_voice=1)
    train_size = None
    validation = True
    alpha = 0.0001
    latent_dim = 25
    activation = 'linear'
    dropout_input = 0.0
    dropout_latent = 0.25
    batch_size = 100
    optimizer = 'adam'
    epochs = 50
    task_prob = 0.5
    n_jobs = 3
    verbose = 2
    seed = 10
    shared_supervised = False
    steps_per_epoch = None

@predict_contrast_hierarchical.command
def test_model(prediction):
    match = prediction['true_label'] == prediction['predicted_label']
    prediction = prediction.assign(match=match)

    score = prediction['match'].groupby(level=['fold', 'dataset']).apply(
        np.mean)
    res = {}
    for fold, sub_score in score.groupby(level='fold'):
        res[fold] = {}
        for (_, dataset), this_score in sub_score.iteritems():
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
                steps_per_epoch,
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
    artifact_dir = join(artifact_dir, 'good')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if verbose:
        print('Fetch data')
    depth_weight = [0., 1 - task_prob, task_prob]
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
    X.sort_index(inplace=True)

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

    print(train, test)

    X = X.reset_index(level=['direction'], drop=True)
    X.sort_index(inplace=True)
    X = X.reset_index(drop=False)
    # Unique labels
    X['task'] = X.apply(lambda row: '_'.join([row['dataset'],
                                              row['task']]), axis=1)
    X['contrast'] = X.apply(lambda row: '_'.join([row['task'],
                                                  row['contrast']]), axis=1)
    y = np.empty((X.shape[0], 3), dtype=np.uint8)
    le_dict = {}
    for i, label in enumerate(['dataset', 'task', 'contrast']):
        le = LabelEncoder()
        y[:, i] = le.fit_transform(X[label].values)
        le_dict[label] = le

    X = X.set_index(['dataset', 'subject', 'task', 'contrast'])
    y = pd.DataFrame(data=y, columns=['dataset', 'task', 'contrast'],
                     index=X.index)

    label_pool = np.vstack({tuple(row) for index, row in y.iterrows()})
    lbin = LabelBinarizer()
    y_oh = lbin.fit_transform(y['contrast'])
    if y_oh.shape[1] == 1:
        y_oh = np.concatenate([y_oh, y_oh == 0], axis=1)
    y_oh = pd.DataFrame(index=X.index, data=y_oh)

    x_test = X.iloc[test].values
    y_test = y.iloc[test].values
    y_oh_test = y_oh.iloc[test].values

    X_train = X.iloc[train]
    y_train = y.iloc[train]
    y_oh_train = y_oh.iloc[train]

    def train_generator():
        batches_generator = {}
        indices = {}
        random_state = check_random_state(_seed)
        for dataset, sub_X in X_train.groupby(level='dataset'):
            len_dataset = sub_X.shape[0]
            indices[dataset] = random_state.permutation(len_dataset)
            batches_generator[dataset] = gen_batches(len_dataset, batch_size)
        while True:
            for dataset, sub_X in X_train.groupby(level='dataset'):
                len_dataset = sub_X.shape[0]
                sub_y = y_train.loc[dataset]
                sub_y_oh = y_oh_train.loc[dataset]
                try:
                    batch = next(batches_generator[dataset])
                    batch = indices[dataset][batch]
                except StopIteration:
                    batches_generator[dataset] = gen_batches(len_dataset,
                                                             batch_size)
                    random_state.shuffle(indices[dataset])
                    batch = next(batches_generator[dataset])
                    batch = indices[dataset][batch]
                x_batch = sub_X.iloc[batch].values
                y_batch = sub_y.iloc[batch].values
                y_oh_batch = sub_y_oh.iloc[batch].values
                sample_weight_batch = np.ones(x_batch.shape[0]) * dataset_weight[dataset]
                yield ([x_batch, y_batch], [y_oh_batch for _ in range(3)],
                       [sample_weight_batch for _ in range(3)])

    if steps_per_epoch is None:
        steps_per_epoch = X_train.shape[0] // batch_size

    init_tensorflow(n_jobs=n_jobs, debug=False)

    model = make_model(X.shape[1],
                       alpha=alpha,
                       latent_dim=latent_dim,
                       activation=activation,
                       dropout_input=dropout_input,
                       dropout_latent=dropout_latent,
                       label_pool=label_pool,
                       seed=_seed,
                       shared_supervised=shared_supervised)
    if not shared_supervised:
        for i, this_depth_weight in enumerate(depth_weight):
            if this_depth_weight == 0:
                model.get_layer('supervised_depth_%i' % i).trainable = False
    model.compile(optimizer=optimizer,
                  loss=['categorical_crossentropy'] * 3,
                  loss_weights=depth_weight,
                  metrics=['accuracy'])
    callbacks = [TensorBoard(log_dir=join(artifact_dir, 'logs'),
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)]
    model.fit_generator(train_generator(), callbacks=callbacks,
                        validation_data=([x_test, y_test],
                                         [y_oh_test] * 3),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs)

    y_pred_oh = model.predict(x=[X.values, y.values])
    # Depth selection
    _run.info['score'] = {}
    depth_name = ['full', 'dataset', 'task']
    for depth in [1, 2]:
        this_y_pred_oh = y_pred_oh[depth]
        if this_y_pred_oh.shape[1] == 2:
            this_y_pred_oh = this_y_pred_oh[:, 0]
        contrasts = y['contrast'].values
        y_pred = lbin.inverse_transform(this_y_pred_oh)
        true_contrast = le.inverse_transform(contrasts)
        predicted_contrast = le.inverse_transform(y_pred)
        prediction = pd.DataFrame({'true_label': true_contrast,
                                   'predicted_label': predicted_contrast},
                                  index=X.index)
        prediction = pd.concat([prediction.iloc[train],
                                prediction.iloc[test]],
                               names=['fold'], keys=['train', 'test'])
        prediction.to_csv(join(artifact_dir,
                               'prediction_depth_%i.csv' % depth))
        res = test_model(prediction)
        _run.info['score'][depth_name[depth]] = res
        print('Prediction at depth %s' % depth_name[depth], res)
        _run.add_artifact(join(artifact_dir,
        'prediction_depth_%i.csv'), 'prediction')
    labels = le.inverse_transform(lbin.inverse_transform(
        np.eye(len(lbin.classes_))))
    dump(labels, join(artifact_dir, 'labels.pkl'))

    # Chance level
    count_dataset = X[0].groupby(level='dataset').aggregate('count')
    count_task = X[0].groupby(level=['dataset', 'task']).aggregate('count')
    chance_level_dataset = []
    chance_level_task = []
    for label in labels:
        dataset, task = label.split('_')[:2]
        task = '_'.join([dataset, task])
        chance_level_dataset.append(count_dataset.loc[dataset] / X.shape[0])
        chance_level_task.append(count_task.loc[dataset, task] / X.shape[0])
    dump(chance_level_dataset, join(artifact_dir, 'chance_level_depth_1.pkl'))
    dump(chance_level_task, join(artifact_dir, 'chance_level_depth_2.pkl'))
    dump(X, join(artifact_dir, 'X.pkl'))
    model.save(join(artifact_dir, 'model.keras'))
    # _run.add_artifact(join(artifact_dir, 'model.keras'), 'model')
