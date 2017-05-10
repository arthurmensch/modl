import os
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
from modl.datasets import get_data_dirs
from modl.hierarchical import make_model, init_tensorflow, make_adversaries, \
    make_multi_model
from modl.model_selection import StratifiedGroupShuffleSplit
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import load, dump
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import gen_batches, check_random_state

idx = pd.IndexSlice

predict_contrast_exp = Experiment('predict_contrast')
collection = predict_contrast_exp.path

observer = MongoObserver.create(db_name='amensch', collection=collection)


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
        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train)
        X_new = standard_scaler.transform(X)
        X = pd.DataFrame(X_new, index=X.index)
    return X, standard_scaler


def simple_generator(train_data, batch_size):
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    y_oh_train = train_data['y_oh_train']
    len_x = X_train.shape[0]
    while True:
        permutation = check_random_state(0).permutation(len_x)
        X_train = X_train.iloc[permutation]
        y_train = y_train.iloc[permutation]
        y_oh_train = y_oh_train.iloc[permutation]
        batches = gen_batches(len_x, batch_size)
        for batch in batches:
            yield [X_train.values[batch], y_train.values[batch]], [
                y_oh_train.values[batch]] * 3


def train_generator(train_data, batch_size, dataset_weight,
                    mix, seed):
    random_state = check_random_state(seed)
    grouped_data = train_data.groupby(level='dataset')
    grouped_data = {dataset: sub_data for dataset, sub_data in
                    grouped_data}
    batches_generator = {}
    n_dataset = len(grouped_data)
    x_batch = np.empty((batch_size, train_data['X'].shape[1]))
    y_batch = np.empty((batch_size, train_data['y'].shape[1]))
    y_oh_batch = np.empty((batch_size, train_data['y_oh'].shape[1]))
    sample_weight_batch = np.empty(batch_size)
    while True:
        start = 0
        for dataset, data_one_dataset in grouped_data.items():
            if not mix:
                start = 0
            len_dataset = data_one_dataset.shape[0]
            try:
                batch = next(batches_generator[dataset])
            except (KeyError, StopIteration):
                batches_generator[dataset] = gen_batches(len_dataset,
                                                         batch_size
                                                         // n_dataset if mix
                                                         else batch_size)
                permutation = random_state.permutation(len_dataset)
                data_one_dataset = data_one_dataset.iloc[permutation]
                grouped_data[dataset] = data_one_dataset
                batch = next(batches_generator[dataset])
            len_batch = batch.stop - batch.start
            stop = start + len_batch
            batch_data = data_one_dataset.iloc[batch]
            x_batch[start:stop] = batch_data['X'].values
            y_batch[start:stop] = batch_data['y'].values
            y_oh_batch[start:stop] = batch_data['y_oh'].values
            sample_weight_batch[start:stop] = np.ones(len_batch) * \
                                              dataset_weight[dataset]
            start = stop
            if not mix:
                yield ([x_batch[:stop].copy(), y_batch[:stop].copy()],
                       [y_oh_batch[:stop].copy()] * 3,
                       [sample_weight_batch[:stop]] * 3)
        if mix:
            yield ([x_batch[:stop].copy(), y_batch[:stop].copy()],
                   [y_oh_batch[:stop].copy()] * 3,
                   [sample_weight_batch[:stop]] * 3)


class MyCallback(Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epochs, logs={}):
        weights = self.model.get_layer('latent').get_weights()[0].flat[:10]
        self.weights.append(weights)


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
    alpha = 0
    latent_dim = 100
    activation = 'linear'
    source = 'hcp_rs_concat'
    optimizer = 'sgd'
    lr = 1e-2
    dropout_input = 0.0
    dropout_latent = 0.6
    batch_size = 128
    per_dataset_std = False
    joint_training = True
    epochs = 100
    depth_weight = [0., 1., 0.]
    n_jobs = 2
    verbose = 2
    seed = 10
    shared_supervised = True
    mix_batch = False
    steps_per_epoch = None
    _seed = 0


@predict_contrast_exp.named_config
def no_geometric():
    datasets = ['camcan']
    validation = False
    geometric_reduction = False
    alpha = 1
    latent_dim = None
    activation = 'linear'
    dropout_input = 0.
    dropout_latent = 0.
    batch_size = 300
    per_dataset_std = False
    joint_training = True
    optimizer = 'sgd'
    epochs = 15
    depth_weight = [0., 1., 0.]
    n_jobs = 2
    verbose = 2
    seed = 10
    shared_supervised = False
    mix_batch = False
    steps_per_epoch = None
    _seed = 0


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


@predict_contrast_exp.automain
def train_model(alpha,
                latent_dim,
                n_subjects,
                unmask_dir,
                geometric_reduction,
                test_size,
                train_size,
                dropout_input,
                joint_training,
                lr,
                mix_batch,
                source,
                dropout_latent,
                optimizer,
                activation,
                datasets,
                per_dataset_std,
                dataset_weight,
                steps_per_epoch,
                depth_weight,
                reduced_dir,
                batch_size,
                artifact_dir,
                epochs,
                verbose,
                shared_supervised,
                validation,
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

    X, standard_scaler = scale(X, train, per_dataset_std)
    dump(standard_scaler, join(artifact_dir, 'standard_scaler.pkl'))

    lbins = {'dataset': {}, 'task': {}}
    Xs_train = {'dataset': {}, 'task': {}}
    Xs_test = {'dataset': {}, 'task': {}}
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
            lbins['task'][dataset][task] = lbin
            Xs_train['task'][dataset][task] = X_task.query("fold == 'train'")
            Xs_test['task'][dataset][task] = X_task.query("fold == 'test'")

    models = make_multi_model(X,
                              alpha=alpha,
                              latent_dim=latent_dim,
                              activation=activation,
                              dropout_input=dropout_input,
                              dropout_latent=dropout_latent,
                              seed=_seed)
    n_iter = 0

    def batch_generator_dataset():
        batch_generators = {}
        random_state = check_random_state(_seed)
        while True:
            for dataset in Xs_train['dataset']:
                lbin = lbins['dataset'][dataset]
                X_dataset = Xs_train['dataset'][dataset]
                model = models['dataset'][dataset]
                try:
                    batch = next(batch_generators[dataset])
                except (KeyError, StopIteration):
                    len_dataset = X_dataset.shape[0]
                    random_state.shuffle(X_dataset)
                    batch_generators[dataset] = gen_batches(len_dataset,
                                                            batch_size)
                    batch = next(batch_generators[dataset])
                X_batch = X_dataset.iloc[batch]
                y_oh_batch = X_batch.index.get_level_values(level='contrast')
                y_oh_batch = lbin.transform(y_oh_batch)
                X_batch = X_batch.values
                yield dataset, model, X_batch, y_oh_batch

    def batch_generator_task(dataset):
        batch_generators = {}
        random_state = check_random_state(_seed)
        while True:
            for task in Xs_train['task'][dataset]:
                lbin = lbins['task'][dataset][task]
                X_task = Xs_train['task'][dataset][task]
                model = models['task'][dataset][task]
                try:
                    batch = next(batch_generators[task])
                except (KeyError, StopIteration):
                    len_task = X_task.shape[0]
                    random_state.shuffle(X_task)
                    batch_generators[task] = gen_batches(len_task,
                                                            batch_size)
                    batch = next(batch_generators[task])
                X_batch = X_task.iloc[batch]
                y_oh_batch = X_batch.index.get_level_values(level='contrast')
                y_oh_batch = lbin.transform(y_oh_batch)
                X_batch = X_batch.values
                yield model, X_batch, y_oh_batch

    our_batch_generator_dataset = batch_generator_dataset()
    our_batch_generator_task = {dataset: batch_generator_task(dataset)
                                for dataset in Xs_train['dataset']}

    while n_iter < 1000:
        dataset, model, X_batch, y_oh_batch = next(our_batch_generator_dataset)
        model.train_on_batch(X_batch, y_oh_batch)
        n_iter += 1
        task, model, X_batch, y_oh_batch = next(our_batch_generator_task[dataset])
        model.train_on_batch(X_batch, y_oh_batch)
        n_iter += 1

    for dataset in Xs_test['dataset']:
        X_dataset = Xs_test['dataset'][dataset]
        lbins['dataset'][dataset] = lbin
        Xs_train['dataset'][dataset] = X_dataset.query("fold == 'train'")
        Xs_test['dataset'][dataset] = X_dataset.query("fold == 'test'")
        lbins['task'][dataset] = {}
        Xs_train['task'][dataset] = {}
        Xs_test['task'][dataset] = {}
        for task, X_task in X_dataset.groupby(level='task'):

    y_pred_oh = model.predict(x=[X.values, y.values])

    _run.info['score'] = {}
    depth_name = ['full', 'dataset', 'task']
    for depth in [0, 1, 2]:
        this_y_pred_oh = y_pred_oh[depth]
        this_y_pred_oh_df = pd.DataFrame(index=X.index,
                                         data=this_y_pred_oh)
        dump(this_y_pred_oh_df, join(artifact_dir,
                                     'y_pred_depth_%i.pkl' % depth))
        y_pred = lbin.inverse_transform(this_y_pred_oh)  # "0_0_0"
        prediction = pd.DataFrame({'true_label': y_tuple,
                                   'predicted_label': y_pred},
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
                               'prediction_depth_%i.csv' % depth),
                          'prediction')

    model.save(join(artifact_dir, 'model.keras'))
    _run.add_artifact(join(artifact_dir, 'model.keras'), 'model')
