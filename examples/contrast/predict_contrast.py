import os
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam
from modl.datasets import get_data_dirs
from modl.hierarchical import make_model, init_tensorflow, make_adversaries
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
        standard_scaler = StandardScaler(with_std=False)
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
    alpha = 1e-4
    latent_dim = 50
    activation = 'linear'
    source = 'hcp_rs_concat'
    optimizer = 'adam'
    lr = 1e-3
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 100
    per_dataset_std = False
    joint_training = True
    epochs = 50
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
                this_X = load(join(this_reduced_dir, 'Xt.pkl'))
            else:
                this_X = load(join(unmask_dir, dataset, 'imgs.pkl'))
            if dataset in ['archi', 'brainomics']:
                this_X = this_X.drop(['effects_of_interest'],
                                     level='contrast', )
            subjects = this_X.index.get_level_values('subject'). \
                unique().values.tolist()
            subjects = subjects[:n_subjects[dataset]]
            this_X = this_X.loc[idx[subjects]]
            X.append(this_X)
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

    # X, standard_scaler = scale(X, train, per_dataset_std)
    # dump(standard_scaler, join(artifact_dir, 'standard_scaler.pkl'))

    y = np.concatenate([X.index.get_level_values(level)[:, np.newaxis]
                        for level in ['dataset', 'task', 'contrast']],
                       axis=1)

    y_tuple = ['__'.join(row) for row in y]
    lbin = LabelBinarizer()
    y_oh = lbin.fit_transform(y_tuple)
    label_pool = lbin.classes_
    label_pool = [np.array(e.split('__')) for e in label_pool]
    label_pool = np.vstack(label_pool)
    y = np.argmax(y_oh, axis=1)
    y_oh = pd.DataFrame(index=X.index, data=y_oh)
    y = pd.DataFrame(index=X.index, data=y)
    dump(lbin, join(artifact_dir, 'lbin.pkl'))

    x_test = X.iloc[test]
    y_test = y.iloc[test]
    y_oh_test = y_oh.iloc[test]

    sample_weight_test = []
    for dataset, this_x in x_test.groupby(level='dataset'):
        sample_weight_test.append(pd.Series(np.ones(this_x.shape[0])
                                            / this_x.shape[0]
                                            * dataset_weight[dataset],
                                            index=this_x.index))
    sample_weight_test = pd.concat(sample_weight_test, axis=0)
    sample_weight_test /= np.min(sample_weight_test)

    x_test = x_test.values
    y_test = y_test.values
    y_oh_test = y_oh_test.values
    sample_weight_test = sample_weight_test.values

    X_train = X.iloc[train]
    y_train = y.iloc[train]
    y_oh_train = y_oh.iloc[train]

    train_data = pd.concat([X_train, y_train, y_oh_train],
                           keys=['X', 'y', 'y_oh'],
                           names=['type'], axis=1)
    train_data.sort_index(inplace=True)

    if steps_per_epoch is None:
        steps_per_epoch = X_train.shape[0] // batch_size

    init_tensorflow(n_jobs=n_jobs, debug=False)

    adversaries = make_adversaries(label_pool)

    np.save(join(artifact_dir, 'adversaries'), adversaries)
    np.save(join(artifact_dir, 'classes'), lbin.classes_)

    model = make_model(X.shape[1],
                       alpha=alpha,
                       latent_dim=latent_dim,
                       activation=activation,
                       dropout_input=dropout_input,
                       dropout_latent=dropout_latent,
                       adversaries=adversaries,
                       seed=_seed,
                       shared_supervised=shared_supervised)
    if not shared_supervised:
        for i, this_depth_weight in enumerate(depth_weight):
            if this_depth_weight == 0:
                model.get_layer('supervised_depth_%i' % i).trainable = False
    if optimizer == 'sgd':
        optimizer = SGD(lr=lr)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr)
    model.compile(loss=['categorical_crossentropy'] * 3,
                  optimizer=optimizer,
                  loss_weights=depth_weight,
                  metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=1e-5)
    callbacks = [TensorBoard(log_dir=join(artifact_dir, 'logs'),
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True),
                 reduce_lr
                 ]
    if joint_training:
        model.fit_generator(train_generator(train_data,
                                            batch_size,
                                            dataset_weight=dataset_weight,
                                            mix=mix_batch,
                                            seed=_seed),
                            callbacks=callbacks,
                            validation_data=([x_test, y_test],
                                             [y_oh_test] * 3,
                                             [sample_weight_test] * 3
                                             ) if validation else None,
                            steps_per_epoch=steps_per_epoch,
                            verbose=verbose,
                            epochs=epochs)
    else:
        model.fit_generator(
            train_generator(train_data.loc[['hcp']], batch_size,
                            dataset_weight=dataset_weight,
                            mix=False,
                            seed=_seed),
            callbacks=callbacks,
            validation_data=([x_test, y_test],
                             [y_oh_test] * 3,
                             [sample_weight_test] * 3
                             ) if validation else None,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            epochs=epochs - 10)
        model.get_layer('latent').trainable = False
        model.compile(loss=['categorical_crossentropy'] * 3,
                      optimizer=optimizer,
                      loss_weights=depth_weight,
                      metrics=['accuracy'])
        model.fit_generator(train_generator(train_data, batch_size,
                            dataset_weight=dataset_weight,
                            mix=False,
                            seed=_seed),
                            callbacks=callbacks,
                            validation_data=([x_test, y_test],
                                             [y_oh_test] * 3,
                                             [sample_weight_test] * 3,
                                             ) if validation else None,
                            steps_per_epoch=steps_per_epoch,
                            verbose=verbose,
                            initial_epoch=epochs - 10,
                            epochs=epochs)
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
