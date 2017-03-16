import os
from os.path import join

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from modl.classification import make_loadings_extractor, FactoredLogistic
from modl.datasets import get_data_dirs
from modl.fixes import OurGridSearchCV
from modl.input_data.fmri.unmask import build_design, retrieve_components
from modl.utils.system import get_cache_dirs

predict_contrast = Experiment('predict_contrast')

artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'prediction')

observer = FileStorageObserver.create(basedir=artifact_dir)
predict_contrast.observers.append(observer)


@predict_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    datasets = ['hcp', 'archi']
    test_size = 0.1
    train_size = None
    n_subjects = 30

    alphas = np.logspace(-4, -1, 4)
    latent_dims = [30, 100, 200]  # Factored only
    activations = ['linear', 'relu']  # Factored only
    dropouts = [True, False]
    penalties = ['l1', 'l2']

    max_iter = 2
    tol = 1e-7  # Non-factored only

    standardize = True
    scale_importance = 'sqrt'

    fit_intercept = True
    identity = False

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


def concatenated_cv(cvs):
    for this_train, this_test in cvs[0]:
        train = [this_train]
        test = [this_train]
        for cv in cvs[1:]:
            this_train, this_test = next(cv)
            train.append(this_train)
            test.append(this_train)
        yield np.concatenate(train), np.concatenate(test)


@predict_contrast.automain
def run(dictionary_penalty,
        alphas,
        latent_dims,
        dropouts,
        penalties,
        activations,
        n_components_list,
        max_iter, n_jobs,
        identity,
        fit_intercept,
        n_subjects,
        scale_importance,
        standardize,
        datasets,
        datasets_dir,
        _run):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    X, masker = memory.cache(build_design)(datasets,
                                           datasets_dir,
                                           n_subjects, split=False)

    print('Split data')
    single_X = X[0].reset_index()
    splitters = []
    for idx, df in single_X.groupby(by='dataset'):
        cv = GroupKFold(n_splits=10)
        splitter = cv.split(df.index.tolist(), groups=df['subject'].values)
        splitters.append(splitter)
    cv = list(concatenated_cv(splitters))

    print('Retrieve components')
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)

    print('Transform and fit data')
    labels = X.index.get_level_values('contrast').values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    y = pd.Series(index=X.index, data=labels, name='label')

    pipeline = make_loadings_extractor(components,
                                       standardize=standardize,
                                       scale_importance=scale_importance,
                                       identity=identity,
                                       factored=True,
                                       scale_bases=True,
                                       n_jobs=n_jobs,
                                       memory=memory)
    classifier = FactoredLogistic(optimizer='adam',
                                  max_iter=max_iter,
                                  fit_intercept=fit_intercept,
                                  batch_size=200,
                                  validation_split=0,
                                  n_jobs=1)
    classifier = OurGridSearchCV(classifier,
                                 {'alpha': alphas,
                                  'latent_dim': latent_dims,
                                  'penalty': penalties,
                                  'dropout': dropouts,
                                  'activation': activations,
                                  },
                                 cv=cv,
                                 refit=False,
                                 verbose=1,
                                 n_jobs=n_jobs)
    estimator = Pipeline([('transformer', pipeline),
                          ('classifier', classifier)],
                         memory=memory)

    # Add a dataset column to the X matrix
    datasets = X.index.get_level_values('dataset').values
    dataset_encoder = LabelEncoder()
    datasets = dataset_encoder.fit_transform(datasets)
    datasets = pd.Series(index=X.index, data=datasets, name='dataset')
    X = pd.concat([X, datasets], axis=1)
    estimator.fit(X, y)

    # predicted_labels = estimator.predict(X)
    # predicted_labels = label_encoder.inverse_transform(predicted_labels)
    # labels = label_encoder.inverse_transform(labels)
    #
    # prediction = pd.DataFrame({'true_label': labels,
    #                            'predicted_label': predicted_labels},
    #                           index=X.index)

    print('Write task prediction artifacts')
    artifact_dir = join(_run.observers[0].dir, '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    # _run.add_artifact(join(artifact_dir, 'prediction.csv'),
    #                   name='prediction.csv')

    dump(label_encoder, join(artifact_dir, 'label_encoder.pkl'))
    dump(estimator, join(artifact_dir, 'estimator.pkl'))
