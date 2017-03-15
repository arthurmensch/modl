import os
from os.path import join

import numpy as np
import pandas as pd
from modl.classification import make_loadings_extractor, make_classifier
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

predict_contrast = Experiment('predict_contrast')

artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'prediction')

observer = FileStorageObserver.create(basedir=artifact_dir)
predict_contrast.observers.append(observer)


def retrieve_components(alpha, masker, n_components_list):
    components_dir = join(get_data_dirs()[0], 'pipeline', 'components', 'hcp')
    components_imgs = [join(components_dir, str(this_n_components), str(alpha),
                            'components.nii.gz')
                       for this_n_components in n_components_list]
    components = masker.transform(components_imgs)
    return components


@predict_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    datasets = ['hcp', 'archi']
    test_size = 0.1
    train_size = None
    n_subjects = 788

    factored = False

    alphas = np.logspace(-4, -1, 4)
    latent_dims = [100]  # Factored only
    activation = 'relu'  # Factored only
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
        alphas,
        latent_dims,
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
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch and split data')
    X, masker = memory.cache(build_design)(datasets,
                                           datasets_dir,
                                           n_subjects, test_size,
                                           train_size,
                                           random_state=_seed)
    train_samples = X.loc['train'].shape[0]
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
                                       factored=factored,
                                       scale_bases=True,
                                       n_jobs=n_jobs,
                                       memory=memory)
    classifier = make_classifier(alphas, latent_dims,
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
        # Add a dataset column to the X matrix
        datasets = X.index.get_level_values('dataset').values
        dataset_encoder = LabelEncoder()
        datasets = dataset_encoder.fit_transform(datasets)
        datasets = pd.Series(index=X.index, data=datasets, name='dataset')
        X = pd.concat([X, datasets], axis=1)
        X_train = X.loc['train']
        y_train = y.loc['train']
        estimator.fit(X_train, y_train)
    else:
        X_train = X.loc['train']
        y_train = y.loc['train']
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
    artifact_dir = join(_run.observers[0].dir, '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'),
                      name='prediction.csv')

    dump(label_encoder, join(artifact_dir, 'label_encoder.pkl'))
    dump(estimator, join(artifact_dir, 'estimator.pkl'))
