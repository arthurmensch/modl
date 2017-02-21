import os

from os.path import join

import pandas as pd
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder

from modl.classification import OurLogisticRegressionCV, \
    make_loadings_extractor
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import get_raw_contrast_data
from modl.utils.system import get_cache_dirs

from sklearn.externals.joblib import dump

predict_contrast = Experiment('predict_contrast')
observer = MongoObserver.create(db_name='amensch', collection='runs')
predict_contrast.observers.append(observer)


@predict_contrast.config
def config():
    alphas = np.logspace(-4, 4, 20).tolist()
    n_jobs = 30
    verbose = 2
    seed = 2
    max_iter = 3000
    tol = 1e-7
    alpha = 1e-4
    multi_class = 'ovr'
    fit_intercept = False
    refit = True
    n_components_list = [16, 64, 256]
    test_size = 0.1
    train_size = None
    n_subjects = 788
    penalty = 'l2'
    solver = 'saga'


@predict_contrast.automain
def run(alphas, tol,
        n_components_list,
        alpha,
        max_iter, n_jobs,
        test_size,
        solver,
        train_size,
        verbose,
        fit_intercept,
        multi_class,
        n_subjects,
        refit,
        penalty,
        _run,
        _seed):
    memory = Memory(cachedir=get_cache_dirs()[0])

    unmask_contrast_dir = join(get_data_dirs()[0], 'pipeline',
                               'unmask', 'contrast', 'hcp')

    _run.info['resource_dir'] = {'unmask_contrast': unmask_contrast_dir}

    print('Fetch data')
    masker, X = get_raw_contrast_data(unmask_contrast_dir)

    subjects = X.index.get_level_values('subject').unique().values.tolist()

    subjects = subjects[:n_subjects]
    X = X.loc[subjects]

    labels = X.index.get_level_values('contrast').values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y = pd.Series(index=X.index, data=y, name='label')

    print('Retrieve components')
    components_dir = join(get_data_dirs()[0], 'pipeline', 'components', 'hcp')
    components_imgs = [join(components_dir, str(this_n_components), str(alpha),
                            'components.nii.gz')
                       for this_n_components in n_components_list]

    components = masker.transform(components_imgs)

    print('Split data')
    train_subjects, test_subjects = \
        train_test_split(subjects, random_state=_seed, test_size=test_size)
    train_subjects = train_subjects[:train_size]
    _run.info['train_subject'] = train_subjects
    _run.info['test_subjects'] = test_subjects

    X = pd.concat([X.loc[train_subjects],
                   X.loc[test_subjects]], keys=['train', 'test'],
                  names=['fold'])
    y = pd.concat([y.loc[train_subjects],
                   y.loc[test_subjects]], keys=['train', 'test'],
                  names=['fold'])

    classifier = OurLogisticRegressionCV(
        solver=solver,
        memory=memory,
        memory_level=2,
        alphas=alphas,
        penalty=penalty,
        fit_intercept=fit_intercept,
        multi_class=multi_class,
        refit=refit,
        random_state=_seed,
        tol=tol,
        max_iter=max_iter,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    pipeline = make_loadings_extractor(components,
                                       standardize=True,
                                       scale_importance=True,
                                       scale_bases=True,
                                       n_jobs=n_jobs,
                                       memory=memory)
    pipeline.append(('logistic_regression', classifier))
    estimator = Pipeline(pipeline, memory=memory)
    print('Transform and fit data')
    estimator.fit(X.loc['train'].values, y.loc['train'].values)

    predicted_y = estimator.predict(X.values)
    predicted_labels = label_encoder.inverse_transform(predicted_y)
    true_labels = label_encoder.inverse_transform(y.values)
    prediction = pd.DataFrame(data=list(zip(true_labels, predicted_labels)),
                              columns=['true_label', 'predicted_label'],
                              index=X.index)

    print('Compute score')
    train_score = np.sum(prediction.loc['train', 'predicted_label']
                         == prediction.loc['train', 'true_label'])
    train_score /= prediction.loc['train'].shape[0]

    _run.info['train_score'] = train_score

    test_score = np.sum(prediction.loc['test', 'predicted_label']
                        == prediction.loc['test', 'true_label'])
    test_score /= prediction.loc['test'].shape[0]

    _run.info['test_score'] = test_score

    _run.info['train_score'] = train_score
    _run.info['test_score'] = test_score
    print('Write task prediction artifacts')
    artifact_dir = join(get_data_dirs()[0], 'pipeline',
                        'contrast', 'prediction', str(_run._id))
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    prediction.to_csv(join(artifact_dir, 'prediction.csv'))
    _run.add_artifact(join(artifact_dir, 'prediction.csv'),
                      name='prediction.csv')

    dump(label_encoder, join(artifact_dir, 'label_encoder.pkl'))
    dump(estimator, join(artifact_dir, 'estimator.pkl'))
