import os
from os.path import join

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from modl.utils import concatenated_cv
from modl.classification import make_loadings_extractor
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design, retrieve_components
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.preprocessing import LabelEncoder

predict_contrast = Experiment('predict_contrast')

artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')

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

    standardize = True
    scale_importance = 'sqrt'
    multi_class = 'multinomial'  # Non-factored only

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


@predict_contrast.automain
def run(dictionary_penalty,
        n_components_list,
        n_jobs,
        test_size,
        train_size,
        identity,
        n_subjects,
        scale_importance,
        standardize,
        datasets,
        datasets_dir,
        _run,
        _seed):
    artifact_dir = join(_run.observers[0].dir, '_artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    X, masker = memory.cache(build_design)(datasets,
                                           datasets_dir,
                                           n_subjects)

    print('Split data')
    single_X = X[0].reset_index()
    splitters = []
    for idx, df in single_X.groupby(by='dataset'):
        cv = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               train_size=train_size, random_state=_seed)
        splitter = cv.split(df.index.tolist(), groups=df['subject'].values)
        splitters.append(splitter)
    train, test = next(concatenated_cv(splitters))
    X = pd.concat([X.iloc[train], X.iloc[test]], keys=['train', 'test'],
                  names=['fold'])
    X.sort_index(inplace=True)

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

    datasets = X.index.get_level_values('dataset').values
    dataset_encoder = LabelEncoder()
    datasets = dataset_encoder.fit_transform(datasets)
    datasets = pd.Series(index=X.index, data=datasets, name='dataset')
    X = pd.concat([X, datasets], axis=1)

    Xt = pipeline.fit_transform(X, y)
    Xt = pd.DataFrame(data=Xt, index=X.index)

    dump(Xt, join(artifact_dir, 'Xt.pkl'))
    dump(y, join(artifact_dir, 'y.pkl'))
    dump(pipeline, join(artifact_dir, 'pipeline.pkl'))
    _run.add_artifact(join(artifact_dir, 'Xt.pkl'),
                      name='Xt.pkl')
    _run.add_artifact(join(artifact_dir, 'y.pkl'),
                      name='y.pkl')
    _run.add_artifact(join(artifact_dir, 'pipeline.pkl'),
                      name='pipeline.pkl')
