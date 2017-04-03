import os
from os.path import join

import pandas as pd

from modl.classification import make_loadings_extractor
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design, retrieve_components
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.preprocessing import LabelEncoder

predict_contrast = Experiment('reduce_contrast')

loadings_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')

observer = MongoObserver.create(db_name='amensch',
                                collection='reduce_contrast')
predict_contrast.observers.append(observer)


@predict_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    datasets = ['hcp', 'archi']
    n_subjects = 788

    standardize = True
    scale_importance = 'sqrt'

    identity = False

    projection = True

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
        identity,
        n_subjects,
        scale_importance,
        standardize,
        datasets,
        projection,
        datasets_dir,
        _run):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    X, masker = memory.cache(build_design)(datasets,
                                           datasets_dir,
                                           n_subjects)
    X = X.astype('double')
    print('Retrieve components')
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)

    print('Transform and fit data')
    labels = X.index.get_level_values('contrast').values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    y = pd.Series(index=X.index, data=labels, name='label')

    if projection:
        pipeline = make_loadings_extractor(components,
                                           standardize=standardize,
                                           scale_importance=scale_importance,
                                           identity=identity,
                                           scale_bases=True,
                                           n_jobs=n_jobs,
                                           memory=memory)

    datasets = X.index.get_level_values('dataset').values
    datasets = pd.Series(index=X.index, data=datasets, name='dataset')
    X = pd.concat([X, datasets], axis=1)

    if projection:
        Xt = pipeline.fit_transform(X, y)
        Xt = pd.DataFrame(data=Xt, index=X.index,
                          columns=list(range(Xt.shape[1] - 1)) + ['dataset']
                          )
    else:
        Xt = X
    this_loadings_dir = join(loadings_dir, str(projection))
    if not os.path.exists(this_loadings_dir):
        os.makedirs(this_loadings_dir)
    dump(Xt, join(this_loadings_dir, 'Xt.pkl'))
    dump(y, join(this_loadings_dir, 'y.pkl'))
    dump(masker, join(this_loadings_dir, 'masker.pkl'))
    dump(label_encoder, join(this_loadings_dir,
                             'label_encoder.pkl'))
