import os
from os.path import join

import pandas as pd

from modl.classification import make_loadings_extractor
from modl.hierarchical import make_projection_matrix
from modl.datasets import get_data_dirs
from modl.input_data.fmri.unmask import build_design, retrieve_components, \
    get_raw_contrast_data
from modl.utils.system import get_cache_dirs
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump
from sklearn.preprocessing import LabelEncoder

reduce_contrast = Experiment('reduce_contrast')



# observer = MongoObserver.create(db_name='amensch',
#                                collection='reduce_contrast')
# reduce_contrast.observers.append(observer)


@reduce_contrast.config
def config():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    dataset = 'human_voice'

    n_jobs = 24
    verbose = 2

    output_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')
    dataset_dir = join(get_data_dirs()[0], 'pipeline', 'unmask', 'contrast')


@reduce_contrast.automain
def run(dictionary_penalty,
        n_components_list,
        output_dir,
        dataset_dir,
        dataset):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    this_dataset_dir = join(dataset_dir, dataset)
    masker, X = get_raw_contrast_data(this_dataset_dir)
    print('Retrieve components')
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)

    print('Transform and fit data')
    proj = memory.cache(make_projection_matrix)(components, scale_bases=True)
    Xt = X.dot(proj)
    Xt = pd.DataFrame(data=Xt, index=X.index)
    this_output_dir = join(output_dir, dataset)
    if not os.path.exists(this_output_dir):
        os.makedirs(this_output_dir)
    dump(Xt, join(this_output_dir, 'Xt.pkl'))
    dump(masker, join(this_output_dir, 'masker.pkl'))
