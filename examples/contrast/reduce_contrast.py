import os
from os.path import join

import pandas as pd
from modl.datasets import get_data_dirs
from modl.hierarchical import make_projection_matrix
from modl.input_data.fmri.unmask import retrieve_components, \
    get_raw_contrast_data
from modl.utils.system import get_cache_dirs
from nilearn.datasets import fetch_atlas_msdl
from sacred import Experiment
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import dump

import numpy as np

reduce_contrast = Experiment('reduce_contrast')


@reduce_contrast.config
def config():
    dictionary_penalty = 1e-4

    dataset = 'archi'

    n_jobs = 24
    verbose = 2

    source = 'hcp_rs_concat'

    output_dir = join(get_data_dirs()[0], 'pipeline', 'contrast')
    dataset_dir = join(get_data_dirs()[0], 'pipeline', 'unmask', 'contrast')


@reduce_contrast.automain
def run(dictionary_penalty,
        output_dir,
        dataset_dir,
        source,
        dataset):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    this_dataset_dir = join(dataset_dir, dataset)
    masker, X = get_raw_contrast_data(this_dataset_dir)
    print('Retrieve components')
    if source == 'msdl':
        components = fetch_atlas_msdl()['maps']
        proj = masker.transform(components).T
    elif source in ['hcp_rs', 'hcp_rs_concat']:
        if source == 'hcp_rs':
            n_components_list = [64]
        else:
            n_components_list = [16, 64, 256]
        components = memory.cache(retrieve_components)(dictionary_penalty,
                                                       masker,
                                                       n_components_list)

        print('Transform and fit data')
        proj, _ = memory.cache(make_projection_matrix)(components,
                                                       scale_bases=True, )
    Xt = X.dot(proj)
    Xt = pd.DataFrame(data=Xt, index=X.index)
    this_output_dir = join(output_dir, source, dataset)
    if not os.path.exists(this_output_dir):
        os.makedirs(this_output_dir)
    dump(Xt, join(this_output_dir, 'Xt.pkl'))
    dump(masker, join(this_output_dir, 'masker.pkl'))
