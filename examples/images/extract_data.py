from modl.datasets import get_data_dirs
from sacred import Experiment
from joblib import dump
from data import load_data, data_ing, patch_ing, make_patches
from os.path import join

extract_ex = Experiment('extract_data',
                          ingredients=[data_ing, patch_ing])

@data_ing.config
def config():
    source = 'aviris'
    gray = False
    scale = 1
    in_memory = True


@patch_ing.config
def config():
    patch_size = (16, 16)
    max_patches = 100000
    test_size = 2000
    normalize_per_channel = False

@extract_ex.automain
def run():
    image = load_data()
    data = make_patches(image)
    dump(data, join(get_data_dirs()[0], 'modl_data', 'aviris_%s.pkl' % normalize_per_channel))
