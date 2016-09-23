from sklearn.datasets import get_data_home as _get_data_home

from .hcp import get_hcp_data

def get_data_home():
    return _get_data_home().replace("scikit_learn", "modl")