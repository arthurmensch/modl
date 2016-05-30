from sklearn.datasets.base import get_data_home as _get_data_home

def get_data_home():
    return _get_data_home().replace("scikit_learn", "modl")

from .hcp import get_hcp_data
from .recsys import get_recsys_data
