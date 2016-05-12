# Author: Mathieu Blondel
# License: BSD
import os

import joblib
import sklearn.externals.joblib as skjoblib
from sklearn.datasets.base import get_data_home as _get_data_home


def get_data_home():
    return _get_data_home().replace("scikit_learn", "modl")


def load_movielens(version):
    data_home = get_data_home()

    if version == "100k":
        path = os.path.join(data_home, "movielens100k", "movielens100k.pkl")
    elif version == "1m":
        path = os.path.join(data_home, "movielens1m", "movielens1m.pkl")
    elif version == "10m":
        path = os.path.join(data_home, "movielens10m", "movielens10m.pkl")
    else:
        raise ValueError("Invalid version of movielens.")

    # FIXME: make downloader
    if not os.path.exists(path):
        raise ValueError("Dowload dataset using 'make download-movielens%s' at"
                         " project root." % version)

    X = skjoblib.load(path)
    return X

def load_netflix():
    data_home = get_data_home()
    path = os.path.join(data_home, "nf_prize", "X_tr.pkl")
    X_tr = joblib.load(path)
    path = os.path.join(data_home, "nf_prize", "X_te.pkl")
    X_te = joblib.load(path)
    return X_tr, X_te