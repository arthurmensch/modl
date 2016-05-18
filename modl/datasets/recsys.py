# Author: Mathieu Blondel
# License: BSD
import os
from os.path import expanduser

import joblib
import sklearn.externals.joblib as skjoblib
from joblib import load
from sklearn.datasets.base import get_data_home as _get_data_home

from modl._utils.cross_validation import train_test_split


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


def get_recsys_data(dataset, random_state):
    if dataset in ['100k', '1m', '10m']:
        X = load_movielens(dataset)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
        X_tr = X_tr.tocsr()
        X_te = X_te.tocsr()
    elif dataset is 'netflix':
        X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
        X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))
    return X_te, X_tr