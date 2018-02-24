# Author: Mathieu Blondel, Arthur Mensch
# From spira
# License: BSD
import os

from sklearn.externals.joblib import load
from modl.utils.recsys.cross_validation import train_test_split

from modl.datasets import get_data_dirs


def load_movielens(version):
    data_home = get_data_dirs()[0]

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

    X = load(path)
    return X


def load_netflix():
    data_home = get_data_dirs()[0]
    path = os.path.join(data_home, "nf_prize", "X_tr.pkl")
    X_tr = load(path)
    path = os.path.join(data_home, "nf_prize", "X_te.pkl")
    X_te = load(path)
    return X_tr, X_te


def load_recsys(dataset, random_state):
    if dataset in ['100k', '1m', '10m']:
        X = load_movielens(dataset)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
        X_tr = X_tr.tocsr()
        X_te = X_te.tocsr()
        return X_tr, X_te
    if dataset is 'netflix':
        return load_netflix()

