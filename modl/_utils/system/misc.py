import json
import re
from pathlib import Path

from pymongo import MongoClient
import gridfs

import numpy as np


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        return super(NumpyAwareJSONEncoder, self).default(obj)


def prepare_folder(name, n_exp):
    folder = Path.home()
    folder = folder.joinpath('output/modl').joinpath(name)
    if folder.exists():
        parent = folder.parent
        n = 0
        for subdir in parent.iterdir():
            match = re.search('_[1-9]*', str(subdir))
            if match:
                n = max(n, int(match.group(0)[1:]))
        n += 1
        subdir = name + '_%i' % n
        folder = parent.joinpath(subdir)
    folder.mkdir()
    for i in range(n_exp):
        exp_folder = folder.joinpath('experiment_%i' % i)
        exp_folder.mkdir()
    return folder


def get_sacred_handlers():
    client = MongoClient('localhost', 27017)
    db = client.sacred
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs