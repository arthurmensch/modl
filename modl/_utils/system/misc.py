import json
from pathlib import Path
import numpy as np
import os
import re
import inspect

from sacred.observers import RunObserver

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

class MyObserver(RunObserver):
    @staticmethod
    def create(test_data):
        client = pymongo.MongoClient(url, **kwargs)
        database = client[db_name]
        for manipulator in SON_MANIPULATORS:
            database.add_son_manipulator(manipulator)
        runs_collection = database[prefix + '.runs']
        fs = gridfs.GridFS(database, collection=prefix)
        return MongoObserver(runs_collection, fs)

    def started_event(self, ex_info, host_info, start_time, config, comment):
        pass

    def heartbeat_event(self, info, captured_out, beat_time):
        pass

    def completed_event(self, stop_time, result):
        pass

    def interrupted_event(self, interrupt_time):
        pass

    def failed_event(self, fail_time, fail_trace):
        pass

    def resource_event(self, filename):
        pass

    def artifact_event(self, filename):
        pass
