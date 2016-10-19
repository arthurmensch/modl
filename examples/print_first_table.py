from os.path import expanduser
from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from bson import ObjectId
from matplotlib.lines import Line2D
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from pymongo import MongoClient
from sacred.experiment import Experiment

import matplotlib.patches as patches

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    name = 'compare_reductions'


def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(name):
    datasets = {
        'hcp': {'sub_db': 'sacred',
                'parent_ids': [ObjectId('57f22495fb5c86780390bca7'),
                               ObjectId('57f22489fb5c8677ec4a8414')]},
        'adhd': {'sub_db': 'sacred',
                 'parent_ids': [ObjectId("5804f140fb5c860e90e8db74"),
                                ObjectId("5804f404fb5c861a5f45a222")
                                ]},
        'aviris': {'sub_db': 'sacred',
                   'parent_ids': [ObjectId("57f665e9fb5c86aff0ab4036")]}
    }
    dataset_exps = {}
    for dataset in ['adhd', 'aviris', 'hcp']:
        parent_ids = datasets[dataset]['parent_ids']
        db, fs = get_connections(datasets[dataset]['sub_db'])
        exps = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": 'full',
                "config.G_agg": 'full' if dataset == 'hcp' else 'average',
                "config.Dx_agg": 'average',
                "config.reduction": {"$ne": [1, 2]},
            },
                # {
                # 'info.parent_id':
                #     {"$in": parent_ids},
                # "config.reduction": 1}
            ]}))
        ref = db.find_one({
            'info.parent_id':
                {"$in": parent_ids},
            "config.reduction": 1})

        tol = 1e-2
        ref_loss = ref['info']['score'][-1]
        rel_score = np.array(ref['info']['score']) / ref_loss
        it_tol = np.where(rel_score < 1 + tol)[0][0]
        ref_time = ref['info']['time'][it_tol]
        rel_times = []
        for exp in exps:
            # ref_loss = exp['info']['score'][-1]
            rel_score = np.array(exp['info']['score']) / ref_loss
            it_tol = np.where(rel_score < 1 + tol)[0]
            if len(it_tol) > 0:
                time = exp['info']['time'][it_tol[0]]
            else:
                time = ref_time
            rel_time = ref_time / time
            rel_times.append([ref_time, time, ref_time / 3600, time / 3600, rel_time, exp['config']
            ['reduction']])
        print(rel_times)
