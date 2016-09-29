from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from nilearn._utils import check_niimg
from pymongo import MongoClient
import numpy as np


def get_connections():
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client.fmri
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


db, fs = get_connections()

parent_exp = db.find({'experiment.name': 'compare',
                      'status': 'COMPLETED'
                      }).sort('_id', -1)[0]
parent_id = parent_exp['_id']
exps = list(db.find({'$and':
                         [{'info.parent_id': parent_id,
                           "config.AB_agg": 'full',
                           "config.Dx_agg": {'$in': ['masked', 'full']},
                           'config.reduction': {'$in': [1, 2, 4]},
                           },
                          ]}))

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
fig.subplots_adjust(bottom=0.3)
ref = min([np.min(np.array(exp['info']['score'])) for exp in exps]) * 0.999
for exp in exps:
    updated_params = exp['info']['updated_params']
    score = np.array(exp['info']['score'])
    rel_score = (score - ref) / ref
    axes[0].plot(np.array(exp['info']['iter']) + 10, score,
                 label=updated_params, marker='o', markersize=2)
    axes[1].plot(exp['info']['time'], rel_score,
                 label=updated_params, marker='o', markersize=2)
    axes[2].plot(np.array(exp['info']['iter']) + 10, exp['info']['time'],
                 label=updated_params, marker='o', markersize=2)
axes[1].legend(bbox_to_anchor=(-1.5, -.2), loc='upper left', ncol=3)
axes[0].set_ylabel('Train loss')
axes[0].set_xlabel('Iter')
axes[1].set_xlabel('Time (s)')
for i in range(2):
    axes[i].set_xscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')
axes[2].set_xscale('log')
axes[2].set_xlabel('Iter')
axes[2].set_ylabel('Time (s)')

# for exp in exps:
#     with NamedTemporaryFile(suffix='.npy', dir='/run/shm') as f:
#         f.write(fs.get(exp['artifacts'][-2]).read())
#         components = np.load(f.name)
#         fig = plot_patches(components, shape=exp['info']['data_shape'])
#         fig.suptitle(exp['info']['updated_params'])

plt.show()
