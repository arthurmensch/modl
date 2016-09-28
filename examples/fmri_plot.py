from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
from modl.plotting.fmri import display_maps
from nilearn._utils import check_niimg
from pymongo import MongoClient


def get_connections():
    client = MongoClient('localhost', 27017)
    db = client.sacred
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs

db, fs = get_connections()

parent_exp = db.find({'experiment.name': 'fmri_compare'}).sort('start_time', -1)[0]
parent_id = parent_exp['_id']
exps = list(db.find({'info.parent_id': parent_id}))

fig, axes = plt.subplots(1, 2, sharey=True)
for exp in exps:
    updated_params = exp['info']['updated_params']
    updated_params.pop('seed')
    axes[0].plot(exp['info']['iter'], exp['info']['score'],
                 label=updated_params, marker='o')
    axes[1].plot(exp['info']['time'], exp['info']['score'],
                 label=updated_params, marker='o')
axes[1].legend()
axes[0].set_ylabel('Train loss')
axes[0].set_xlabel('Iter')
axes[1].set_xlabel('Time (s)')
for i in range(2):
    axes[i].set_xscale('log')

# for exp in exps:
#     with NamedTemporaryFile(suffix='.nii.gz', dir='/run/shm') as f:
#         f.write(fs.get(exp['artifacts'][-1]).read())
#         components = check_niimg(f.name)
#         fig = display_maps(components)
#         fig.suptitle(exp['info']['updated_params'])

plt.show()