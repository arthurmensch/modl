import gridfs
import matplotlib as mpl
import numpy as np
from bson.json_util import dumps
from pymongo import MongoClient

mpl.use('Qt5Agg')

import seaborn.apionly as sns

import matplotlib.pyplot as plt

from collections import OrderedDict


def get_connections():
    client = MongoClient('localhost', 27018, document_class=OrderedDict)
    # client = MongoClient('localhost', 27017, document_class=OrderedDict)
    db = client['sacred']
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs

def plot():
    db, fs = get_connections()

    query = [
        # Stage 1: filter experiments
        {
            "$match": {
                'experiment.name': 'decompose_images',
                'config.batch_size': 200,
                'config.data.source': 'aviris',
                'info.time.1': {"$exists": True},
                'config.non_negative_A': False,
                "$or": [
                    {
                        'config.reduction': 1,
                        'config.AB_agg': 'full',
                        'config.G_agg': 'full',
                        'config.Dx_agg': 'full',
                        'info.data_shape.1': 16,
                        # 'info.data_shape.2':> 3
                    },
                    {
                        'config.AB_agg': "async",

                        'config.G_agg': {"$in": ['masked', 'average']},
                        'config.Dx_agg': {"$in": ['masked', 'average']},
                        'info.data_shape.1': 16,
                        'config.reduction': {"$in": [6, 12, 24]}
                        # 'info.data_shape.2': 3
                    }
                ]
            }
        },
        # Stage 2: project interesting values
        {
            "$project": {
                '_id': 1,
                'heartbeat': 1,
                'start_time': 1,
                'parent_id': "$info.parent_id",
                'AB_agg': "$config.AB_agg",
                'G_agg': "$config.G_agg",
                'Dx_agg': "$config.Dx_agg",
                'reduction': '$config.reduction',
                'iter': "$info.iter",
                'profiling': "$info.profiling",
                'time': "$info.time",
                'score': "$info.score",
                'artifacts': "$artifacts",
                'shape': "$info.data_shape"
            }
        },
        # # Stage 4: Sort by last exp
        {
            "$sort": {
                "start_time": -1
            }
        },
        # Stage 5
        {
            "$skip": 0
        },
        # Stage 5
        {
            "$limit": 7
        },
        # Stage 6: Ungroup experiments
        {
            "$sort": {"reduction": 1}
        }
    ]

    exps = list(db.aggregate(query))
    for exp in exps:
        print(exp)

    profiling_indices = [0, 2, 3, 1, 4, 6, 7]
    n_indices = len(profiling_indices)
    profiling_labels = ['Code', 'Lasso', 'Surrogate parameters', 'Gram matrix',
                        "Dictionary", "io", "correction"]

    colormap = sns.cubehelix_palette(n_indices, start=0,
                                     rot=1, reverse=True)
    fig, ax = plt.subplots(len(exps), 1, sharey=True, sharex=True)
    for i, exp in enumerate(exps):
        iter = np.array(exp['iter'])
        profiling = np.array(exp['profiling'])[:, profiling_indices]
        profiling[:, -1] -= profiling[:, -3]
        profiling[:, -1] = np.maximum(profiling[:, -1], 0)
        time = np.cumsum(profiling, axis=1)
        time = (time[1:] - time[:-1]) / (iter[1:] - iter[:-1])[:, np.newaxis]
        for j in reversed(range(n_indices)):
            ax[i].fill_between(iter[1:], time[:, j], time[:, j - 1] if j > 0 else 0,
                               label=profiling_labels[j], color=colormap[j])
        ax[i].set_xscale('log')
        ax[i].annotate("Reduction = %s" % exp['reduction'], xy=(0.5, 0.8), xycoords='axes fraction')
    ax[-1].legend()

    fig, ax = plt.subplots(1, 1)
    fig, ax2 = plt.subplots(1, 1)
    c = sns.cubehelix_palette(len(exps))
    for i, exp in enumerate(exps):
        print('%i %s' % (i, exp['shape']))
        iter = np.array(exp['iter'])
        score = np.array(exp['score'])
        time = np.array(exp['profiling'])[:, 5] + 10
        # time = np.array(exp['time']) + 0.001
        ax.plot(iter, score, label="%s %s %s" % (exp['reduction'], exp['G_agg'], exp['Dx_agg']), color=c[i], linestyle='--' if exp['G_agg'] == 'masked' else '-')
        ax2.plot(time, score, label="%s %s %s" % (exp['reduction'], exp['G_agg'], exp['Dx_agg']), color=c[i], linestyle='--' if exp['G_agg'] == 'masked' else '-')
    ax2.set_xscale('log')

    profilings = np.array(exps[0]['profiling'])[:, 5]
    profilings_red = np.array(exps[1]['profiling'])[:, 5]
    min_len = min(profilings.shape[0], profilings_red.shape[0])
    diff = profilings[1:min_len] / profilings_red[1:min_len]
    print(diff)

    s = dumps(exps)
    with open('qualitative.json', 'w+') as f:
        f.write(s)

    for i, exp in enumerate(exps):
        print(exp['reduction'])
        shape = exp['shape']
        idx = [3, -2] if exp['reduction'] == 1 else [36, -2]
        for this_idx in idx:
            print('Time : %s' % np.array(exp['profiling'])[this_idx, 5])
            print('Iter: %s' % exp['iter'][this_idx])
            components = np.load(fs.get(exp['artifacts'][this_idx]))
            np.save('components_negative_%i_%is_%ip' % (exp['reduction'],
                                               np.array(exp['profiling'])[this_idx, 5],
                                               exp['iter'][this_idx]),
                    components)
    ax.legend()
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    plot()
