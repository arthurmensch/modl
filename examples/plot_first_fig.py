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
    name = 'compare'


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
        'adhd': {'sub_db': 'fmri',
                 'parent_ids': [ObjectId("57ed4726fb5c86523cf8e674")]},
        'aviris': {'sub_db': 'sacred',
                   'parent_ids': [ObjectId("57ee4810fb5c86bbd1786c35")]}
    }
    dataset_exps = {}
    for dataset in ['adhd', 'aviris', 'hcp']:
        parent_ids = datasets[dataset]['parent_ids']
        db, fs = get_connections(datasets[dataset]['sub_db'])
        dataset_exps[dataset] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": 'full',
                "config.G_agg": 'full',
                "config.Dx_agg": 'full',
                "config.reduction": {"$ne": [1, 2]},
            }, {
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1}
            ]}))

    reductions = [1, 2, 4, 8, 12, 16, 20, 24]
    n_red = len(reductions)

    fig, axes = plt.subplots(1, 3, figsize=(7.166, 1.5))
    fig.subplots_adjust(right=0.97, left=0.1, bottom=0.3, top=0.9,
                        wspace=0.24)

    # Plotting
    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2, light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip(reductions, colormap)}
    color_dict[1] = ref_colormap[0]
    for i, algorithm in enumerate(['adhd', 'aviris', 'hcp']):
        exps = dataset_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            score = np.array(exp['info']['score'])
            time = np.array(exp['info']['time'])
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            axes[i].plot(time, score,
                      label="$r = %i$" % reduction if reduction != 1 else "None",
                      color=color,
                      markersize=2)
        axes[i].set_xscale('log')
        sns.despine(fig, axes)

    axes[0].set_ylabel('Test objective function')

    axes[2].set_xlabel('Time')
    axes[2].xaxis.set_label_coords(1.02, -.105)

    axes[0].set_xlim([5e1, 5e3])
    axes[1].set_xlim([4e1, 5e3])
    axes[0].set_ylim([21800, 24200])
    axes[1].set_ylim([9600, 11000])
    axes[2].set_xlim([5e1, 3e4])
    axes[2].set_ylim([97000, 104200])

    axes[0].annotate('ADHD',
    # '$p = 6\\cdot 10^4\\ \\: n = 6000$',
                     xy=(0.6, 0.9), xycoords='axes fraction', ha='center')
    axes[1].annotate('Aviris',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 0.9), xycoords='axes fraction', ha='center')
    axes[2].annotate('HCP',
    # 'p = 2\\cdot 10^5\\ \\: n = 2\\cdot 10^6$',
                                             xy=(0.6, 0.9), xycoords='axes fraction', ha='center')

    axes[0].annotate('\\textbf{2 GB}', xy=(0.9, 0.6), xycoords='axes fraction', ha='center')
    axes[1].annotate('\\textbf{45 GB}', xy=(0.9, 0.6), xycoords='axes fraction', ha='center')
    axes[2].annotate('\\textbf{2 TB}', xy=(0.9, 0.6), xycoords='axes fraction', ha='center')

    handles, labels = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = axes[2].get_legend_handles_labels()

    handles += handles_2[-1:]
    labels += labels_2[-1:]

    axes[0].annotate('Reduction', xy=(-0.18, -0.25), xycoords='axes fraction', va='top')
    axes[0].legend(handles[:n_red], labels[:n_red],
                                  bbox_to_anchor=(0.1, -0.15),
                                  loc='upper left',
                                  ncol=8,
                                  frameon=False)

    print('Done plotting figure')
    plt.savefig(name + 'bench' + '.pdf')
    plt.show()
