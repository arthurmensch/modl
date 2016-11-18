from os.path import expanduser
from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from bson import ObjectId

import matplotlib as mpl
mpl.rcParams['ytick.labelsize'] = 7

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


datasets = {
    'hcp': {'sub_db': 'sacred',
            'parent_ids': [ObjectId('580e4a3cfb5c865d3c831640'),
                           ObjectId('580e4a30fb5c865d262d391a')]},
    'adhd': {'sub_db': 'sacred',
             'parent_ids': [ObjectId("5804f140fb5c860e90e8db74"),
                            ObjectId("5804f404fb5c861a5f45a222")
                            ]},
    'aviris': {'sub_db': 'sacred',
               'parent_ids': [ObjectId("582d876ffb5c869561bf336f")]}
}


@plot_ex.command
def table():
    for dataset in ['adhd', 'aviris', 'hcp']:
        print(dataset)
        parent_ids = datasets[dataset]['parent_ids']
        db, fs = get_connections(datasets[dataset]['sub_db'])
        exps = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": 'full' if dataset == 'adhd' else 'async',
                "config.G_agg": 'average',
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

        time = np.array(ref['info']['profiling'])[:, 5]
        tol = 1e-2
        ref_loss = ref['info']['score'][-1]
        rel_score = np.array(ref['info']['score']) / ref_loss
        it_tol = np.where(rel_score < 1 + tol)[0][0]
        ref_time = time[it_tol]
        rel_times = []
        for exp in exps:
            # ref_loss = exp['info']['score'][-1]
            time = np.array(exp['info']['profiling'])[:, 5]
            rel_score = np.array(exp['info']['score']) / ref_loss
            it_tol = np.where(rel_score < 1 + tol)[0]
            if len(it_tol) > 0:
                time = time[it_tol[0]]
            else:
                time = ref_time
            rel_time = ref_time / time
            rel_times.append("OMF: %.0f s - %.2f min, SOMF: %.0f s - %.2f min, Speed-up: %.2f ($r = %i$)" %
                             (ref_time, ref_time / 60, time, time / 60, rel_time, exp['config']['reduction']))
        for rel_time in rel_times:
            print("%s" % rel_time)


@plot_ex.automain
def plot(name):
    dataset_exps = {}
    for dataset in ['adhd', 'aviris', 'hcp']:
        parent_ids = datasets[dataset]['parent_ids']
        db, fs = get_connections(datasets[dataset]['sub_db'])
        dataset_exps[dataset] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": 'full' if dataset == 'adhd' else 'async',
                "config.G_agg": 'average',
                "config.Dx_agg": 'average',
                "config.reduction": {"$ne": [1, 2]},
            }, {
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1}
            ]}))

    reductions = [1, 4, 6, 8, 12, 24]
    n_red = len(reductions)

    fig, axes = plt.subplots(1, 3, figsize=(7.166, 1.6))
    fig.subplots_adjust(right=0.94, left=0.08, bottom=0.25, top=0.9,
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
            # time = np.array(exp['info']['profiling'])[:, 5] / 3600 + 1e-3
            time = np.array(exp['info']['time']) / 3600 + 1e-3
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            axes[i].plot(time, score,
                         label="$r = %i$" % reduction if reduction != 1 else "None",
                         zorder=reduction if reduction != 1 else 1,
                         color=color,
                         markersize=2)
        axes[i].set_xscale('log')
        sns.despine(fig, axes)

    axes[0].set_ylabel('Test objective function')

    axes[2].set_xlabel('Time (h)')
    axes[2].xaxis.set_label_coords(1.1, -.105)

    axes[0].set_xlim([1e-3, 2e-1])
    axes[0].set_ylim([21800, 26200])
    axes[1].set_xlim([60 / 3600, 35841 / 3600])
    axes[1].set_ylim([0, 15])
    axes[2].set_xlim([5e1 / 3600, 2e5 / 3600])
    axes[2].set_ylim([96800, 104200])

    axes[0].annotate('ADHD',
                     # '$p = 6\\cdot 10^4\\ \\: n = 6000$',
                     xy=(0.6, 0.9), xycoords='axes fraction', ha='center')
    axes[1].annotate('Aviris',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 0.9), xycoords='axes fraction', ha='center')
    axes[2].annotate('HCP',
                     # 'p = 2\\cdot 10^5\\ \\: n = 2\\cdot 10^6$',
                     xy=(0.6, 0.9), xycoords='axes fraction', ha='center')

    axes[0].annotate('\\textbf{2 GB}', xy=(0.9, 0.6), xycoords='axes fraction',
                     ha='center')
    axes[1].annotate('\\textbf{103 GB}', xy=(0.9, 0.6),
                     xycoords='axes fraction', ha='center')
    axes[2].annotate('\\textbf{2 TB}', xy=(0.9, 0.6), xycoords='axes fraction',
                     ha='center')

    handles, labels = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = axes[2].get_legend_handles_labels()

    handles += handles_2[-1:]
    labels += labels_2[-1:]

    axes[0].annotate('Reduction', xy=(-0.18, -0.25), xycoords='axes fraction',
                     va='top')
    axes[0].legend(handles[:n_red],
                   [('$r = %s$' % reduction) if reduction != 1 else 'None' for
                    reduction in reductions],
                   bbox_to_anchor=(0.1, -0.15),
                   loc='upper left',
                   ncol=8,
                   frameon=False)

    plt.savefig(name + '.pdf')
    plt.show()
