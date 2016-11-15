import itertools
from tempfile import NamedTemporaryFile

from matplotlib import gridspec
from modl.plotting.images import plot_patches
from pymongo import MongoClient
import gridfs

import numpy as np
import pandas as pd

import matplotlib as mpl

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


def flatten(a):
    return dict(heartbeat=a['heartbeat'], **a['config'], **a['info'])


def run():
    db, fs = get_connections()

    a = db.find({'experiment.name': 'decompose_fmri',
                 'info.iter.10': {"$exists": True},

                 "config.data.dataset": "adhd"},
                {"config": 1,
                 'info.iter': 1,
                 'info.time': 1,
                 'info.profile': 1,
                 'info.score': 1,
                 'heartbeat': 1}
                )

    df = pd.DataFrame(list(map(flatten, a)))

    df = df.sort_values(by='heartbeat', ascending=False)
    df = df[df['l1_ratio'] == 0.9]
    df = df[df['AB_agg'] == 'async']
    df = df[df['G_agg'] == 'full']
    df = df[df['Dx_agg'] == 'full']
    df = df[df['l1_ratio'] == 0.9]
    # df = df[df['batch_size'] == 200]

    df['score'] = df['score'].apply(np.array)
    df['time'] = df['time'].apply(np.array)
    indices = [np.all(df.iloc[i]['time'] < 500 * 3600) for i in
               range(df.shape[0])]
    df = df[indices]

    fig, ax = plt.subplots(1, 1)
    reductions = np.sort(np.unique(df['reduction']))
    colormap = sns.cubehelix_palette(len(reductions), rot=0.3, light=0.85,
                                     reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip(reductions, colormap)}
    for reduction, sub_df in df.groupby('reduction'):

        for idx, line in list(sub_df.iterrows()):
            ax.plot(line['time'] / 3600, line['score'],
                    label="%s %s" % (line['l1_ratio'], line['heartbeat']),
                    color=color_dict[reduction])
    # ax.legend()
    ax.set_xscale('log')
    plt.show()


def plot_score():
    db, fs = get_connections()

    query = [
        # Stage 1: filter experiments
        {
            "$match": {
                'experiment.name': 'decompose_images',
                'config.batch_size': 100,
                'config.data.source': 'aviris',
                'info.time.1': {"$exists": True},
                "$or": [
                    {
                        'config.reduction': 1,
                        'config.AB_agg': 'full',
                        'config.G_agg': 'full',
                        'config.Dx_agg': 'full'
                    },
                    {
                        'config.AB_agg': 'async',
                        'config.G_agg': {"$in": ['full']},
                        'config.Dx_agg': 'average'
                    }
                ]
            }
        },
        # Stage 2: project interesting values
        {
            "$project": {
                '_id': 1,
                'heartbeat': 1,
                'parent_id': "$info.parent_id",
                'AB_agg': "$config.AB_agg",
                'G_agg': "$config.G_agg",
                'Dx_agg': "$config.Dx_agg",
                'reduction': '$config.reduction',
                'iter': "$info.iter",
                'profiling': "$info.profiling",
                'time': "$info.time",
                'score': "$info.score",
                'artifacts': "$artifacts"
            }
        },
        # Stage 3: Group by compare_exp (if any)
        {
            "$group": {
                "_id": "$parent_id",
                "experiments": {"$push": "$$ROOT"},
                "heartbeat": {"$max": '$heartbeat'}
            }
        },
        # Stage 4: Sort by last exp
        {
            "$sort": {
                "heartbeat": -1
            }
        },
        # Stage 5
        {
            "$skip": 0
        },
        # Stage 5
        {
            "$limit": 1
        },
        # Stage 6: Ungroup experiments
        {
            "$unwind": "$experiments"
        },
        # Stage 7
        {
            "$project": {
                '_id': "$experiments._id",
                'heartbeat': "$experiments.heartbeat",
                'parent_id': "$_id",
                'AB_agg': "$experiments.AB_agg",
                'G_agg': "$experiments.G_agg",
                'Dx_agg': "$experiments.Dx_agg",
                'reduction': "$experiments.reduction",
                'iter': "$experiments.iter",
                'profiling': "$experiments.profiling",
                'score': "$experiments.score",
                'time': "$experiments.time",
                'artifacts': "$experiments.artifacts"
            }
        },
        {
            "$sort": {'reduction': 1}
        }
    ]

    exps = list(db.aggregate(query))
    print(exps)

    profiling_indices = [0, 2, 3, 1, 4, 6, 7]
    n_indices = len(profiling_indices)
    profiling_labels = ['Code', 'Lasso', 'Surrogate parameters', 'Gram matrix',
                        "Dictionary", "io", "correction"]

    colormap = sns.cubehelix_palette(n_indices, start=0,
                                     rot=1, reverse=True)
    fig, ax = plt.subplots(len(exps), 1, sharey=True, sharex=True)
    for i, exp in enumerate(exps):
        iter = np.array(exp['iter'])
        time = np.cumsum(np.array(exp['profiling'])[:, profiling_indices], axis=1)
        time = (time[1:] - time[:-1]) / (iter[1:] - iter[:-1])[:, np.newaxis]
        for j in reversed(range(n_indices)):
            ax[i].fill_between(iter[1:], time[:, j], time[:, j - 1] if j > 0 else 0,
                               label=profiling_labels[j], color=colormap[j])
        ax[i].set_xscale('log')
        ax[i].annotate("Reduction = %s" % exp['reduction'], xy=(0.5, 0.8), xycoords='axes fraction')
    ax[-1].legend()

    fig, ax = plt.subplots(1, 1)
    fig, ax2 = plt.subplots(1, 1)
    for i, exp in enumerate(exps):
        iter = np.array(exp['iter'])
        score = np.array(exp['score'])
        time = np.array(exp['profiling'])[:, 5] + 0.001
        # time = np.array(exp['time']) + 0.001
        # time = iter / exp['reduction']
        ax.plot(iter, time, label=exp['reduction'])
        ax2.plot(time, score, label=exp['reduction'])
    ax2.set_xscale('log')

    # for exp in exps:
    #     with NamedTemporaryFile(suffix='.npy',
    #                             dir='/run/shm') as f:
    #         f.write(fs.get(exp['artifacts'][-1]).read())
    #         components = np.load(f.name)
    #         fig = plt.figure(figsize=(4.2, 4))
    #         fig = plot_patches(fig, components,
    #                            shape=(8, 8, 224))
    #         fig.suptitle(exp['reduction'])
    ax.legend()
    ax2.legend()
    plt.show()

def plot_profiling():
    # db, fs = get_connections()
    #
    # algorithm_config = {'full': {'AB_agg': 'full',
    #                              'G_agg': 'full',
    #                              'Dx_agg': 'full'},
    #                     'async': {'AB_agg': 'async',
    #                               'G_agg': 'full',
    #                               'Dx_agg': 'full'},
    #                     'tsp': {'AB_agg': 'async',
    #                             'G_agg': 'full',
    #                             'Dx_agg': 'average'}}
    #
    # profiling_labels = ['Code', 'Surrogate\nparameters', 'Gram matrix',
    #                     u"Dictionary\n\u00A0"]
    # profiling_indices = [0, 3, 1, 4]
    # n_indices = len(profiling_indices)
    #
    # query = [
    #     # Stage 1: filter experiments
    #     {
    #         "$match": {
    #             'experiment.name': 'decompose_fmri',
    #             'config.batch_size': 200,
    #             'info.time.1': {"$exists": True}
    #         }
    #     },
    #     # Stage 2: project interesting values
    #     {
    #         "$project": {
    #             '_id': 1,
    #             'heartbeat': 1,
    #             'parent_id': "$info.parent_id",
    #             'AB_agg': "$config.AB_agg",
    #             'G_agg': "$config.G_agg",
    #             'Dx_agg': "$config.Dx_agg",
    #             'reduction': '$config.reduction',
    #             'iter': "$info.iter",
    #             'profiling': "$info.profiling",
    #             'score': "$info.score",
    #         }
    #     },
    #     # Stage 3: Group by compare_exp (if any)
    #     {
    #         "$group": {
    #             "_id": "$parent_id",
    #             "experiments": {"$push": "$$ROOT"},
    #             "heartbeat": {"$max": '$heartbeat'}
    #         }
    #     },
    #     # Stage 4: Sort by last exp
    #     {
    #         "$sort": {
    #             "heartbeat": -1
    #         }
    #     },
    #     # Stage 5
    #     {
    #         "$limit": 1
    #     },
    #     # Stage 6: Ungroup experiments
    #     {
    #         "$unwind": "$experiments"
    #     },
    #     # Stage 7
    #     {
    #         "$project": {
    #             '_id': "$experiments._id",
    #             'heartbeat': "$experiments.heartbeat",
    #             'parent_id': "$_id",
    #             'AB_agg': "$experiments.AB_agg",
    #             'G_agg': "$experiments.G_agg",
    #             'Dx_agg': "$experiments.Dx_agg",
    #             'reduction': "$experiments.reduction",
    #             'iter': "$experiments.iter",
    #             'profiling': "$experiments.profiling",
    #             'score': "$experiments.score",
    #         }
    #     }
    # ]
    # cum_profilings = []
    #
    # for reduction in [1, 6, 12]:
    #     algorithms = ['full', 'async', 'tsp'] if reduction != 1 else ['full']
    #     for algorithm in algorithms:
    #         this_query = query.copy()
    #         this_config = algorithm_config[algorithm]
    #         this_query += [
    #             {
    #                 "$match": {'reduction': reduction,
    #                            'AB_agg': this_config['AB_agg'],
    #                            'G_agg': this_config['G_agg'],
    #                            'Dx_agg': this_config['Dx_agg']
    #                            }
    #             },
    #             {"$limit": 1}
    #         ]
    #         try:
    #             exp = db.aggregate(this_query).next()
    #             profiling = np.array(exp['profiling'])[:, profiling_indices]
    #             iter = np.array(exp['iter'])
    #             mean_profiling = profiling[-1, :] / iter[-1, np.newaxis]
    #         except StopIteration:
    #             mean_profiling = np.zeros(len(profiling_indices))
    #         cum_profiling = np.cumsum(mean_profiling)
    #         cum_profilings.append(cum_profiling)
    #
    # cum_profilings = np.concatenate([cum_profiling[np.newaxis, :]
    #                                  for cum_profiling in cum_profilings])
    # np.save('profiling', cum_profilings)
    cum_profilings = np.load('profiling.npy')
    # cum_profilings *= 1000

    profiling_labels = ['Code', 'Surrogate\nparameters', 'Gram matrix',
                        u"Dictionary"]
    profiling_indices = [0, 3, 1, 4]
    n_indices = len(profiling_indices)

    # Plotting
    ratio = 3
    main_ylim = [0.0, 15]
    offset = 38
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, ratio])

    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 7

    x_pos = np.array([1, 2.5, 3.5, 4.5, 6, 7, 8]) - 0.45

    colormap = sns.cubehelix_palette(n_indices, start=0,
                                     rot=1, reverse=True)

    fig = plt.figure(figsize=(252 / 72, 90 / 72))
    fig.subplots_adjust(left=0.14, right=0.68, bottom=0.2)

    axes = [plt.subplot(gs[0, 0])]
    axes.append(plt.subplot(gs[1, 0], sharex=axes[0]))

    for j in reversed(range(n_indices)):
        for ax in axes:
            ax.bar(x_pos, cum_profilings[:, j] * 1000, 0.9,
                   bottom=0, label=profiling_labels[j],
                   zorder=n_indices - j,
                   linewidth=0.5,
                   edgecolor='black',
                   color=colormap[j]
                   # cmap=plt.cm.Set3
                   )

    axes[0].annotate('No subsampling', xy=(1.2, 41.5),
                     xytext=(0, 9), ha='center', va='center',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('Dictionary', xy=(2.05, 15), ha='left', va='top',
                     xytext=(0, 6),
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Parameters', xy=(3.05, 15),
                     rotation=15,
                     xytext=(0, 4), ha='left', va='top',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Code', xy=(4.05, 15),
                     rotation=15,
                     xytext=(0, -6), ha='left', va='top',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('Subsampling for:', xy=(6, 15),
                     xytext=(0, 15), ha='center', va='top',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('Dictionary', xy=(5.55, 15), xytext=(0, -6),
                     ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Parameters', xy=(6.55, 15),
                     xytext=(0, -8), ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Code', xy=(7.55, 15),
                     xytext=(0, -18), ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    # axes[0].set_axisbelow(False)
    legend = axes[0].legend(title='Time to compute:', ncol=1,
                            loc='upper left',
                            bbox_to_anchor=(1.06, 1.75),
                            markerscale=0.6,
                            handlelength=1,
                            frameon=False,
                            # columnspacing=0.5,
                            # borderpad=0.22,
                            handletextpad=0.7,
                            # mode='expand'
                            )
    # legend.get_title().set_position((0, -5))
    # legend.get_texts()[0]
    axes[0].set_xlim(0, 9)
    axes[1].set_xticks([1., 3.5, 7])
    axes[1].set_xticklabels(["No reduction", "$r = 6$", '$r = 24$'])
    axes[1].set_ylabel('Computation time\nper sample (ms)')
    axes[1].yaxis.set_label_coords(-0.13, 0.73)
    axes[1].set_ylim(main_ylim)
    axes[0].set_ylim(offset, offset + (main_ylim[1] - main_ylim[0]) / ratio)
    axes[0].set_yticks([40, 42])
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    plt.setp(axes[0].xaxis, visible=False)

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False,
                  linewidth=1)

    on = (1 + ratio)
    om = (1 + ratio) / ratio
    axes[0].plot((- d, d), (- d * on, + d * on),
                 **kwargs)
    kwargs.update(transform=axes[1].transAxes)
    axes[1].plot((-d, d), (1 - d * om, 1 + d * om), **kwargs)
    plt.savefig('profiling.pdf')
    plt.show()


if __name__ == '__main__':
    # plot_score()
    plot_profiling()
