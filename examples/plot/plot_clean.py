import itertools
from tempfile import NamedTemporaryFile

from bson import ObjectId
from matplotlib import gridspec
from matplotlib import patches
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
                'config.batch_size': 200,
                'config.data.source': 'aviris',
                'info.time.1': {"$exists": True},
                # 'status': 'RUNNING',
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
                'artifacts': "$artifacts",
                'shape': "$info.data_shape"
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
        # # Stage 4: Sort by last exp
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
                'artifacts': "$experiments.artifacts",
                'shape': "$experiments.shape"
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
        # time = iter
        ax.plot(iter, time, label=exp['reduction'])
        ax2.plot(time, score, label=exp['reduction'])
    ax2.set_xscale('log')

    # for i, exp in enumerate(exps):
    #     # shape = exp['shape']
    #     # with NamedTemporaryFile(suffix='.npy',
    #     #                         dir='/run/shm') as f:
    #     #     f.write(fs.get(exp['artifacts'][-1]).read())
    #     #     components = np.load(f.name)
    #     #     components = components.reshape((components.shape[0], *shape))
    #     #     fig = plt.figure(figsize=(4.2, 4))
    #     #     fig = plot_patches(fig, components)
    #     #     fig.suptitle(exp['reduction'])
    #     try:
    #         with open('image_%i.png' % i, 'wb') as f:
    #             f.write(fs.get(exp['artifacts'][-1]).read())
    #     except IndexError:
    #         print('skip %i' % i)
    ax.legend()
    ax2.legend()
    plt.show()


def plot_profiling():
    # Get values from MongoDB
    db, fs = get_connections()
    algorithm_config = {'full': {'AB_agg': 'full',
                                 'G_agg': 'full',
                                 'Dx_agg': 'full'},
                        'async': {'AB_agg': 'async',
                                  'G_agg': 'full',
                                  'Dx_agg': 'full'},
                        'tsp': {'AB_agg': 'async',
                                'G_agg': 'full',
                                'Dx_agg': 'average'}}

    query = [
        # Stage 1: filter experiments
        {
            "$match": {
                'experiment.name': 'decompose_fmri',
                'config.batch_size': 200,
                'info.time.1': {"$exists": True}
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
                'score': "$info.score",
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
            }
        }
    ]
    cum_profilings = []

    profiling_labels = ['Code', 'Surrogate\nparameters', 'Gram matrix',
                        u"Dictionary"]
    profiling_indices = [0, 3, 1, 4]
    n_indices = len(profiling_indices)

    for reduction in [1, 6, 12]:
        algorithms = ['full', 'async', 'tsp'] if reduction != 1 else ['full']
        for algorithm in algorithms:
            this_query = query.copy()
            this_config = algorithm_config[algorithm]
            this_query += [
                {
                    "$match": {'reduction': reduction,
                               'AB_agg': this_config['AB_agg'],
                               'G_agg': this_config['G_agg'],
                               'Dx_agg': this_config['Dx_agg']
                               }
                },
                {"$limit": 1}
            ]
            try:
                exp = db.aggregate(this_query).next()
                profiling = np.array(exp['profiling'])[:, profiling_indices]
                iter = np.array(exp['iter'])
                mean_profiling = profiling[-1, :] / iter[-1, np.newaxis]
            except StopIteration:
                mean_profiling = np.zeros(len(profiling_indices))
            cum_profiling = np.cumsum(mean_profiling)
            cum_profilings.append(cum_profiling)

    cum_profilings = np.concatenate([cum_profiling[np.newaxis, :]
                                     for cum_profiling in cum_profilings])
    np.save('profiling', cum_profilings)

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


def plot_method_comparison():
    ylim_zoom = [1e-3, 0.01]
    xlim_zoom = [0.5, 15]
    xlim = [1e-1, 15]
    ylim = [97000, 100000]
    db, fs = get_connections()
    algorithms = {
        'tsp gram': ['full', 'average', 'async'],
        'tsp': ['average', 'average', 'async'],
        'full': ['full', 'full', 'async'],
        'icml': ['masked', 'masked', 'async']
    }
    algorithm_keys = [
        'full',
        'tsp gram',
        'icml'
    ]
    reductions = [
        12,
        24
    ]
    ref = db.find_one(
        {
            '_id': ObjectId("580e4a30fb5c865d262d391e")
        })
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = []
        for reduction in reductions:
            res = db.find({"$or":
                [{
                    'info.parent_id':
                        ObjectId("580e4a3cfb5c865d3c831640"),
                    "config.AB_agg": AB_agg,
                    "config.G_agg": G_agg,
                    "config.Dx_agg": Dx_agg,
                    "config.reduction": reduction,
                    'info.profiling': {"$ne": []},
                }]}).sort("start_time", -1)[0]
            algorithm_exps[algorithm].append(res)
        algorithm_exps[algorithm].append(ref)
    n_red = len(reductions) + 1

    # Plotting
    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.3, light=0.85,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip([1] + reductions, colormap)}
    color_dict[1] = ref_colormap[0]

    ref = min([np.min(np.array(exp['info']['score']))
               for this_algorithm in algorithm_exps for exp in
               algorithm_exps[this_algorithm]]) * (1 - ylim_zoom[0])

    style = {'icml': ':', 'tsp': '-.', 'full': '--', 'tsp gram': '-'}
    names = {
        'tsp': 'Variance reduction (b)',
        'tsp gram': 'Variance reduction (c)',
        'icml': 'Masked loss (a)',
        'full': 'No subsampling (19)'}

    fig = plt.figure(figsize=(7.166, 1.4))
    fig.subplots_adjust(right=0.75, left=0.09, bottom=0.18, top=0.86,
                        wspace=0.24)
    axes = []
    axes.append(plt.subplot2grid((2, 2), (0, 0), rowspan=2))
    axes.append(plt.subplot2grid((2, 2), (0, 1)))
    axes.append(
        plt.subplot2grid((2, 2), (1, 1), sharex=axes[1], sharey=axes[1]))
    handles_method = []
    labels_method = []
    labels_reduction = []
    handles_reduction = []
    for i, algorithm in enumerate(algorithm_keys):
        exps = algorithm_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            score = np.array(exp['info']['score'])
            iter = np.array(exp['info']['iter'])
            # time = np.sum(np.array(exp['info']['profiling'])[:, :5], axis=1) / 3600
            time = np.array(exp['info']['profiling'])[:, 5] / 3600
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            rel_score = (score - ref) / ref
            handle, = axes[0].plot(time + 0.001, score,
                                   label="r = %i" % reduction if reduction != 1 else "None",
                                   zorder=reduction if reduction != 1 else 1,
                                   color=color,
                                   linestyle=style[algorithm],
                                   markersize=2)
            if reduction == 24:
                handles_method.append(handle)
                labels_method.append(names[algorithm])
            if algorithm == 'tsp gram':
                handles_reduction.append(handle)
                labels_reduction.append(
                    "r = %i" % reduction if reduction != 1 else "None")
            if reduction in [1, 12]:
                axes[1].plot(time + 0.001,
                             rel_score,
                             linestyle=style[algorithm],
                             label="r = %i" % reduction if reduction != 1 else "None",
                             zorder=reduction if reduction != 1 else 1,
                             color=color,
                             markersize=2)
            if reduction in [1, 24]:
                axes[2].plot(time + 0.001,
                             rel_score,
                             linestyle=style[algorithm],
                             label="r = %i" % reduction if reduction != 1 else "None",
                             zorder=reduction if reduction != 1 else 1,
                             color=color,
                             markersize=2)
    axes[0].set_ylabel('Test objective function')
    axes[1].set_ylabel('(relative to lowest value)', rotation=0)
    axes[1].yaxis.set_label_coords(0.3, 1)
    for j in range(3):
        axes[j].set_xscale('log')
        sns.despine(fig, axes[j])
    axes[2].set_xlabel('Time')
    ticklab = axes[2].xaxis.get_ticklabels()[0]
    trans = ticklab.get_transform()
    axes[2].xaxis.set_label_coords(axes[1].get_xlim()[1] * 0.32, 0,
                                   transform=trans)
    axes[1].set_yscale('log')
    axes[1].set_ylim([ylim_zoom[0], ylim_zoom[1]])
    axes[1].set_xlim(xlim_zoom)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    axes[0].set_ylim(ylim)
    axes[0].set_xlim(xlim)

    rect_0 = (xlim_zoom[0], (1 + ylim_zoom[0]) * ref)
    rect_len = (
        xlim_zoom[1] - rect_0[0], (1 + ylim_zoom[1]) * ref - rect_0[1])
    axes[0].add_patch(
        patches.Rectangle(
            rect_0,
            rect_len[0] * 0.9,
            rect_len[1],
            fill=False,
            linestyle='dashed'  # remove background
        )
    )
    axes[0].annotate("Zoom", xycoords='data',
                     xy=(
                         rect_0[0] * 1.9, rect_0[1] + 1.3 * rect_len[1]),
                     va='center', ha='right')

    axes[1].legend(handles_reduction, labels_reduction,
                                  bbox_to_anchor=(1.49, 1.5),
                                  loc='upper center',
                                  ncol=2, title='Subsampling ratio',
                                  frameon=False)
    l = axes[2].legend(handles_method,
                       labels_method,
                       bbox_to_anchor=(1.49, 1.3), loc='upper center',
                       ncol=1,
                       title='Code computation',
                       frameon=False)
    plt.savefig('method_comparison.pdf')
    plt.show()


if __name__ == '__main__':
    plot_score()
    # plot_method_comparison()
    # plot_profiling()
