"""Run with ssh -Nf drago-mongo and mongoDB running,
or simply load the associated json files"""

import gridfs
import matplotlib as mpl
import numpy as np
from bson import ObjectId
from bson.json_util import dumps, loads
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from pymongo import MongoClient

from modl.plotting.images import plot_single_patch

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

def plot_qualitative():
    ref = np.load('components_1_49316s_841000p.npy')

    reduced = np.load('components_1_49316s_841000p.npy')

    full = np.load('components_1_177s_3000p.npy')

    ref = ref.reshape((-1, 16, 16, 223))
    reduced = reduced.reshape((-1, 16, 16, 223))
    full = full.reshape((-1, 16, 16, 223))

    fig, axes = plt.subplots(3, 3, figsize=(252 / 72.25, 80 / 72.25))
    fig.subplots_adjust(right=0.79, left=0.11, bottom=0.00, top=0.9,
                        wspace=0.1, hspace=0.1)
    names = ["""$r = 1$""",
             """$r = 24$""",
             """$r = 1$"""]
    times = ['\\textbf{14h}', '\\textbf{179 s}', '\\textbf{177 s}']
    patches = ["""841k patches""", """87k patches""", """3k patches"""]

    order = [0, 2, 1]

    for j, idx in enumerate([18, 49, 90]):
        for i, comp in enumerate([ref, full, reduced]):
            axes[i, j] = plot_single_patch(axes[i, j], comp[idx], 1, 3)
        axes[0, j].set_xlabel('Comp. %i' % (j + 1))
        axes[0, j].xaxis.set_label_coords(0.5, 1.4)
    for i in range(3):
        axes[i, 2].annotate('Time: %s' % times[order[i]],
                            xycoords='axes fraction', xy=(1.1, 0.6),
                            va='bottom', ha='left')
        axes[i, 2].annotate('%s' % patches[order[i]],
                            xycoords='axes fraction', xy=(1.1, 0.0),
                            va='bottom', ha='left')
    axes[1, 0].annotate("""\\textbf{\\textsc{OMF}}""",
                        xycoords='axes fraction', xy=(-0.5, 1.3), va='bottom',
                        ha='left')
    axes[1, 0].annotate("""$r = 1$""", xycoords='axes fraction',
                        xy=(-0.5, 0.7), va='bottom', ha='left')
    axes[2, 0].annotate("""\\textbf{\\textsc{SOMF}}""",
                        xycoords='axes fraction', xy=(-0.5, 0.6), va='bottom',
                        ha='left')
    axes[2, 0].annotate("""$r = 24$""", xycoords='axes fraction',
                        xy=(-0.5, 0.0), va='bottom', ha='left')
    trans = blended_transform_factory(fig.transFigure, axes[1, 0].transAxes)
    line = Line2D([0, 1], [-0.11, -0.11], color='black', linestyle=':',
                  linewidth=0.8, transform=trans)
    fig.lines.append(line)
    plt.savefig('patches.pdf')
    plt.show()

def prepare_qualitative():
    # On frodo
    db, fs = get_connections()

    query = [
        # Stage 1: filter experiments
        {
            "$match": {
                'experiment.name': 'decompose_images',
                'config.batch_size': 200,
                'config.data.source': 'aviris',
                'info.time.1': {"$exists": True},
                'config.non_negative_A': True,
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

                        'config.G_agg': {"$in": ['masked']},
                        'config.Dx_agg': {"$in": ['masked']},
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
            "$limit": 4
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
        idx = [10, -2] if exp['reduction'] == 1 else [29, -2]
        for this_idx in idx:
            print('Time : %s' % np.array(exp['profiling'])[this_idx, 5])
            print('Iter: %s' % exp['iter'][this_idx])
            components = np.load(fs.get(exp['artifacts'][this_idx]))
            np.save('components_%i_%is_%ip' % (exp['reduction'],
                                               np.array(exp['profiling'])[this_idx, 5],
                                               exp['iter'][this_idx]),
                    components)
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
                'info.time.1': {"$exists": True},
                'info.parent_id': ObjectId("58106de1fb5c8612b5c15a9d")
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
    exps = []
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
                exps.append(exp)
                profiling = np.array(exp['profiling'])[:, profiling_indices]
                iter = np.array(exp['iter'])
                mean_profiling = profiling[-1, :] / iter[-1, np.newaxis]
            except StopIteration:
                mean_profiling = np.zeros(len(profiling_indices))
            cum_profiling = np.cumsum(mean_profiling)
            cum_profilings.append(cum_profiling)

    s = dumps(exps)
    with open('profiling.json', 'w+') as f:
        f.write(s)

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

    fig = plt.figure(figsize=(252 / 72.25, 80 / 72.25))
    fig.subplots_adjust(left=0.14, right=0.68, bottom=0.18, top=0.87)

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
    offset_text = 4
    axes[0].annotate('No subsampling', xy=(1.2, 41.5),
                     xytext=(0, 9), ha='center', va='center',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('Dictionary', xy=(2.05, 15), ha='left', va='top',
                     xytext=(0, 6 + offset_text),
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Surrogate', xy=(3.05, 15),
                     rotation=15,
                     xytext=(0, 4 + offset_text), ha='left', va='top',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Code', xy=(4.05, 15),
                     rotation=15,
                     xytext=(0, -6 + offset_text), ha='left', va='top',
                     xycoords='data',
                     textcoords='offset points')

    axes[0].annotate('Subsampling for:', xy=(6, 41.5),
                     xytext=(0, 9), ha='center', va='center',
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('Dictionary', xy=(5.55, 15), xytext=(0, -6 + offset_text),
                     ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Surrogate', xy=(6.55, 15),
                     xytext=(0, -8 + offset_text), ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    axes[1].annotate('+ Code', xy=(7.55, 15),
                     xytext=(0, -18 + offset_text), ha='left', va='top',
                     rotation=15,
                     xycoords='data',
                     textcoords='offset points')

    # axes[0].set_axisbelow(False)
    legend = axes[0].legend(title='Time to compute:', ncol=1,
                            loc='upper left',
                            bbox_to_anchor=(1.06, 2.4),
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
    axes[0].set_yticks([39, 42])
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
    s = dumps(algorithm_exps)
    with open('method_comparison.json', 'w+') as f:
        f.write(s)

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
        'tsp gram': 'Averaged estimators (c)',
        'icml': 'Masked loss (a)',
        'full': 'No subsampling (19)'}

    fig = plt.figure(figsize=(7.141, 1.4))
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


def plot_compare_reductions_poster():
    with open('compare_reductions.json', 'r') as f:
        s = f.read()
    dataset_exps = loads(s)
    # Plotting
    reductions = [1, 4, 6, 8, 12, 24]
    n_red = len(reductions)

    # fig, axes = plt.subplots(1, 4, figsize=(7.141, 1.4))
    fig, axes = plt.subplots(2, 2, figsize=(4, 2.1))
    axes = axes.ravel()
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(right=0.98, left=0.07, bottom=0.1, top=0.96,
                        wspace=0.25)

    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85,
                                     reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2,
                                         light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip(reductions, colormap)}
    color_dict[1] = ref_colormap[0]
    for i, algorithm in enumerate(['adhd', 'aviris', 'aviris_dl', 'hcp']):
        exps = dataset_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            score = np.array(exp['info']['score'])
            if algorithm == 'hcp':
                score -= 1e5
            elif algorithm == 'adhd':
                score /= 1e3
            time = np.array(exp['info']['profiling'])[:, 5] / 3600
            time += 0.001 if algorithm == 'adhd' else 0.01
            # time = np.array(exp['info']['time']) / 3600 + 1e-3
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
    axes[0].yaxis.set_label_coords(-0.3, 0.4)

    for i in [0, 2]:
        axes[i].set_xlabel('Time (h)')
        ticklab = axes[i].xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        axes[i].xaxis.set_label_coords(axes[i].get_xlim()[0] * 1.5, 0,
                                       transform=trans)

    axes[0].set_xlim([1.5e-3, 2e-1])
    axes[0].set_ylim([21.8, 27])
    axes[1].set_xlim([0.015, 10])
    axes[1].set_ylim([3.2, 4])
    axes[2].set_xlim([0.015, 10])
    axes[2].set_ylim([34, 45])
    axes[3].set_xlim([0.02, 30])
    axes[3].set_ylim([-3200, 4500])
    axes[3].annotate('$+ 10^5$', xy=(0.04, 1.1), xycoords='axes fraction', va='top',
                     ha='left', fontsize=6)
    axes[0].annotate('$\\times 10^3$', xy=(0.04, 1.1), xycoords='axes fraction',
                     va='top',
                     ha='left', fontsize=6)
    axes[0].annotate('ADHD\nSparse dictionary',
                     # '$p = 6\\cdot 10^4\\ \\: n = 6000$',
                        xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[1].annotate('Aviris\nNMF',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[2].annotate('Aviris\nDictionary learning',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[3].annotate('HCP\nSparse dictionary',
                     # 'p = 2\\cdot 10^5\\ \\: n = 2\\cdot 10^6$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')

    axes[0].annotate('\\textbf{2 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction',
                     ha='center')
    axes[1].annotate('\\textbf{103 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction', ha='center')
    axes[2].annotate('\\textbf{103 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction', ha='center')
    axes[3].annotate('\\textbf{2 TB}', xy=(0.8, 0.5),
                     xycoords='axes fraction',
                     ha='center')

    handles, labels = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = axes[3].get_legend_handles_labels()

    handles += handles_2[-1:]
    labels += labels_2[-1:]

    # axes[0].annotate('Reduction', xy=(-0.24, -0.25),
    # axes[2].annotate('Reduction', xy=(-0.15, -0.35),
    #                  xycoords='axes fraction',
    #                  va='top')
    # axes[0].legend(handles[:n_red],
    plt.savefig('compare_reductions_poster.pdf')
    legend_fig = plt.figure()
    legend = plt.figlegend(handles[:n_red],
                   [('$r = %s$' % reduction) if reduction != 1 else 'None'
                    for
                    reduction in reductions],
                   loc='center',
                   title='Reduction',
                   handlelength=1,

                   ncol=1,
                   frameon=False)
    legend_fig.canvas.draw()
    legend_fig.savefig('legend_cropped.pdf',
                       bbox_inches=legend.get_window_extent().transformed(
                           legend_fig.dpi_scale_trans.inverted()))
    plt.show()


def plot_compare_reductions():
    datasets = {
        'hcp': {
                'parent_ids': [ObjectId('580e4a3cfb5c865d3c831640'),
                               ObjectId('580e4a30fb5c865d262d391a')]},
        'adhd': {
                 'parent_ids': [ObjectId("5804f140fb5c860e90e8db74"),
                                ObjectId("5804f404fb5c861a5f45a222")
                                ]},
        'aviris': {
                   'parent_ids': [ObjectId("58337e42fb5c867671262f89")]},
        'aviris_dl': {
                      'parent_ids': [ObjectId("58372740fb5c8626cb8d1b66")]}
    }

    dataset_exps = {}
    for dataset in ['adhd', 'aviris', 'aviris_dl', 'hcp']:
        print(dataset)
        parent_ids = datasets[dataset]['parent_ids']
        db, fs = get_connections()
        exps = list(db.find({
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": 'full' if dataset == 'adhd' else 'async',
                "config.G_agg": 'average' if dataset
                                             not in ['aviris', 'aviris_dl'] else 'masked',
                "config.Dx_agg": 'average' if dataset
                                              not in ['aviris', 'aviris_dl'] else 'masked',
                "config.reduction": {"$ne": [1, 2]},
            }))
        if dataset == 'aviris':
            ref = db.find_one({'_id': ObjectId('58337b4dfb5c8669e75e32a1')})
        else:
            ref = db.find_one({
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1})
        exps.append(ref)
        dataset_exps[dataset] = exps

        time = np.array(ref['info']['profiling'])[:, 5]
        tol = 1e-2
        ref_loss = ref['info']['score'][-1]
        rel_score = np.array(ref['info']['score']) / ref_loss
        it_tol = np.where(rel_score < 1 + tol)[0][0]
        ref_time = time[it_tol]
        rel_times = []
        for exp in exps:
            ref_loss = exp['info']['score'][-1]
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
    s = dumps(dataset_exps)
    with open('compare_reduction.json', 'w+') as f:
        f.write(s)
    # Plotting
    reductions = [1, 4, 6, 8, 12, 24]
    n_red = len(reductions)

    fig, axes = plt.subplots(1, 4, figsize=(7.141, 1.4))
    fig.subplots_adjust(right=0.98, left=0.08, bottom=0.25, top=0.96,
                        wspace=0.24)

    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85,
                                     reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2,
                                         light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip(reductions, colormap)}
    color_dict[1] = ref_colormap[0]
    for i, algorithm in enumerate(['adhd', 'aviris', 'aviris_dl', 'hcp']):
        exps = dataset_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            score = np.array(exp['info']['score'])
            if algorithm == 'hcp':
                score -= 1e5
            time = np.array(exp['info']['profiling'])[:, 5] / 3600
            time += 0.001 if algorithm == 'adhd' else 0.01
            # time = np.array(exp['info']['time']) / 3600 + 1e-3
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
    axes[0].yaxis.set_label_coords(-0.3, 0.4)

    axes[0].set_xlabel('Time (h)')
    ticklab = axes[0].xaxis.get_ticklabels()[0]
    trans = ticklab.get_transform()
    axes[0].xaxis.set_label_coords(axes[1].get_xlim()[0] * 0.1, 0,
                                   transform=trans)

    axes[0].set_xlim([1.5e-3, 2e-1])
    axes[0].set_ylim([21800, 27000])
    axes[1].set_xlim([0.015, 10])
    axes[1].set_ylim([3.2, 4])
    axes[2].set_xlim([0.015, 10])
    axes[2].set_ylim([34, 45])
    axes[3].set_xlim([0.02, 30])
    axes[3].set_ylim([-3200, 4500])
    axes[3].annotate('$+ 10^5$', xy=(0.04, 1), xycoords='axes fraction', va='top',
                     ha='left', fontsize=6)
    axes[0].annotate('ADHD\nSparse dictionary',
                     # '$p = 6\\cdot 10^4\\ \\: n = 6000$',
                        xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[1].annotate('Aviris\nNMF',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[2].annotate('Aviris\nDictionary learning',
                     # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
    axes[3].annotate('HCP\nSparse dictionary',
                     # 'p = 2\\cdot 10^5\\ \\: n = 2\\cdot 10^6$',
                     xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')

    axes[0].annotate('\\textbf{2 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction',
                     ha='center')
    axes[1].annotate('\\textbf{103 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction', ha='center')
    axes[2].annotate('\\textbf{103 GB}', xy=(0.8, 0.5),
                     xycoords='axes fraction', ha='center')
    axes[3].annotate('\\textbf{2 TB}', xy=(0.8, 0.5),
                     xycoords='axes fraction',
                     ha='center')

    handles, labels = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = axes[3].get_legend_handles_labels()

    handles += handles_2[-1:]
    labels += labels_2[-1:]

    axes[0].annotate('Reduction', xy=(-0.24, -0.25),
                     xycoords='axes fraction',
                     va='top')
    axes[0].legend(handles[:n_red],
                   [('$r = %s$' % reduction) if reduction != 1 else 'None'
                    for
                    reduction in reductions],
                   bbox_to_anchor=(0.2, -0.15),
                   loc='upper left',
                   ncol=8,
                   frameon=False)

    plt.savefig('compare_reduction.pdf')
    plt.show()


if __name__ == '__main__':
    # prepare_qualitative()
    # plot_qualitative()
    # plot_method_comparison()
    # plot_compare_reductions()
    plot_compare_reductions_poster()
    # plot_profiling()
