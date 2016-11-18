import warnings
from os.path import expanduser
from tempfile import NamedTemporaryFile

import datetime
import matplotlib

matplotlib.use('Qt5Agg')
ticklabelpad = matplotlib.rcParams['xtick.major.pad']

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from bson import ObjectId
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from pymongo import MongoClient
from sacred.experiment import Experiment

import matplotlib.patches as patches
from math import log

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    sub_db = 'sacred'
    exp_name = 'compare_hyperspectral'
    name = 'compare_aviris'
    ylim_zoom = None
    xlim_zoom = None
    ylim = None
    xlim = None
    oid = None
    plot_type = 'debug'
    last = 1
    first = 0


@plot_ex.named_config
def aviris():
    ylim_zoom = [.1e-2, 5e-2]
    # xlim_zoom = [1e3, 2e4]
    # xlim = [1e1, 2e4]
    # ylim = [97000, 106500]


@plot_ex.named_config
def adhd():
    sub_db = 'sacred'
    exp_name = 'compare_adhd'
    name = 'compare_adhd'
    ylim_zoom = [1e-2, 5e-1]
    # ylim = [21000, 31000]
    # xlim = [10, 1000]
    # xlim_zoom = [100, 1000]


@plot_ex.named_config
def hcp():
    sub_db = 'sacred'
    exp_name = 'compare_hcp'
    # oid = ['57f22495fb5c86780390bca7',
    #        '57f22489fb5c8677ec4a8414']
    # oid = ['5805374bfb5c8663ab366cf4',
    #        '5804f8d4fb5c862f0efe72fd',
    #        '57f22489fb5c8677ec4a8414']
    # oid = ['58075073fb5c866f3e7a89e9',
    #         '58074ff6fb5c866eb5eb4ecb']
    oid = ['580e4a3cfb5c865d3c831640']
    name = 'compare_hcp'
    ylim_zoom = [1e-3, 0.01]
    xlim_zoom = [0.5, 15]
    xlim = [1e-1, 15]
    ylim = [97000, 100000]
    first = 2
    last = 3


@plot_ex.capture
def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.capture
def get_parent_ids(oid, exp_name, first, last):
    db, fs = get_connections()
    if oid is not None:
        oid = [ObjectId(this_oid) for this_oid in oid]
        parent_exps = db.find({'_id': {"$in": oid},
                               }).sort('_id', -1)
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]
    else:
        parent_exps = db.find({'experiment.name': exp_name,
                               }).sort('_id', -1)[first:last]
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]
        print(parent_ids)
    return parent_ids


@plot_ex.command
def get_ref_data():
    db, fs = get_connections()

    res = db.find_one(
        {
            # batch_size = 50
            # '_id': ObjectId('58078fa8fb5c86596f8948f1')
            # batch_size = 400 old
            # '_id': ObjectId('580dfbc7fb5c86bbab94b5d1')
            # batch size =  400 new
            # '_id':    ObjectId("580e403bfb5c8612b6624b18")
            # batch_size = 200 new
            '_id': ObjectId("580e4a30fb5c865d262d391e")
        })
    print(res['info']['parent_id'])
    return res


@plot_ex.command
def plot_batch_size():
    db, fs = get_connections()
    parent_ids = get_parent_ids()
    exps = list(db.find({'info.parent_id': {"$in": parent_ids}}))
    fig, ax = plt.subplots(1, 1)
    for exp in exps:
        reduction = exp['config']['reduction']
        batch_size = exp['config']['batch_size']
        iter = np.array(exp['info']['iter'])
        score = np.array(exp['info']['score'])
        time = np.array(exp['info']['profiling'])[:, 5]
        ax.plot(time, score,
                label='batch size: %s\n reduction: %s\nalpha %s' % (
                    batch_size, reduction, exp['config']['alpha']))
    ax.legend()
    plt.show()


@plot_ex.command
def plot_batch_size_hcp():
    db, fs = get_connections()
    exps = list(db.find({'config.reduction': 1,
                         'config.data.dataset': 'hcp',
                         "info.iter.10": {"$exists": True},
                         # 'start_time': {"$gte": datetime.datetime(2016, 10, 24, 15)}
                         }))
    fig, ax = plt.subplots(1, 1)
    for exp in exps:
        reduction = exp['config']['reduction']
        batch_size = exp['config']['batch_size']
        iter = np.array(exp['info']['iter'])
        score = np.array(exp['info']['score'])
        time = np.array(exp['info']['profiling'])[:, 5]
        ax.plot(time, score,
                label='batch size: %s\n reduction: %s\nalpha %s' % (
                    batch_size, reduction, exp['start_time']))

    ax.legend()
    plt.show()


@plot_ex.command
def plot_profiling_data():
    db, fs = get_connections


@plot_ex.command
def plot_profiling_data(name):
    db, fs = get_connections()
    # old 400
    parent_ids = [ObjectId('580df955fb5c86aaa1407778')]
    ref = db.find_one(
    {
        '_id': ObjectId('58078fa8fb5c86596f8948f1')
    })
    # parent_ids = [ObjectId('580e4176fb5c8618c52641bb')]
    # ref = db.find_one(
    #     {
    #         '_id': ObjectId("580e403bfb5c8612b6624b18")
    #     })
    algorithms = {
        'full async': ['full', 'full', 'async'],
        'full': ['full', 'full', 'full'],
        'tsp gram': ['full', 'average', 'async'],
        'icml': ['masked', 'masked', 'async']
    }
    reductions = [6, 12]
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [
                {
                    'info.parent_id': {"$in": parent_ids},
                    "config.AB_agg": AB_agg,
                    "info.profiling": {"$ne": []},
                    "config.G_agg": G_agg,
                    "config.Dx_agg": Dx_agg,
                    'status': {'$in': ['RUNNING']},
                    "config.reduction": {"$in": reductions}
                },
            ]}))
        algorithm_exps[algorithm].append(ref)
    fig, axes = plt.subplots(len(algorithms), len(reductions) + 1,
                             figsize=(7.166, 2),
                             sharey='col', sharex=True)
    plt.subplots_adjust(right=0.9, left=0.06, bottom=0.2, wspace=0.1)
    subplot_idx = {reduction: i for i, reduction in
                   enumerate([1] + reductions)}
    algorithm_name = {'full': 'Partial freeze of dictionary',
                      'full async': 'Partial freeze of dictionary\n'
                                    'Aynchronous parameter aggregation',
                      'tsp gram': 'Partial freeze of dictionary\n'
                                  'Aynchronous parameter aggregation\n'
                                  'Sketched code computation',
                      'icml': 'ICML'}
    for i, algorithm in enumerate(['full', 'full async', 'tsp gram', 'icml']):
        for exp in algorithm_exps[algorithm]:
            reduction = exp['config']['reduction']
            idx = subplot_idx[reduction]
            iter = np.array(exp['info']['iter'])
            profile = np.array(exp['info']['profiling'])
            profile = profile[:, [0, 3, 2, 1, 4, 7]]
            labels = np.array(
                ['',
                 '$\\boldsymbol{\\beta}_t$',
                 '$(\\bar \\mathbf{B}_t, \\bar \\mathbf{C}_t)$',
                 '$\\boldsymbol{\\alpha}_t$',
                 '$\\mathbf{G}_t$',
                 '$\\mathbf{D}_t$',
                 'Async update'])
            average_time = np.zeros(
                (profile.shape[0] - 1, profile.shape[1] + 1))
            average_time[:, 1:] = (profile[1:] - profile[:-1]) \
                                  / (iter[1:] - iter[:-1])[:, np.newaxis]
            average_time = np.cumsum(average_time, axis=1)

            palette = sns.color_palette("deep", profile.shape[1])
            for j in range(1, profile.shape[1] + 1):
                axes[i, idx].fill_between(iter[1:], average_time[:, j],
                                          average_time[:, j - 1],
                                          facecolor=palette[j - 1],
                                          label=labels[j])
            axes[i, idx].set_xscale('log')
            # axes[i, idx].set_ylim([0, np.max(average_time[:, 5])])
            if i == 0:
                axes[i, idx].annotate("Reduction: %i" % reduction,
                                      xy=(0.5, 1.1),
                                      ha='center',
                                      xycoords='axes fraction')
            if idx > 0:
                yticks = axes[i, idx].get_yticks().tolist()
                yticks[0] = ''
                axes[i, idx].set_yticklabels(yticks)
            sns.despine(fig, axes[i, idx])
            for tick in axes[i, idx].yaxis.get_major_ticks():
                tick.label.set_fontsize(6)

        axes[i, -1].annotate(algorithm_name[algorithm],
                             xy=(1, 0.5),
                             xycoords='axes fraction',
                             ha='left',
                             va='center')
        axes[i, 0].set_ylabel('Time')
    handles, labels = axes[0, -1].get_legend_handles_labels()
    axes[-1, 0].legend(reversed(handles), reversed(labels),
                       frameon=False,
                       loc='upper left',
                       ncol=5,
                       bbox_to_anchor=(0.3, -.2), )
    axes[-1, 0].annotate('Computation', xytext=(0, -8),
                         textcoords='offset points',
                         xy=(-.2, -.2), xycoords='axes fraction', va='top',
                         ha='left')
    axes[-1, -1].set_xlabel('Iteration')
    ticklab = axes[i, -1].xaxis.get_ticklabels()[0]
    trans = ticklab.get_transform()
    axes[-1, -1].xaxis.set_label_coords(axes[i, -1].get_xlim()[1] * 3,
                                        0, transform=trans)

    plt.savefig(name + '_profiling.pdf')
    plt.show()


@plot_ex.command
def plot_components(name):
    db, fs = get_connections()
    parent_ids = get_parent_ids()
    algorithms = {
        'tsp gram': ['full', 'average', 'async'],
        'full': ['full', 'full', 'async']
    }
    algorithm_keys = ['full', 'tsp gram']
    reductions = [6, 12, 24]
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                "config.reduction": {"$in": reductions}
            }, {
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1}
            ]}))

    for i, algorithm in enumerate(algorithm_keys):
        exps = algorithm_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            print('Plot')
            reduction = exp['config']['reduction']
            if name == 'compare_aviris':
                with NamedTemporaryFile(suffix='.npy',
                                        dir='/run/shm') as f:
                    f.write(fs.get(exp['artifacts'][-1]).read())
                    components = np.load(f.name)
                    fig = plot_patches(components,
                                       shape=exp['info']['data_shape'])
            else:
                with open(expanduser(u"~/artifacts/tsp_{}.nii.gz"
                                             .format(reduction)),
                          'wb+') as f:
                    f.write(fs.get(exp['artifacts'][-1]).read())
                    fig = display_maps(f.name, 0)
            fig.suptitle('%s %s' % (algorithm, reduction))
            plt.savefig('exp_%s_%i.pdf' % (algorithm, reduction))
    plt.show()


@plot_ex.command
def plot_method_comparison(xlim, ylim, ylim_zoom, xlim_zoom, name):
    db, fs = get_connections()
    parent_ids = get_parent_ids()
    algorithms = {
        'tsp gram': ['full', 'average', 'async'],
        'tsp': ['average', 'average', 'async'],
        'full': ['full', 'full', 'async'],
        'icml': ['masked', 'masked', 'async']
    }
    algorithm_keys = [
        'full',
        # 'tsp',
        'tsp gram',
        'icml'
    ]
    reductions = [
        # 6,
        12,
        24
    ]
    algorithm_exps = {}
    ref = get_ref_data()
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = []
        for reduction in reductions:
            res = db.find({"$or":
                [{
                    'info.parent_id': {"$in": parent_ids},
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
        'tsp': 'Variance reduction',
        'tsp gram': 'Variance reduction',
        'icml': 'Masked loss',
        'full': 'No subsampling'}

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
    axes[2].xaxis.set_label_coords(axes[1].get_xlim()[1] * 0.7, 0,
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
                         rect_0[0] * 1.7, rect_0[1] + 1.2 * rect_len[1]),
                     va='center', ha='right')

    axes[1].legend(handles_reduction, labels_reduction,
                                  bbox_to_anchor=(1.45, 1.5),
                                  loc='upper center',
                                  ncol=2, title='Subsampling ratio',
                                  frameon=False)
    # axes[1].add_artist(first_legend)

    l = axes[2].legend(handles_method,
                       labels_method,
                       bbox_to_anchor=(1.45, 1.3), loc='upper center',
                       ncol=1,
                       title='Code computation',
                       frameon=False)
    plt.savefig(name + '_method_comparison.pdf')
    plt.show()


@plot_ex.automain
def plot_debug(xlim, ylim, ylim_zoom, xlim_zoom, name):
    db, fs = get_connections()
    parent_ids = get_parent_ids()
    algorithms = {
        'tsp gram': ['full', 'average', 'async'],
        'full': ['full', 'full', 'async'],
        # 'icml': ['masked', 'masked', 'async']
    }
    algorithm_keys = ['full', 'tsp gram']
    reductions = [1, 6, 12, 24]
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                "info.profiling": {"$ne": []},
                'status': {'$in': ['RUNNING', 'COMPLETED']},
                "config.reduction": {"$in": reductions}
            }]}))
        algorithm_exps[algorithm].append(get_ref_data())
    n_red = len(reductions) + 1

    # Plotting
    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2, light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip([1] + reductions, colormap)}
    color_dict[1] = ref_colormap[0]
    ref = np.finfo('f8').max
    for this_algorithm in algorithm_exps:
        for exp in algorithm_exps[this_algorithm]:
            score = np.array(exp['info']['score'])
            if len(score) == 0:
                print(this_algorithm, exp['info'])
            else:
                ref = min(ref, np.min(np.array(exp['info']['score'])))

    ref *= 1 - ylim_zoom[0]

    style = {'icml': ':', 'tsp': '-', 'full': '--', 'tsp gram': '-'}
    names = {
        'tsp': 'Variance reduction',
        'tsp gram': 'Variance reduction',
        'icml': 'Masked loss',
        'full': 'No subsampling'}
    fig, axes = plt.subplots(5, 3, figsize=(12, 10))
    fig.subplots_adjust(top=0.95, bottom=0.2, wspace=0.3)
    for i, algorithm in enumerate(algorithms):
        exps = algorithm_exps[algorithm]
        axes[i, 1].annotate(algorithm, xy=(0.5, 0.8),
                            xycoords='axes fraction')
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            updated_params = exp['info']['updated_params']
            score = np.array(exp['info']['score'])
            iter = np.array(exp['info']['iter'])
            time = np.array(exp['info']['profiling'])[:, 5] / 3600
            # iter = np.array(exp['info']['time'])
            # diff_time = (time[1:] - time[:-1]) / (iter[1:] - iter[:-1])
            # mean_diff_time = np.min(diff_time)
            # time = iter[0] + iter * mean_diff_time
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            rel_score = (score - ref) / ref
            axes[i, 0].plot(iter + 10, score,
                            label="Reduction = %i" % reduction,
                            color=color,
                            linestyle=style[algorithm],
                            zorder=reduction if reduction != 1 else 100,
                            # marker='o',
                            markersize=2)
            axes[i, 1].plot(time, rel_score,
                            label="Reduction = %i" % reduction,
                            color=color,
                            linestyle=style[algorithm],
                            zorder=reduction if reduction != 1 else 100,
                            # marker='o',
                            markersize=2)
            axes[i, 2].plot(iter + 10,
                            time,
                            label="Reduction = %i" % reduction,
                            color=color,
                            linestyle=style[algorithm],
                            zorder=reduction if reduction != 1 else 100,
                            markersize=2)
            if reduction != 1 or (reduction == 1 and algorithm == 'tsp'):
                axes[-1, 0].plot(iter + 10, score,
                                 label="Reduction = %i" % reduction,
                                 color=color,
                                 linestyle=style[algorithm],
                                 markersize=2)
                axes[-1, 1].plot(time, rel_score,
                                 label="Reduction = %i" % reduction,
                                 color=color,
                                 linestyle=style[algorithm],
                                 zorder=reduction if reduction != 1 else 100,
                                 markersize=2)
                axes[-1, 2].plot(iter + 10,
                                 time,
                                 linestyle=style[algorithm],
                                 label="Reduction = %i" % reduction,
                                 zorder=reduction if reduction != 1 else 100,
                                 color=color,
                                 markersize=2)
            axes[i, 0].set_ylabel('Test loss')
            axes[i, 1].set_ylabel('Test loss (relative)')
            axes[i, 0].set_xscale('log')
            # axes[i, 1].set_yscale('log')
            axes[i, 1].set_xscale('log')
            # axes[i, 2].set_yscale('log')
            # axes[i, 2].set_xscale('log')
            axes[i, 2].set_ylabel('Time (s)')
            axes[i, 1].set_ylim(ylim_zoom)
            axes[i, 1].set_xlim(xlim_zoom)

            axes[i, 0].set_ylim(ylim)
            axes[i, 0].set_xlim(xlim)

    for i in range(3):
        for j in range(3):
            sns.despine(fig, axes[i, j])
    axes[-1, 0].set_xlabel('Iter')
    axes[-1, 1].set_xlabel('Time (s)')
    axes[-1, 2].set_xlabel('Iter')
    axes[-1, 0].set_xscale('log')
    # axes[-1, 1].set_yscale('log')
    axes[-1, 1].set_xscale('log')
    # axes[-1, 2].set_yscale('log')
    # axes[-1, 2].set_xscale('log')
    axes[-1, 1].set_ylim(ylim_zoom)
    axes[-1, 1].set_xlim(xlim_zoom)
    axes[-1, 0].set_ylim(ylim)
    axes[-1, 0].set_xlim(xlim)

    handles, labels = axes[3, 0].get_legend_handles_labels()

    first_legend = axes[3, 0].legend(handles[:n_red], labels[:n_red],
                                     bbox_to_anchor=(0, -.3),
                                     loc='upper left',
                                     ncol=1)
    axes[3, 0].add_artist(first_legend)

    axes[3, 0].legend(handles[::n_red], algorithm_keys,
                      bbox_to_anchor=(1, -.3), loc='upper left', ncol=1)

    plt.savefig(name + '_debug.pdf')
    plt.show()
