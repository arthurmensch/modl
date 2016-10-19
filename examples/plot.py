from os.path import expanduser
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use('Qt5Agg')

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
    exp_name = ['compare_hcp']
    # oid = ['57f22495fb5c86780390bca7',
    #        '57f22489fb5c8677ec4a8414']
    # oid = ['5805374bfb5c8663ab366cf4',
    #        '5804f8d4fb5c862f0efe72fd',
    #        '57f22489fb5c8677ec4a8414']
    # oid = ['58075073fb5c866f3e7a89e9',
    #         '58074ff6fb5c866eb5eb4ecb']
    name = 'compare_hcp'
    ylim_zoom = [1e-3, 5e-2]
    # xlim_zoom = [0.5, 100]
    # xlim = [1e-3, 100]
    # ylim = [96600, 106000]

@plot_ex.named_config
def hcp_compare():
    sub_db = 'sacred'
    exp_name = ['compare_hcp']
    oid = ['57f22495fb5c86780390bca7',
           '57f22489fb5c8677ec4a8414']
    name = 'method_comparison'
    plot_type = 'method_comparison'
    ylim_zoom = [.1e-2, 5e-2]
    xlim_zoom = [1e3, 2e4]
    xlim = [1e1, 2e4]
    ylim = [97000, 106500]

@plot_ex.capture
def get_connections(sub_db):
    client = MongoClient('localhost', 27017)
    # client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(exp_name, oid, xlim, ylim, ylim_zoom,
         xlim_zoom, name, plot_type):
    db, fs = get_connections()
    if oid is not None:
        oid = [ObjectId(this_oid) for this_oid in oid]
        parent_exps = db.find({'_id': {"$in": oid},
                               }).sort('_id', -1)
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]
    else:
        if not isinstance(exp_name, (list, tuple)):
            exp_name = [exp_name]
        parent_exps = db.find({'experiment.name': {"$in": exp_name},
                               }).sort('_id', -1)[:1]
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]

    print(parent_ids)
    algorithms = {
        # 'icml': ['masked', 'masked', 'full'],
        'tsp': ['average', 'average', 'full'],
        'tsp gram': ['average', 'average', 'async'],
        # 'full': ['full', 'full', 'full']
    }
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                "config.reduction": {"$in": [6, 12, 24]}
            }, {
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1}
            ]}))

    reductions = np.unique(np.array([exp['config']
                                     ['reduction']
                                     for this_algorithm in algorithms
                                     for exp in algorithm_exps[this_algorithm]
                                     ]))
    # reductions = [1, 4, 12, 24]
    n_red = len(reductions)

    # Plotting
    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2, light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip(reductions, colormap)}
    color_dict[1] = ref_colormap[0]


    ref = min([np.min(np.array(exp['info']['score']))
               for this_algorithm in algorithm_exps for exp in
               algorithm_exps[this_algorithm]]) * (1 - ylim_zoom[0])

    style = {'icml': ':', 'tsp': '-', 'full': '--', 'tsp gram': '-.'}
    names = {
        'tsp': 'Variance reduction',
        'tsp_gram': 'Variance reduction (full Gram)',
        'icml': 'Masked loss',
        'full': 'No subsampling'}
    algorithm_keys = ['tsp', 'icml']
    if plot_type == 'debug':
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
                time = np.array(exp['info']['time'])
                # time = iter * time[-1] / iter[-1] + 1
                print(time)
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
                axes[i, 1].set_yscale('log')
                axes[i, 1].set_xscale('log')
                axes[i, 2].set_yscale('log')
                axes[i, 2].set_xscale('log')
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
        axes[-1, 1].set_yscale('log')
        axes[-1, 1].set_xscale('log')
        axes[-1, 2].set_yscale('log')
        axes[-1, 2].set_xscale('log')
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
        print('Done plotting figure')
        # plt.savefig(name + '.pdf')

    elif plot_type == 'method_comparison':
        fig, axes = plt.subplots(1, 2, figsize=(7.166, 1.5))
        fig.subplots_adjust(right=0.75, left=0.09, bottom=0.13, top=0.9,
                            wspace=0.24)
        for i, algorithm in enumerate(algorithm_keys):
            exps = algorithm_exps[algorithm]
            for exp in sorted(exps,
                              key=lambda exp: int(exp['config']['reduction'])):
                score = np.array(exp['info']['score'])
                iter = np.array(exp['info']['iter'])
                time = np.array(exp['info']['time']) / 3600
                # time = iter / 2e4
                reduction = exp['config']['reduction']
                color = color_dict[reduction]
                rel_score = (score - ref) / ref
                axes[0].plot(time, score,
                             label="r = %i" % reduction if reduction != 1 else "None",
                             zorder=reduction if reduction != 1 else 1,
                             color=color,
                             linestyle=style[algorithm],
                             markersize=2)
                axes[1].plot(time,
                             rel_score,
                             linestyle=style[algorithm],
                             label="r = %i" % reduction if reduction != 1 else "None",
                             zorder=reduction if reduction != 1 else 1,
                             color=color,
                             markersize=2)
        axes[0].set_ylabel('Test objective function')
        axes[1].set_ylabel('(relative to lowest value)', rotation=0)
        axes[1].yaxis.set_label_coords(0.3, 0.01)
        for j in range(2):
            axes[j].set_xscale('log')
            sns.despine(fig, axes[j])
        axes[1].set_xlabel('Time')
        axes[1].xaxis.set_label_coords(1.18, -.08)

        axes[1].set_yscale('log')
        axes[1].set_ylim([ylim_zoom[0], ylim_zoom[1]])
        axes[1].set_xlim(xlim_zoom)
        axes[0].set_ylim(ylim)
        axes[0].set_xlim(xlim)
        if 'compare_hcp' in exp_name:
            axes[1].set_xticks([1, 10, 100])
            axes[1].set_xticklabels(["1 h", "10 h", "100 h"])
            axes[1].set_yticks([1e-3, 1e-2])
            axes[1].set_yticklabels(["0.1\%", "1\%"])
            axes[0].set_xticks([1 / 60, 1, 10, 100])
            axes[0].set_xticklabels(["1 min", "1 h", "10 h", "100 h"])

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
                         rect_0[0] * 0.9, rect_0[1] + 0.5 * rect_len[1]), va='center', ha='right')

        handles, labels = axes[1].get_legend_handles_labels()

        first_legend = axes[1].legend(handles[:n_red], labels[:n_red],
                                      bbox_to_anchor=(1.05, 1.1),
                                      loc='upper left',
                                      ncol=2, title='Subsampling ratio',
                                      frameon=False)
        axes[1].add_artist(first_legend)

        l = axes[1].legend(handles[n_red - 1::n_red],
                           [names[algorithm] for algorithm in algorithm_keys],
                           bbox_to_anchor=(1.05, 0.65), loc='upper left',
                           ncol=1,
                           title='Code computation',
                           frameon=False)
        plt.savefig(name + '.pdf')

    elif plot_type == 'plot_components':
        for i, algorithm in enumerate(['tsp']):
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
                plt.savefig('exp_%i.pdf' % reduction)
    plt.show()
