from os.path import expanduser
from tempfile import NamedTemporaryFile

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

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    sub_db = 'sacred'
    exp_name = 'compare_hyperspectral'
    name = 'compare_hyperspectral'
    status = 'INTERRUPTED'
    ylim_zoom = None
    xlim_zoom = None
    ylim = None
    xlim = None
    oid = None
    AB_agg = 'full'
    plot_components = False
    plot_type = 'debug'


@plot_ex.named_config
def aviris():
    pass


@plot_ex.named_config
def adhd():
    sub_db = 'fmri'
    exp_name = 'compare_adhd'
    name = 'compare_adhd'
    status = 'RUNNING'
    ylim_zoom = [.9e-2, 2e-1]
    ylim = [21000, 34000]
    AB_agg = 'full'


@plot_ex.named_config
def hcp():
    sub_db = 'sacred'
    exp_name = ['compare_hcp']
    oid = ['57f22495fb5c86780390bca7',
           '57f22489fb5c8677ec4a8414']
    name = 'compare_hcp'
    status = 'RUNNING'
    ylim_zoom = [.1e-2, 5e-2]
    xlim_zoom = [1e3, 3e4]
    xlim = [1e1, 3e4]
    ylim = [97000, 106500]
    AB_agg = 'full'
    plot_components = False


@plot_ex.named_config
def hcp_first_run():
    sub_db = 'sacred'
    exp_name = ['compare_hcp', 'compare_hcp_high_red']
    oid = ["57ee59fdfb5c866503c34aef", "57ee917cfb5c869f5d171f60"]
    name = 'hcp_first_run'
    status = 'INTERRUPTED'
    ylim_zoom = [1e-2, 10e-2]
    xlim_zoom = [1e4, 2e5]
    ylim = [95000, 106500]
    xlim = [1e1, 3e5]
    AB_agg = 'full'


@plot_ex.capture
def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(exp_name, oid, status, xlim, ylim, ylim_zoom,
         xlim_zoom, AB_agg, name, plot_type, plot_components):
    db, fs = get_connections()
    if oid != None:
        oid = [ObjectId(this_oid) for this_oid in oid]
        parent_exps = db.find({'_id': {"$in": oid},
                               'status': status
                               }).sort('_id', -1)
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]
    else:
        if not isinstance(exp_name, (list, tuple)):
            exp_name = [exp_name]
        parent_exps = db.find({'experiment.name': {"$in": exp_name},
                               'status': status
                               }).sort('_id', -1)[:2]
        parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]

    print(parent_ids)
    algorithms = {
        'icml': ['masked', 'masked', AB_agg],
        'tsp': ['full', 'average', AB_agg],
        'full': ['full', 'full', 'full']
    }
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, this_AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": this_AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                "config.reduction": {"$ne": 1},
                # "config.reduction": {"$in": [4, 12, 24]}
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
    style = {'icml': ':', 'tsp': '-', 'full': '--'}
    names = {
        'tsp': 'Variance reduction',
             'icml': 'Masked loss',
             'full': 'No subsampling'}
    algorithm_keys = ['tsp', 'full', 'icml']
    if plot_type == 'debug':
        fig, axes = plt.subplots(4, 3, figsize=(12, 10))
        fig.subplots_adjust(top=0.95, bottom=0.2, wspace=0.3)
        for i, algorithm in enumerate(algorithms):
            exps = algorithm_exps[algorithm]
            for exp in sorted(exps,
                              key=lambda exp: int(exp['config']['reduction'])):
                axes[i, 1].annotate(algorithm, xy=(0.5, 0.8),
                                    xycoords='axes fraction')
                updated_params = exp['info']['updated_params']
                score = np.array(exp['info']['score'])
                iter = np.array(exp['info']['iter'])
                time = np.array(exp['info']['time'])
                # time = np.logspace(1e-1, log(time[-1], 10), time.shape[0])
                reduction = exp['config']['reduction']
                color = color_dict[reduction]
                rel_score = (score - ref) / ref
                axes[i, 0].plot(iter + 10, score,
                                label="Reduction = %i" % reduction,
                                color=color,
                                linestyle=style[algorithm],
                                # marker='o',
                                markersize=2)
                axes[i, 1].plot(time, rel_score,
                                label="Reduction = %i" % reduction,
                                color=color,
                                linestyle=style[algorithm],
                                # marker='o',
                                markersize=2)
                axes[i, 2].plot(iter + 10,
                                time,
                                label="Reduction = %i" % reduction,
                                color=color,
                                linestyle=style[algorithm],
                                markersize=2)
                axes[3, 0].plot(iter + 10, score,
                                label="Reduction = %i" % reduction,
                                color=color,
                                linestyle=style[algorithm],
                                markersize=2)
                axes[3, 1].plot(time, rel_score,
                                label="Reduction = %i" % reduction,
                                color=color,
                                linestyle=style[algorithm],
                                markersize=2)
                axes[3, 2].plot(iter + 10,
                                time,
                                linestyle=style[algorithm],
                                label="Reduction = %i" % reduction,
                                color=color,
                                markersize=2)
            axes[i, 0].set_ylabel('Test loss')
            axes[i, 1].set_ylabel('Test loss (relative)')
            axes[0, 1].set_xscale('log')
            axes[1, 1].set_xscale('log')
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
        axes[3, 0].set_xlabel('Iter')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 2].set_xlabel('Iter')
        axes[3, 2].set_yscale('log')
        axes[3, 2].set_xscale('log')
        # axes[3, 1].set_xscale('log')
        # axes[3, 1].set_yscale('log')
        axes[3, 0].set_xscale('log')

        axes[3, 1].set_ylim(ylim_zoom)
        axes[3, 1].set_xlim(xlim_zoom)

        axes[3, 0].set_ylim(ylim)
        axes[3, 0].set_xlim(xlim)

        handles, labels = axes[3, 0].get_legend_handles_labels()

        first_legend = axes[3, 0].legend(handles[:n_red], labels[:n_red],
                                         bbox_to_anchor=(0, -.3),
                                         loc='upper left',
                                         ncol=1)
        axes[3, 0].add_artist(first_legend)

        axes[3, 0].legend(handles[::n_red], labels[::n_red],
                          bbox_to_anchor=(1, -.3), loc='upper left', ncol=1)
        print('Done plotting figure')
        plt.savefig(name + AB_agg + '.pdf')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.166, 1.5))
        fig.subplots_adjust(right=0.85, left=0.09, bottom=0.13, top=0.9,
                            wspace=0.18)
        for i, algorithm in enumerate(algorithm_keys):
            exps = algorithm_exps[algorithm]
            for exp in sorted(exps,
                              key=lambda exp: int(exp['config']['reduction'])):
                score = np.array(exp['info']['score'])
                iter = np.array(exp['info']['iter'])
                time = np.array(exp['info']['time'])
                # time = np.logspace(1e-1, log(time[-1], 10), time.shape[0])
                reduction = exp['config']['reduction']
                color = color_dict[reduction]
                rel_score = (score - ref) / ref
                axes[0].plot(time, score,
                             label="r = %i" % reduction if reduction != 1 else "None",
                             color=color,
                             linestyle=style[algorithm],
                             markersize=2)
                axes[1].plot(time,
                             rel_score,
                             linestyle=style[algorithm],
                             label="r = %i" % reduction if reduction != 1 else "None",
                             color=color,
                             markersize=2)
        axes[0].set_ylabel('Test loss (left: relative)')
        # axes[1].set_ylabel('Test loss (relative)')
        for j in range(2):
            axes[j].set_xscale('log')
            sns.despine(fig, axes[j])
        axes[1].set_xlabel('Time')

        axes[1].set_yscale('log')
        axes[1].set_ylim(ylim_zoom)
        axes[1].set_xlim(xlim_zoom)
        axes[0].set_ylim(ylim)
        axes[0].set_xlim(xlim)
        axes[1].set_xticks([1000, 10000])
        axes[1].set_xticklabels(["1000 s", "10000 s"])
        axes[1].set_yticks([1e-2, 1e-3])
        axes[1].set_yticklabels(["1\%", ".1\%"])
        axes[0].set_xticks([10, 100, 1000, 10000])
        axes[0].set_xticklabels(["10 s", "100 s", "1000 s", "10000 s"])

        axes[0].xaxis.set_label_coords(1.1, -.065)
        axes[1].xaxis.set_label_coords(1.1, -.065)
        rect_0 = (xlim_zoom[0], (1 + ylim_zoom[0]) * ref)
        rect_len = (
        xlim_zoom[1] - rect_0[0], (1 + ylim_zoom[1]) * ref - rect_0[1])
        patch = patches.Rectangle(
            rect_0,
            rect_len[0],
            rect_len[1],
            fill=False,
            linestyle='dashed'  # remove background
        )
        axes[0].add_patch(
            patches.Rectangle(
                rect_0,
                rect_len[0] * 2,
                rect_len[1],
                fill=False,
                linestyle='dashed'  # remove background
            )
        )
        axes[0].annotate("Zoom", xycoords='data',
                         xy=(rect_0[0], (rect_0[1] + rect_len[1]) * 1.001))

        handles, labels = axes[1].get_legend_handles_labels()

        first_legend = axes[1].legend(handles[:n_red], labels[:n_red],
                                      bbox_to_anchor=(0.8, 1.1),
                                      loc='upper left',
                                      ncol=2, title='Subsampling ratio',
                                      frameon=False)
        axes[1].add_artist(first_legend)

        l = axes[1].legend(handles[::n_red],
                       [names[algorithm] for algorithm in algorithm_keys],
                       bbox_to_anchor=(0.8, 0.65), loc='upper left', ncol=1,
                       title='Code computation method',
                       frameon=False)
        # l.get_title().set_ha('center')
        print('Done plotting figure')
        plt.savefig(name + AB_agg + '.pdf')

    if plot_components:
        for i, algorithm in enumerate(['tsp']):
            exps = algorithm_exps[algorithm]
            for exp in sorted(exps,
                              key=lambda exp: int(exp['config']['reduction'])):
                print('Plot')
                reduction = exp['config']['reduction']
                if exp_name == 'compare_hyperspectral':
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
