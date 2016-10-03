from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from nilearn._utils import check_niimg
from pymongo import MongoClient
import numpy as np
import seaborn.apionly as sns

from sacred.experiment import Experiment

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    sub_db = 'sacred'
    exp_name = 'compare_hyperspectral'
    status = 'INTERRUPTED'
    ylim_log = [.9e-2, 4e-1]
    ylim = [9000, 16000]
    AB_agg = 'full'


@plot_ex.named_config
def aviris():
    pass


@plot_ex.named_config
def adhd():
    sub_db = 'fmri'
    exp_name = 'compare_adhd'
    status = 'RUNNING'
    ylim_log = [.9e-2, 2e-1]
    ylim = [21000, 34000]
    AB_agg = 'full'


@plot_ex.named_config
def hcp():
    sub_db = 'fmri'
    exp_name = 'compare_hcp'
    status = 'RUNNING'
    ylim_log = [.9e-2, 2e-1]
    ylim = [97000, 106500]
    AB_agg = 'full'

@plot_ex.named_config
def hcp_high_red():
    sub_db = 'sacred'
    exp_name = 'compare_hcp_high_red'
    status = 'RUNNING'
    ylim_log = [.9e-2, 2e-1]
    ylim = [97000, 106500]
    AB_agg = 'full'


@plot_ex.capture
def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(exp_name, status, ylim, ylim_log, AB_agg):
    db, fs = get_connections()

    parent_exp = db.find({'experiment.name': exp_name,
                          'status': status
                          }).sort('_id', -1)[0]
    parent_id = parent_exp['_id']

    algorithms = {
        'icml': ['masked', 'masked', AB_agg],
        'tsp': ['full', 'average', AB_agg],
        'full': ['full', 'average', 'async']
    }
    algorithm_exps = {}
    for algorithm in algorithms:
        (G_agg, Dx_agg, this_AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': parent_id,
                "config.AB_agg": this_AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                'config.reduction': {
                    '$in': [16, 20, 24]},
            }, {
                'info.parent_id':
                    parent_id,
                "config.reduction": 1}
            ]}))

    # Plotting
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    fig.subplots_adjust(top=0.95, bottom=0.2, wspace=0.3)

    colormap = sns.cubehelix_palette(5, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(5, start=2, rot=0.2, light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip([16, 20, 24], colormap)}
    color_dict[1] = ref_colormap[0]
    ref = min([np.min(np.array(exp['info']['score'])) for exp in
               algorithm_exps[algorithm]
               for algorithm in algorithm_exps]) * 0.99
    style = {'icml': ':', 'tsp': '-', 'full': '--'}
    for i, algorithm in enumerate(['tsp', 'icml', 'full']):
        exps = algorithm_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            axes[i, 1].annotate(algorithm, xy=(0.5, 0.8),
                                xycoords='axes fraction')
            updated_params = exp['info']['updated_params']
            score = np.array(exp['info']['score'])
            iter = np.array(exp['info']['iter'])
            time = np.array(exp['info']['time'])
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            rel_score = (score - ref) / ref
            axes[i, 0].plot(iter + 10, score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            # marker='o',
                            markersize=2)
            axes[i, 1].plot(time, rel_score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            # marker='o',
                            markersize=2)
            axes[i, 2].plot(iter + 10,
                            time,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            markersize=2)
            axes[3, 0].plot(iter + 10, score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            markersize=2)
            axes[3, 1].plot(time, rel_score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            markersize=2)
            axes[3, 2].plot(iter + 10,
                            time,
                            linestyle=style[algorithm],
                            label="Reduction = %i" % reduction, color=color,
                            markersize=2)
        axes[i, 0].set_ylabel('Test loss')
        axes[i, 1].set_ylabel('Test loss (relative)')
        for j in range(2):
            axes[i, j].set_xscale('log')
        axes[i, 1].set_yscale('log')
        axes[i, 2].set_yscale('log')
        axes[i, 2].set_xscale('log')
        axes[i, 2].set_ylabel('Time (s)')
        axes[i, 1].set_ylim(ylim_log)
        axes[i, 0].set_ylim(ylim)
    for i in range(3):
        for j in range(3):
            sns.despine(fig, axes[i, j])
    axes[3, 0].set_xlabel('Iter')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 2].set_xlabel('Iter')
    axes[3, 2].set_yscale('log')
    axes[3, 2].set_xscale('log')
    axes[3, 1].set_yscale('log')
    axes[3, 1].set_xscale('log')
    axes[3, 0].set_xscale('log')
    axes[3, 1].set_ylim(ylim_log)
    axes[3, 0].set_ylim(ylim)
    handles, labels = axes[3, 0].get_legend_handles_labels()

    first_legend = axes[3, 0].legend(handles[:5], labels[:5],
                                     bbox_to_anchor=(0, -.3), loc='upper left',
                                     ncol=1)
    axes[3, 0].add_artist(first_legend)

    axes[3, 0].legend(handles[::3], ['tsp', 'icml', 'full'],
                      bbox_to_anchor=(1, -.3), loc='upper left', ncol=1)
    print('Done plotting figure')
    plt.savefig(exp_name + AB_agg + '.pdf')
    plt.show()
    plt.close(fig)
    # for i, algorithm in enumerate(['tsp']):
    #     exps = algorithm_exps[algorithm]
    #     for exp in sorted(exps,
    #                       key=lambda exp: int(exp['config']['reduction'])):
    #         print('Plot')
    #         reduction = exp['config']['reduction']
    #         if exp_name == 'compare_hyperspectral':
    #             with NamedTemporaryFile(suffix='.npy', dir='/run/shm') as f:
    #                 f.write(fs.get(exp['artifacts'][-1]).read())
    #                 components = np.load(f.name)
    #                 fig = plot_patches(components,
    #                                    shape=exp['info']['data_shape'])
    #         else:
    #             with NamedTemporaryFile(suffix='.nii.gz', dir='/run/shm') as f:
    #                 f.write(fs.get(exp['artifacts'][-1]).read())
    #                 fig = display_maps(f.name, 0)
    #         fig.suptitle('%s %s' % (algorithm, reduction))
