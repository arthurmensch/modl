from os.path import expanduser
from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from nilearn._utils import check_niimg
from pymongo import MongoClient
import numpy as np
import seaborn.apionly as sns
from math import log
from sacred.experiment import Experiment

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    sub_db = 'sacred'
    exp_name = 'compare_hyperspectral'
    name = 'compare_hyperspectral'
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
    name = 'compare_adhd'
    status = 'RUNNING'
    ylim_log = [.9e-2, 2e-1]
    ylim = [21000, 34000]
    AB_agg = 'full'


@plot_ex.named_config
def hcp():
    sub_db = 'fmri'
    exp_name = 'compare_hcp'
    name = 'compare_hcp'
    status = 'RUNNING'
    ylim_log = [.9e-2, 2e-1]
    ylim = [97000, 106500]
    AB_agg = 'full'

@plot_ex.named_config
def hcp_high_red():
    sub_db = 'sacred'
    exp_name = ['compare_hcp', 'compare_hcp_high_red']
    name = 'compare_hcp'
    status = 'RUNNING'
    ylim_log = [95000, 100000]
    xlim_log = [1e3, 2e5]
    ylim = [95000, 106500]
    AB_agg = 'full'


@plot_ex.capture
def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(exp_name, status, ylim, ylim_log, xlim_log, AB_agg, name):
    db, fs = get_connections()
    if not isinstance(exp_name, (list, tuple)):
        exp_name = [exp_name]
    parent_exps = db.find({'experiment.name': {"$in": exp_name},
                          'status': status
                          }).sort('_id', -1)[:len(exp_name)]
    parent_ids = [parent_exp['_id'] for parent_exp in parent_exps]

    algorithms = {
        'icml': ['masked', 'masked', AB_agg],
        'tsp': ['full', 'average', AB_agg],
        'full': ['full', 'full', 'full']
    }
    algorithm_exps = {}
    reductions = [4, 12, 24]
    n_red = len(reductions) + 1
    for algorithm in algorithms:
        (G_agg, Dx_agg, this_AB_agg) = algorithms[algorithm]
        algorithm_exps[algorithm] = list(db.find({"$or":
            [{
                'info.parent_id': {"$in": parent_ids},
                "config.AB_agg": this_AB_agg,
                "config.G_agg": G_agg,
                "config.Dx_agg": Dx_agg,
                'config.reduction': {
                    '$in': reductions},
            }, {
                'info.parent_id':
                    {"$in": parent_ids},
                "config.reduction": 1}
            ]}))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), squeeze=False)
    fig.subplots_adjust(top=0.95, bottom=0.2, wspace=0.3)

    colormap = sns.cubehelix_palette(n_red, rot=0.3, light=0.85, reverse=False)
    ref_colormap = sns.cubehelix_palette(n_red, start=2, rot=0.2, light=0.7,
                                         reverse=False)
    color_dict = {reduction: color for reduction, color in
                  zip([1] + reductions, colormap)}
    color_dict[1] = ref_colormap[0]
    ref = min([np.min(np.array(exp['info']['score'])) for exp in
               algorithm_exps[algorithm]
               for algorithm in algorithm_exps]) * 0.99
    style = {'icml': ':', 'tsp': '-', 'full': '--'}
    for i, algorithm in enumerate(['tsp', 'icml', 'full']):
        exps = algorithm_exps[algorithm]
        for exp in sorted(exps,
                          key=lambda exp: int(exp['config']['reduction'])):
            # axes[i, 1].annotate(algorithm, xy=(0.5, 0.8),
            #                     xycoords='axes fraction')
            updated_params = exp['info']['updated_params']
            score = np.array(exp['info']['score'])
            iter = np.array(exp['info']['iter'])
            time = np.array(exp['info']['time'])
            # time = np.logspace(1e-1, log(time[-1], 10), time.shape[0])
            reduction = exp['config']['reduction']
            color = color_dict[reduction]
            rel_score = (score - ref) / ref
            axes[0, 0].plot(time, score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            markersize=2)
            axes[0, 1].plot(time, score,
                            label="Reduction = %i" % reduction, color=color,
                            linestyle=style[algorithm],
                            markersize=2)
    for j in range(2):
        sns.despine(fig, axes[0, j])
    axes[0, 0].set_xlabel('Iter')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 1].set_xscale('log')
    axes[0, 0].set_xscale('log')
    axes[0, 1].set_ylim(ylim_log)
    axes[0, 1].set_xlim(xlim_log)
    axes[0, 0].set_ylim(ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()

    first_legend = axes[0, 0].legend(handles[:n_red], labels[:n_red],
                                     bbox_to_anchor=(0, -.3), loc='upper left',
                                     ncol=1)
    axes[0, 0].add_artist(first_legend)

    axes[0, 0].legend(handles[::n_red], ['tsp', 'icml', 'full'],
                      bbox_to_anchor=(1, -.3), loc='upper left', ncol=1)
    print('Done plotting figure')
    plt.savefig(name + AB_agg + '.pdf')
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
    #             with open(expanduser("~/artifacts/tsp_%s.nii.gz" % reduction), 'wb+') as f:
    #                 f.write(fs.get(exp['artifacts'][-1]).read())
    #                 fig = display_maps(f.name, 0)
    #         fig.suptitle('%s %s' % (algorithm, reduction))

    plt.show()
    plt.close(fig)
