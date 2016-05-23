import datetime
import fnmatch
import glob
import json
import os
from math import log
from os.path import expanduser, join

import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.apionly as sns
from joblib import delayed, Parallel
from matplotlib import gridspec
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.decomposition.base import explained_variance
from nilearn.decomposition.dict_fact import DictMF
from nilearn.decomposition.sparse_pca import objective_function
from nilearn.input_data import MultiNiftiMasker
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from sklearn.utils import check_random_state


def load_data(init='rsn70',
              dataset='hcp'):
    if dataset == 'hcp':
        with open(expanduser('~/data/HCP_unmasked/data.json'), 'r') as f:
            data = json.load(f)
        for this_data in data:
            this_data['array'] += '.npy'
        mask_img = expanduser('~/data/HCP_mask/mask_img.nii.gz')
    elif dataset == 'adhd':
        with open(expanduser('~/data/ADHD_unmasked/data.json'), 'r') as f:
            data = json.load(f)
        mask_img = expanduser('~/data/ADHD_mask/mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask_img, smoothing_fwhm=4,
                              standardize=True)
    masker.fit()
    smith2009 = fetch_atlas_smith_2009()
    if init == 'rsn70':
        init = smith2009.rsn70
    elif init == 'rsn20':
        init = smith2009.rsn20
    dict_init = masker.transform(init)
    return masker, dict_init, sorted(data, key=lambda t: t['filename'])


def compute_exp_var(X, masker, filename, alpha=None):
    print('Computing explained variance')
    components = masker.transform(filename)
    densities = np.sum(components != 0) / components.size
    if alpha is None:
        exp_var = explained_variance(X, components,
                                     per_component=False).flat[0]
    else:
        exp_var = objective_function(X, components, alpha)
    return exp_var, densities


def analyse_l1l2_in_dir(output_dir, dataset='hcp', objective=False):
    masker, _, data = load_data(dataset=dataset)

    masker.set_params(smoothing_fwhm=None, standardize=False)
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    output_files = os.listdir(output_dir)
    records = []
    l1l2s = []
    for filename in sorted(fnmatch.filter(output_files, 'a_*.nii.gz'),
                           key=lambda t: int(t[2:-7]))[
                    ::int(exp_dict['reduction'])]:
        components = masker.transform(join(output_dir, filename))
        l1l2 = np.sum(np.abs(components)) / np.sqrt(np.sum(components ** 2))
        records.append(int(filename[2:-7]))
        l1l2s.append(l1l2)
    order = np.argsort(np.array(records))
    records = np.array(records)[order].tolist()
    l1l2s = np.array(l1l2s)[order].tolist()
    exp_dict['records'] = records
    exp_dict['l1l2s'] = l1l2s
    with open(join(output_dir, 'experiment_l1L2.json'), 'w+') as f:
        json.dump(exp_dict, f)


def analyse_exp_var_in_dir(output_dir, dataset='hcp', objective=False):
    masker, _, data = load_data(dataset=dataset)

    if dataset == 'hcp':
        data = data[400:404:4]
    elif dataset == 'adhd':
        data = data[36:]

    masker.set_params(smoothing_fwhm=None, standardize=False)
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    concatenated_data = [np.load(this_data['array']) for this_data in data]
    X = np.concatenate(concatenated_data, axis=0)
    output_files = os.listdir(output_dir)
    records = []
    exp_vars = []
    densities = []
    for filename in sorted(fnmatch.filter(output_files, 'a_*.nii.gz'),
                           key=lambda t: int(t[2:-7]))[
                    ::int(exp_dict['reduction'])]:
        exp_var, density = compute_exp_var(X, masker,
                                           join(output_dir, filename),
                                           alpha=exp_dict['alpha'] if objective
                                           else None)
        records.append(int(filename[2:-7]))
        exp_vars.append(exp_var)
        densities.append(density)
        exp_dict['records'] = records
        if objective:
            exp_dict['objective'] = exp_vars
        else:
            exp_dict['exp_vars'] = exp_vars
        exp_dict['densities'] = densities
        with open(join(output_dir, 'experiment.json'), 'w+') as f:
            json.dump(exp_dict, f)
    order = np.argsort(np.array(records))
    exp_vars = np.array(exp_vars)[order].tolist()
    densities = np.array(densities)[order].tolist()
    records = np.array(records)[order].tolist()
    exp_dict['records'] = records
    if objective:
        exp_dict['objective'] = exp_vars
    else:
        exp_dict['exp_vars'] = exp_vars
    exp_dict['densities'] = densities
    with open(join(output_dir, 'experiment_stat.json'), 'w+') as f:
        json.dump(exp_dict, f)


def single(output_dir, alpha, reduction, impute, dataset, init, records_range,
           random_state=0):
    masker, dict_init, data = load_data(dataset=dataset, init=init)
    n_components = dict_init.shape[0]
    dict_mf = DictMF(batch_size=20, reduction=reduction,
                     random_state=random_state,
                     learning_rate=1,
                     dict_init=dict_init,
                     alpha=alpha,
                     impute=impute,
                     l1_ratio=.5,
                     fit_intercept=False,
                     n_components=n_components,
                     backend='python',
                     debug=True,
                     )
    random_state = check_random_state(0)
    data = [data[i] for i in records_range]
    for e in range(5):
        random_state.shuffle(data)
        for i, this_data in enumerate(data):
            X = np.load(this_data['array'])
            dict_mf.partial_fit(X)
            print('Loaded record %i, '
                  ' seen rows: %i' % (i, dict_mf.counter_[0]))
            if i % 5 == 0:
                density = np.sum(dict_mf.Q_ != 0) / dict_mf.Q_.size
                print('Red. %.2f, '
                      'dictionary density: %.4f' % (dict_mf.reduction,
                                                    density))
                if density < 1e-3:
                    print('Dictionary is too sparse, giving up')
                    return
                components = masker.inverse_transform(dict_mf.Q_)
                components.to_filename(join(output_dir, 'a_%i.nii.gz'
                                            % dict_mf.counter_[0]))
                if dict_mf.debug:
                    with open(join(output_dir, 'loss.json'), 'w+') as f:
                        json.dump(dict_mf.loss_, f)
                    with open(join(output_dir, 'diff.json'), 'w+') as f:
                        json.dump(dict_mf.diff_, f)


def launch_from_dir(output_dir):
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    single(output_dir, exp_dict['alpha'], exp_dict['reduction'],
           exp_dict['impute'], exp_dict['dataset'], exp_dict['init'],
           exp_dict['records_range'], exp_dict['random_state'])


def main(dataset='hcp', init='rsn70', n_jobs=1):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    output_dir = expanduser('~/output/fast_spca/%s' % timestamp)

    os.makedirs(output_dir)

    # alphas = np.logspace(-6, -2, 5)
    # reductions = np.linspace(1, 9, 5)
    alphas = [0.01]
    reductions = [1, 3]
    random_states = {1: [0], 3: list(range(2))}
    imputes = [False]

    if dataset == 'hcp':
        records_range = np.arange(400)
    elif dataset == 'adhd':
        records_range = np.arange(36)
    records_range = records_range.tolist()

    i = 0
    experiment_dirs = []
    for alpha in alphas:
        for reduction in reductions:
            for impute in imputes:
                for this_random_state in random_states[reduction]:
                    experiment = {}
                    experiment_dir = join(output_dir, 'experiment_%i' % i)
                    experiment_dirs.append(experiment_dir)
                    os.makedirs(experiment_dir)
                    experiment['alpha'] = alpha
                    experiment['reduction'] = reduction
                    experiment['impute'] = impute
                    experiment['dataset'] = dataset
                    experiment['init'] = init
                    experiment['records_range'] = records_range
                    experiment['random_state'] = this_random_state
                    print(experiment)
                    with open(join(output_dir, 'experiment_%i' % i,
                                   'experiment.json'),
                              'w+') as f:
                        json.dump(experiment, f)
                    i += 1

    Parallel(n_jobs=n_jobs)(delayed(launch_from_dir)(experiment_dir)
                            for experiment_dir in experiment_dirs)


def simple(alpha, reduction, impute, dataset, init, records_range):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    output_dir = expanduser('~/output/fast_spca/%s' % timestamp)

    os.makedirs(output_dir)

    single(output_dir, alpha, reduction, impute, dataset, init, records_range,
           random_state=0)


def analyse_dir(output_dir, dataset='hcp', objective=False, n_jobs=1):
    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    Parallel(n_jobs=n_jobs)(
        # delayed(analyse_exp_var_in_dir)(join(output_dir, experiment_dir),
        delayed(analyse_l1l2_in_dir)(join(output_dir, experiment_dir),
                                     objective=objective,
                                     dataset=dataset)
        for experiment_dir in experiment_dirs)


def analyse_distance(output_dir, dataset='hcp'):
    masker, _, _ = load_data(dataset=dataset)

    masker.set_params(smoothing_fwhm=None, standardize=False)

    dictionaries = {}
    records = {}

    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    min_len = 10000
    for exp in experiment_dirs:
        output_exp = join(output_dir, exp)
        output_files = os.listdir(output_exp)
        dictionaries[exp] = []
        records[exp] = []
        for filename in fnmatch.filter(output_files, 'a_*.nii.gz'):
            dictionaries[exp].append(
                masker.transform(join(output_dir, output_exp,
                                      filename)))
            records[exp].append(int(filename[2:-7]))

        records[exp] = np.array(records[exp])
        order = records[exp].argsort()
        records[exp] = records[exp][order]
        dictionaries[exp] = np.array(dictionaries[exp])
        dictionaries[exp] = dictionaries[exp][order]
        min_len = min(len(dictionaries[exp]), min_len)
    ref_dict = dictionaries.pop('experiment_0')[:min_len]
    dictionaries = [dictionary[:min_len] for dictionary in
                    dictionaries.values()]
    dictionaries = np.array(dictionaries)
    mean_dict = dictionaries.mean(axis=0)
    diff_norm = np.sum((mean_dict - ref_dict) ** 2, axis=(1, 2))
    var_dict = (dictionaries - ref_dict).var(axis=0)
    var_norm = np.sum(var_dict, axis=(1, 2))
    results = dict(diff_norm=diff_norm.tolist(), var_norm=var_norm.tolist(),
                   records=records[0][:len(mean_dict)].tolist())
    json.dump(results, open(join(output_dir, 'diff.json'), 'w+'))


def plot_diff(output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    results = json.load(open(join(output_dir, 'diff.json'), 'r'))
    ax.plot(results['records'], results['diff'])
    plt.savefig(join(output_dir, 'diff.pdf'))


def display_explained_variance_epoch(output_dir, impute=True):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    fig.set_figwidth(3.25653379549)
    fig.set_figheight(0.7 * fig.get_figheight())
    fig.subplots_adjust(bottom=0.09)
    fig.subplots_adjust(top=0.935)
    fig.subplots_adjust(left=0.11)
    fig.subplots_adjust(right=.95)

    stat = []
    alphas = []
    reductions = []
    for filename in glob.glob(join(output_dir, '**/experiment_stat.json'),
                              recursive=True):
        with open(filename, 'r') as f:
            print(filename)
            stat.append(json.load(f))
        alphas.append(stat[-1]['alpha'])
        reductions.append(stat[-1]['reduction'])
    h_reductions = []
    min_len = 10000
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute:
            min_len = min(min_len, len(this_stat['objective']))
    min_len -= 1
    min_len = -1
    ax = {}
    ylim = {1e-2: [2.475e8, 2.522e8], 1e-3: [2.3e8, 2.335e8],
            1e-4: [2.155e8, 2.23e8]}
    for i, alpha in enumerate([1e-3]):
        ax[alpha] = fig.add_subplot(gs[:, i])
        if i == 0:
            ax[alpha].set_ylabel('Objective value on test set')
        # ax[alpha].annotate('Regularization  $\\lambda  = 10^{%.0f}$' % log(alpha, 10), xy=(.9, .9),
        #                    xycoords='axes fraction', ha='right', fontsize=6)
        ax[alpha].set_xlim([.1, 5])

        for tick in ax[alpha].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ax[alpha].set_xscale('log')
        # ax[alpha].set_yscale('log')

        ax[alpha].set_xticks([.2, 1, 2, 3, 4, 5])
        ax[alpha].set_xticklabels(['.2', '1', '2', '3', '4', '5'])

        ax[alpha].set_ylim(ylim[alpha])
        sns.despine(fig=fig, ax=ax[alpha])

        ax[alpha].spines['left'].set_color((.6, .6, .6))
        ax[alpha].spines['bottom'].set_color((.6, .6, .6))
        ax[alpha].xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax[alpha].yaxis.set_tick_params(color=(.6, .6, .6), which='both')
        for tick in ax[alpha].xaxis.get_major_ticks():
            tick.label.set_color('black')
        for tick in ax[alpha].yaxis.get_major_ticks():
            tick.label.set_fontsize(4)

            tick.label.set_color('black')
        t = ax[alpha].yaxis.get_offset_text()
        t.set_size(4)
    ax[1e-3].set_xlabel('Epoch')
    ax[1e-3].xaxis.set_label_coords(-0.05, -0.047)

    colormap = sns.cubehelix_palette(5, start=0, rot=0., hue=1, dark=.3,
                                     light=.7,
                                     reverse=False)

    other_colormap = sns.cubehelix_palette(3, start=0, rot=.5, hue=1, dark=.3,
                                           light=.7,
                                           reverse=False)
    colormap[0] = other_colormap[0]
    x_bar = []
    y_bar_objective = []
    y_bar_density = []
    hue_bar = []
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute \
                and this_stat['alpha'] in [1e-3] and this_stat[
            'reduction'] in [1, 5, 9]:

            print("%s %s" % (this_stat['alpha'], this_stat['reduction']))
            s, = ax[this_stat[
                'alpha']].plot(np.array(this_stat['records']) / (1200 *
                                                                 400),
                               this_stat['objective'],
                               color=colormap[(int(this_stat[
                                                       'reduction']) - 1) // 2],
                               linewidth=1.5,
                               linestyle='--' if this_stat[
                                                     'reduction'] == 1 else '-',
                               zorder=this_stat['reduction'] if this_stat[
                                                                    'reduction'] > 1 else 100)
            x_bar.append(this_stat['alpha'])
            y_bar_objective.append(this_stat['objective'][-1])
            y_bar_density.append(this_stat['densities'][-1])
            hue_bar.append(this_stat['reduction'])
            if this_stat['alpha'] == 1e-3:
                h_reductions.append(
                    (s, '$r = %.0f$' % this_stat['reduction']))

    handles, labels = list(zip(*h_reductions[::-1]))
    argsort = sorted(range(len(labels)), key=labels.__getitem__)
    handles = [handles[i] for i in argsort]
    labels = [labels[i] for i in argsort]
    labels[0] = 'No reduction\n(original alg.)'

    # ax[1e-4].annotate('Original alg.:', xy=(0.67, .84), xycoords='axes fraction',
    #                   horizontalalignment='right', verticalalignment='top', fontsize=6)
    # ax[1e-4].annotate('Proposed red. rati:', xy=(0.67, .64), xycoords='axes fraction',
    #                   horizontalalignment='right', verticalalignment='top', fontsize=6)

    ax[alpha].annotate('$\\lambda  = 10^{%.0f}$' % log(alpha, 10),
                       xy=(0.66, 0.32),
                       ha='left',
                       va='bottom',
                       fontsize=7,
                       xycoords='axes fraction')
    legend_ratio = mlegend.Legend(ax[1e-3], handles[0:], labels[0:],
                                  loc='upper right',
                                  ncol=1,
                                  numpoints=1,
                                  handlelength=2,
                                  # markerscale=1.4,
                                  frameon=False,
                                  bbox_to_anchor=(1, 1)
                                  )
    ax[1e-3].add_artist(legend_ratio)

    fig.savefig(expanduser('~/output/icml/hcp_epoch.pdf'))


def display_explained_variance_density(output_dir, impute=True):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    fig.set_figwidth(fig.get_figwidth() * .73)
    fig.set_figheight(1.0 * fig.get_figheight())
    # ax_pareto = fig.add_subplot(gs[3])
    fig.subplots_adjust(bottom=0.27)
    fig.subplots_adjust(left=0.075)
    fig.subplots_adjust(right=.95)

    stat = []
    alphas = []
    reductions = []
    for filename in glob.glob(join(output_dir, '**/experiment_stat.json'),
                              recursive=True):
        with open(filename, 'r') as f:
            print(filename)
            stat.append(json.load(f))
        alphas.append(stat[-1]['alpha'])
        reductions.append(stat[-1]['reduction'])
    h_reductions = []
    min_len = 10000
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute:
            min_len = min(min_len, len(this_stat['objective']))
    min_len -= 1
    min_len = -1
    ax = {}
    ylim = {1e-2: [2.475e8, 2.522e8], 1e-3: [2.3e8, 2.44e8],
            1e-4: [2.15e8, 2.35e8]}
    for i, alpha in enumerate([1e-2, 1e-3, 1e-4]):
        ax[alpha] = fig.add_subplot(gs[:, i])
        if i == 0:
            ax[alpha].set_ylabel('Objective value on test set')
        ax[alpha].annotate('$\\lambda  = 10^{%.0f}$' % log(alpha, 10),
                           xy=(.65, .85),
                           fontsize=7,
                           xycoords='axes fraction')
        ax[alpha].set_xlim([.1, 200 ])

        # ax[alpha].set_xticklabels(['$10^{-1}$', '$10^{0}$', '$10^{1}$',
        #                            '$10^{2}$'])
        for tick in ax[alpha].xaxis.get_major_ticks():
            tick.label.set_fontsize(5)
        ax[alpha].set_xscale('log')

        ax[alpha].set_xticks([.1, 1, 10, 100])
        ax[alpha].set_xticklabels(['.1 h', '1 h', '10 h', '100 h'])

        ax[alpha].set_ylim(ylim[alpha])
        sns.despine(fig=fig, ax=ax[alpha])

        ax[alpha].spines['left'].set_color((.6, .6, .6))
        ax[alpha].spines['bottom'].set_color((.6, .6, .6))
        ax[alpha].xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax[alpha].yaxis.set_tick_params(color=(.6, .6, .6), which='both')
        for tick in ax[alpha].xaxis.get_major_ticks():
            tick.label.set_color('black')
        for tick in ax[alpha].yaxis.get_major_ticks():
            tick.label.set_fontsize(4)

            tick.label.set_color('black')
        t = ax[alpha].yaxis.get_offset_text()
        t.set_size(4)
        # if alpha == 1e-3:
        #     t.set_transform(ax[alpha].transAxes)
        #     t.set_position((0, 0))
            # ax[alpha].yaxis.stale = True
    ax[1e-4].set_xlabel('CPU\ntime', ha='right')
    ax[1e-4].xaxis.set_label_coords(1.17, -0.05)

    colormap = sns.cubehelix_palette(5, start=0, rot=0., hue=1, dark=.3,
                                     light=.7,
                                     reverse=False)

    other_colormap = sns.cubehelix_palette(3, start=0, rot=.5, hue=1, dark=.3,
                                           light=.7,
                                           reverse=False)
    colormap[0] = other_colormap[0]
    # colormap[0] = [1, .7, .7]
    x_bar = []
    y_bar_objective = []
    y_bar_density = []
    hue_bar = []
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute \
                and this_stat['alpha'] in [1e-2, 1e-3, 1e-4]:

            print("%s %s" % (this_stat['alpha'], this_stat['reduction']))
            s, = ax[this_stat[
                'alpha']].plot(np.array(this_stat['records']) /
                               this_stat['reduction'] / (1200 *
                                                         400) * 158 ,
                               this_stat['objective'],
                               color=colormap[(int(this_stat[
                                                       'reduction']) - 1) // 2],
                               linewidth=2,
                               linestyle='--' if this_stat[
                                                     'reduction'] == 1 else '-',
                               zorder=this_stat['reduction'])
            if this_stat['alpha'] == 1e-4:
                h_reductions.append(
                    (s, '%.0f' % this_stat['reduction']))

    handles, labels = list(zip(*h_reductions[::-1]))
    argsort = sorted(range(len(labels)), key=labels.__getitem__)
    handles = [handles[i] for i in argsort]
    labels = [labels[i] for i in argsort]

    offset = .7
    legend_vanilla = mlegend.Legend(ax[1e-2], handles[:1], ['No reduction'],
                                    loc='lower left',
                                    ncol=5,
                                    numpoints=1,
                                    handlelength=2,
                                    markerscale=1.4,
                                    bbox_to_anchor=(0.3 + offset, -.35),
                                    frameon=False
                                    )

    legend_ratio = mlegend.Legend(ax[1e-2], handles[1:], labels[1:],
                                  loc='lower left',
                                  ncol=5,
                                  markerscale=1.4,
                                  handlelength=2,
                                  bbox_to_anchor=(0.3 + offset, -.5),
                                  frameon=False
                                  )
    # ax[1e-2].annotate('Reduction', xy=(0, -.35), xycoords='axes fraction')
    ax[1e-2].annotate('Original online algorithm', xy=(0.28 + offset, -.27),
                      xycoords='axes fraction',
                      horizontalalignment='right', verticalalignment='bottom',
                      fontsize=7)
    ax[1e-2].annotate('Proposed reduction factor $r$', xy=(0.28 + offset, -.42),
                      xycoords='axes fraction',
                      horizontalalignment='right', verticalalignment='bottom',
                      fontsize=7)
    ax[1e-2].add_artist(legend_ratio)
    ax[1e-2].add_artist(legend_vanilla)

    ax[1e-3].annotate('(a) Convergence speed', xy= (0.5, 1.05), ha='center', va='bottom', xycoords='axes fraction')

    fig.savefig(expanduser('~/output/icml/hcp_bench.pdf'))

    for this_stat in stat:
        x_bar.append(this_stat['alpha'])
        y_bar_objective.append(this_stat['objective'][-1])
        y_bar_density.append(this_stat['densities'][-1])
        hue_bar.append(this_stat['reduction'])
    ref_objective = {}
    ref_density = {}
    for objective, alpha, reduction in zip(y_bar_objective, x_bar, hue_bar):
        if reduction == 1:
            ref_objective[alpha] = objective

    for i, (objective, alpha) in enumerate(zip(y_bar_objective, x_bar)):
        y_bar_objective[i] /= ref_objective[alpha]
        y_bar_objective[i] -= 1

    ####################### Final objective
    fig = plt.figure()
    fig.set_figheight(1.05 * fig.get_figheight())
    # ax_pareto = fig.add_subplot(gs[3])
    fig.subplots_adjust(bottom=0.27)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=1.2)
    fig.set_figwidth(fig.get_figwidth() * 0.27)
    gs = gridspec.GridSpec(2, 1, width_ratios=[1, 1])
    ax_bar_objective = fig.add_subplot(gs[0])
    ax_bar_objective.set_ylim(-0.002, 0.002)
    ax_bar_objective.set_yticks([-.002, -0.001, 0, 0.001, .002])
    ax_bar_objective.set_yticklabels(['$-0.2\%$', '', '$0\%$', '',
                                      '$+0.2\%$'])
    ax_bar_objective.set_ylim(-0.002, 0.002)
    ax_bar_objective.tick_params(axis='y', labelsize=6)

    sns.despine(fig=fig, ax=ax_bar_objective, left=True, right=False)

    sns.barplot(x=x_bar, y=y_bar_objective, hue=hue_bar, ax=ax_bar_objective,
                order=[1e-2, 1e-3, 1e-4],
                palette=colormap)
    plt.setp(ax_bar_objective.patches, linewidth=0.1)
    ax_bar_objective.legend_ = None
    ax_bar_objective.get_xaxis().set_visible(False)
    ax_bar_objective.set_xlim([-.5, 2.6])
    ax_bar_objective.annotate('Final\nobjective\ndeviation\n(relative)',
                              xy=(1.26, 0.45), fontsize=6, va='center',
                              xycoords='axes fraction')
    ax_bar_objective.annotate('(Less is better)', xy=(.06, 0.17), fontsize=5,
                              va='center', xycoords='axes fraction')
    ax_bar_objective.yaxis.set_label_position('right')

    ################################## Density
    stat = []
    x_bar = []
    y_bar_density = []
    hue_bar = []
    for filename in glob.glob(join(output_dir, '**/experiment_l1L2.json'),
                              recursive=True):
        with open(filename, 'r') as f:
            print(filename)
            stat.append(json.load(f))
    for this_stat in stat:
        x_bar.append(this_stat['alpha'])
        y_bar_density.append(this_stat['l1l2s'][-1])
        hue_bar.append(this_stat['reduction'])

    ax_bar_density = fig.add_subplot(gs[1])
    ax_bar_density.set_yscale('log')
    ax_bar_density.set_ylim(100, 1000)
    ax_bar_density.set_yticks([100, 1000])
    ax_bar_density.set_yticklabels(['100', '1000'])
    ax_bar_density.tick_params(axis='y', labelsize=4)

    sns.barplot(x=x_bar, y=y_bar_density, hue=hue_bar, ax=ax_bar_density,
                order=[1e-2, 1e-3, 1e-4],
                palette=colormap)
    ax_bar_density.set_xticklabels(['$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])
    sns.despine(fig=fig, ax=ax_bar_density, left=True, right=False)
    # ax_bar_density.get_xaxis().set_ticks([])
    ax_bar_density.set_xlim([-.5, 2.6])
    ax_bar_density.set_xlabel('Regularization $\\lambda$')
    ax_bar_density.annotate('$\\frac{\\ell_1}{\\ell_2}(\\mathbf D)$',
                            xy=(1.26, 0.45),
                            fontsize=6, va='center', xycoords='axes fraction')
    ax_bar_density.yaxis.set_label_position('right')

    plt.setp(ax_bar_density.patches, linewidth=0.1)
    ax_bar_density.legend_ = None

    for ax in [ax_bar_density, ax_bar_objective]:
        ax.spines['right'].set_color((.6, .6, .6))
        ax.spines['bottom'].set_color((.6, .6, .6))
        ax.xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax.yaxis.set_tick_params(color=(.6, .6, .6), which='both')

    for tic in ax_bar_density.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    ax_bar_objective.spines['bottom'].set_position(('data', 0))
    ax_bar_objective.spines['bottom'].set_linewidth(.3)
    ax_bar_objective.annotate('(b) Decomposition quality', xy= (0.5, 1.05), ha='center', va='bottom', xycoords='axes fraction')

    fig.savefig(expanduser('~/output/icml/bar_plot.pdf'))


def retrieve_last(output_dir, n_jobs=1):
    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    if not os.path.exists(join(output_dir, 'pdf')):
        os.makedirs(join(output_dir, 'pdf'))
    Parallel(n_jobs=n_jobs)(delayed(single_retrieve)(experiment_dir,
                                                     output_dir) for
                            experiment_dir in experiment_dirs)

def retrieve_csv(output_dir):
    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    data = []
    for experiment_dir in experiment_dirs:
        with open(join(output_dir, experiment_dir, 'experiment.json'), 'r') as f:
                exp_dict = json.load(f)
        data.append([experiment_dir, exp_dict['reduction'], float(exp_dict['alpha'])])
    df = pd.DataFrame(data)
    df.to_csv(expanduser('~/exp.csv'))


def single_retrieve(experiment_dir, output_dir):
    with open(join(output_dir, experiment_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    output_files = os.listdir(join(output_dir, experiment_dir))
    filenames = sorted(fnmatch.filter(output_files, 'a_*.nii.gz'),
                       key=lambda t: int(t[2:-7]))
    last_filename = filenames[-1]
    name = "%s_%.4e.pdf" % (exp_dict['reduction'], float(exp_dict['alpha']))
    name = name.replace('.', '_')
    name = name.replace('-', '_')
    print( join(output_dir, 'pdf', name))
    print(join(output_dir, experiment_dir, last_filename))
    plot_to_pdf(join(output_dir, experiment_dir, last_filename), join(output_dir, 'pdf', name))


if __name__ == '__main__':
    # main('adhd', 'rsn20', n_jobs=3)
    # analyse_distance('/home/arthur/output/fast_spca/2016-01-28_18-06-42', 'adhd')
    # plot_diff('/media/storage/output/fast_spca/2016-01-28_17-16-23')
    # simple(1e-4, 3, True, 'adhd', 'rsn20', list(range(0, 36)))
    # analyse_dir(
    #     '/storage/workspace/amensch/output/fast_spca/2016-01-26_15-31-43',
    #     n_jobs=15, objective=True)
    display_explained_variance_density(
        expanduser('~/json_hcp'),
        impute=False)
    display_explained_variance_epoch(
        expanduser('~/json_hcp'),
        impute=False)
    # retrieve_last(
    #     '/storage/workspace/amensch/output/fast_spca/2016-01-26_15-31-43',
    #     n_jobs=15)
    # retrieve_csv(expanduser('~/drago/output/fast_spca/2016-01-26_15-31-43'))
    # display_explained_variance_density('/home/parietal/amensch/output/fast_spca/2016-01-25_23-56-39')
    # display_explained_variance_density(
    #     expanduser('~/drago/output/fast_spca/2016-01-25_23-56-39'),
    #     impute=False)
    # display_explained_variance_density(
    #     expanduser('/home/arthur/drago/output/fast_spca/2016-01-25_22-21-50'),
    #     impute=False)