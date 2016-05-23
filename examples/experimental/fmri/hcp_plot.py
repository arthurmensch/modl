import json
import os
from math import log
from os.path import expanduser, join

import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from matplotlib import gridspec


def display_explained_variance_density(output_dir):
    dir_list = [join(output_dir, f) for f in os.listdir(output_dir) if
                os.path.isdir(join(output_dir, f))]

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    fig.set_figwidth(fig.get_figwidth() * .73)
    fig.set_figheight(1.03 * fig.get_figheight())
    fig.subplots_adjust(bottom=0.27)
    fig.subplots_adjust(left=0.075)
    fig.subplots_adjust(right=.95)

    results = []
    analyses = []
    ref_time = 1000000
    for dir_name in dir_list:
        try:
            analyses.append(json.load(open(join(dir_name, 'analysis.json'), 'r')))
            results.append(json.load(open(join(dir_name, 'results.json'), 'r')))
            if results[-1]['reduction'] == 12:
                timings = np.array(results[-1]['timings'])
                diff = timings[1:] - timings[:1]
                ref_time = min(ref_time, np.min(diff))
        except IOError:
            pass
    h_reductions = []
    ax = {}
    ylim = {1e-2: [2.455e8, 2.525e8], 1e-3: [2.3e8, 2.47e8],
            1e-4: [2.16e8, 2.42e8]}
    for i, alpha in enumerate([1e-2, 1e-3, 1e-4]):
        ax[alpha] = fig.add_subplot(gs[:, i])
        if i == 0:
            ax[alpha].set_ylabel('Objective value on test set')
        ax[alpha].annotate('$\\lambda  = 10^{%.0f}$' % log(alpha, 10),
                           xy=(.65, .85),
                           fontsize=7,
                           xycoords='axes fraction')
        ax[alpha].set_xlim([.05, 200])
        ax[alpha].set_ylim(ylim[alpha])

        for tick in ax[alpha].xaxis.get_major_ticks():
            tick.label.set_fontsize(5)
        ax[alpha].set_xscale('log')

        ax[alpha].set_xticks([.1, 1, 10, 100])
        ax[alpha].set_xticklabels(['.1h', '1h', '10h', '100h'])

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
    ax[1e-4].set_xlabel('CPU\ntime', ha='right')
    ax[1e-4].xaxis.set_label_coords(1.17, -0.05)

    colormap = sns.cubehelix_palette(5, start=0, rot=0., hue=1, dark=.3,
                                     light=.7,
                                     reverse=False)
    other_colormap = sns.cubehelix_palette(3, start=0, rot=.5, hue=1, dark=.3,
                                           light=.7,
                                           reverse=False)
    colormap[0] = other_colormap[0]
    colormap_dict = {reduction: color for reduction, color in
                     zip([1, 2, 4, 8, 12],
                         colormap)}

    x_bar = []
    y_bar_objective = []
    y_bar_density = []
    hue_bar = []

    for result, analysis in zip(results, analyses):
        if True : #int(result['reduction']) != 3:
            print("%s %s" % (result['alpha'], result['reduction']))
            timings = (np.array(analysis['records']) + 1) / int(result['reduction']) * 12 * ref_time / 3600
            # timings = np.array(result['timings'])[np.array(analysis['records']) + 1] / 3600
            s, = ax[result[
                'alpha']].plot(
                timings,
                np.array(analysis['objectives']) / 4,
                color=colormap_dict[int(result['reduction'])],
                linewidth=2,
                linestyle='--' if result[
                                      'reduction'] == 1 else '-',
                zorder=result['reduction'] if result['reduction'] !=  1 else 100)
            if result['alpha'] == 1e-2:
                h_reductions.append(
                    (s, '%.0f' % result['reduction']))

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
    ax[1e-2].annotate('Original online algorithm', xy=(0.28 + offset, -.27),
                      xycoords='axes fraction',
                      horizontalalignment='right', verticalalignment='bottom',
                      fontsize=7)
    ax[1e-2].annotate('Proposed reduction factor $r$',
                      xy=(0.28 + offset, -.42),
                      xycoords='axes fraction',
                      horizontalalignment='right', verticalalignment='bottom',
                      fontsize=7)
    ax[1e-2].add_artist(legend_ratio)
    ax[1e-2].add_artist(legend_vanilla)

    ax[1e-3].annotate('(a) Convergence speed', xy=(0.5, 1.05), ha='center',
                      va='bottom', xycoords='axes fraction')

    fig.savefig(join(output_dir, 'hcp_bench.pdf'))

    for result, analysis in zip(results, analyses):
        x_bar.append(result['alpha'])
        y_bar_objective.append(analysis['objectives'][-1])
        y_bar_density.append(analysis['densities'][-1])
        hue_bar.append(result['reduction'])
    ref_objective = {}
    for objective, alpha, reduction in zip(y_bar_objective, x_bar, hue_bar):
        if reduction == 1:
            ref_objective[alpha] = objective

    for i, (objective, alpha) in enumerate(zip(y_bar_objective, x_bar)):
        y_bar_objective[i] /= ref_objective[alpha]
        y_bar_objective[i] -= 1

    ####################### Final objective
    fig = plt.figure()
    fig.set_figheight(1.05 * fig.get_figheight())
    fig.subplots_adjust(bottom=0.27)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=1.2)
    fig.set_figwidth(fig.get_figwidth() * 0.27)
    gs = gridspec.GridSpec(2, 1, width_ratios=[1, 1])
    ax_bar_objective = fig.add_subplot(gs[0])
    ax_bar_objective.set_ylim(-0.01, 0.01)
    ax_bar_objective.set_yticks([-.01, -0.005, 0, 0.005, .01])
    ax_bar_objective.set_yticklabels(['$-1\%$', '', '$0\%$', '',
                                      '$+1\%$'])
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
    x_bar = []
    y_bar_density = []
    hue_bar = []
    for result, analysis in zip(results, analyses):
        x_bar.append(result['alpha'])
        y_bar_density.append(analysis['densities'][-1])
        hue_bar.append(result['reduction'])

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
    ax_bar_objective.annotate('(b) Decomposition quality', xy=(0.5, 1.1),
                              ha='center', va='bottom',
                              xycoords='axes fraction')

    fig.savefig(expanduser(join(output_dir, 'bar_plot.pdf')))


if __name__ == '__main__':
    output_dir = expanduser('~/output/modl/hcp_no_replacement_reduction')
    display_explained_variance_density(output_dir)
