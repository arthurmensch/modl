import os
from os.path import join
import json

import pandas as pd
import numpy as np

import sys

import matplotlib as mpl
from matplotlib import rc_params_from_file

from modl.utils.system import get_output_dir

dir_name = os.path.dirname(os.path.realpath(sys.argv[0]))
rc_params = rc_params_from_file(join(dir_name, "matplotlibrc"))
mpl.RcParams(rc_params)
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['axes.formatter.useoffset'] = False

import matplotlib.pyplot as plt
import seaborn.apionly as sns

dataset_exps = json.load(open(join(dir_name, 'compare_reductions.json')))

reductions = [1, 4, 6, 8, 12, 24]
n_red = len(reductions)

fig, axes = plt.subplots(1, 4, figsize=(7.141, 1.4))
fig.subplots_adjust(right=0.99, left=0.07, bottom=0.25, top=0.96,
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
        # if algorithm == 'hcp':
        #     score -= 1e5
        time = np.array(exp['info']['profiling'])[:, 5]
        time += 3.6 if algorithm == 'adhd' else 36
        reduction = exp['config']['reduction']
        color = color_dict[reduction]
        axes[i].plot(time, score,
                     label="$r = %i$" % reduction if reduction != 1 else "\\textbf{OMF}",
                     zorder=reduction if reduction != 1 else 1,
                     color=color,
                     markersize=2)
    axes[i].set_xscale('log')
    axes[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2),
                             useOffset=False)

    if algorithm == 'hcp':
        axes[i].annotate('$\\times 10^5$', xy=(0, 1), xytext=(-15, 3),
                         va='top', ha='left',
                         xycoords='axes fraction',
                         fontsize=6,
                         textcoords='offset points')
        plt.setp(axes[i].yaxis.get_offset_text(), visible=False)
    elif algorithm == 'adhd':
        axes[i].annotate('$\\times 10^4$', xy=(0, 1), xytext=(-15, 3),
                         va='top', ha='left',
                         xycoords='axes fraction',
                         fontsize=6,
                         textcoords='offset points')
        plt.setp(axes[i].yaxis.get_offset_text(), visible=False)

    sns.despine(fig, axes)
    if algorithm != 'adhd':
        xticks = np.array([15, 100., 3600, 3600 * 5, 3600 * 24])
        xticks -= 36.
        axes[i].set_xticklabels(['15s', '100s', '1h', '5h', '24h'])
    else:
        xticks = np.array([15, 100., 360])
        xticks -= 3.6
        axes[i].set_xticklabels(['15s', '100s', '6min'])
    axes[i].set_xticks(xticks)

axes[0].set_ylabel('Test objective value')
axes[0].yaxis.set_label_coords(-0.25, 0.4)

axes[0].annotate('Time', xy=(0, 0), xytext=(-4, -7),
                 va='top', ha='right',
                 xycoords='axes fraction',
                 textcoords='offset points')

axes[0].set_xlim([1.5e-3 * 3600, 2e-1 * 3600])
axes[0].set_ylim([21800, 27000])
axes[1].set_xlim([0.015 * 3600, 10 * 3600])
axes[1].set_ylim([3.2, 4])
axes[2].set_xlim([0.015 * 3600, 10 * 3600])
axes[2].set_ylim([34, 45])
axes[3].set_xlim([0.017 * 3600, 24 * 3600])
axes[3].set_ylim([-3200 + 1e5, 5000 + 1e5])
# axes[3].annotate('$+ 10^5$', xy=(0.04, 1), xycoords='axes fraction', va='top',
#                  ha='left', fontsize=6)
axes[0].annotate('ADHD\nSparse dictionary',
                 # '$p = 6\\cdot 10^4\\ \\: n = 6000$',
                 xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
axes[1].annotate('Aviris\nNMF',
                 # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                 xy=(0.6, 1), xycoords='axes fraction', va='top', ha='center')
axes[2].annotate('Aviris\nDictionary learning',
                 # '$p = 6\\cdot 10^4\\ \\: n = 1\\cdot 10^5$',
                 xy=(0.55, 1), xycoords='axes fraction', va='top', ha='center')
axes[3].annotate('HCP\nSparse dictionary',
                 # 'p = 2\\cdot 10^5\\ \\: n = 2\\cdot 10^6$',
                 xy=(0.65, 1), xycoords='axes fraction', va='top', ha='center')

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

######## SGD
run_id = 12
dir = join(get_output_dir(), 'decompose_rest_multi', str(run_id), 'run')
analysis_dir = join(get_output_dir(), 'decompose_rest_multi', str(run_id),
                    'analysis')
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

data = []
for i in range(22):
    this_dir = join(dir, str(i))
    try:
        config = json.load(open(join(this_dir, 'config.json'), 'r'))
        info = json.load(open(join(this_dir, 'info.json'), 'r'))
    except FileNotFoundError:
        print('Skipping %i' % i)
    optimizer = config['optimizer']
    step_size = config['step_size']
    reduction = config['reduction']
    score = info['score']
    time = info['time']
    data.append({'step_size': step_size,
                 'optimizer': optimizer,
                 'reduction': reduction,
                 'score': score,
                 'time': time})
data = pd.DataFrame(data)
data.set_index(['optimizer', 'reduction', 'step_size'], inplace=True)
data.to_csv(join(analysis_dir, 'data.csv'))
data.sort_index(inplace=True)
data.to_csv(join(analysis_dir, 'data.csv'))

step_size = data.loc['sgd'].index[1][1]

idx = ('sgd', 12, step_size)
this_data = data.loc[idx]
data.reset_index(inplace=True)

# Plot
ax = axes[3]

sgd_colormap = sns.cubehelix_palette(6, start=1, rot=0.2,
                                     light=0.7,
                                     reverse=False)
color = sgd_colormap[0]
axes[3].plot(this_data['time'], this_data['score'],
             label='SGD (best step-size)',
             color=color,
             linestyle='-')

handles, labels = axes[0].get_legend_handles_labels()
# Add 6
handles_2, labels_2 = axes[2].get_legend_handles_labels()
handles = handles[:2] + handles_2[1:2] + handles[2:]
labels = labels[:2] + labels_2[1:2] + labels[2:]
handles_3, labels_3 = axes[3].get_legend_handles_labels()
handles_3, labels_3 = handles_3[-1:], labels_3[-1:]
#
# handles += handles_2[-1:]
# labels += labels_2[-1:]

axes[0].annotate('\\textbf{OMF:}', xy=(-0.3, -0.25),
                 xycoords='axes fraction',
                 va='top')

axes[0].annotate('\\textbf{SOMF:}', xy=(0.55, -0.25),
                 xycoords='axes fraction',
                 va='top')

l2 = axes[0].legend(handles[:1],
                    ['$r = 1$'],
                    bbox_to_anchor=(-0.1, -0.15),
                    markerfirst=False,
                    loc='upper left',
                    ncol=1,
                    frameon=False)  # this removes l1 from the axes.

l1 = axes[0].legend(handles[1:n_red],
                    labels[1:n_red],
                    bbox_to_anchor=(0.9, -0.15),
                    markerfirst=False,
                    loc='upper left',
                    ncol=8,
                    frameon=False)
l3 = axes[3].legend(handles_3,
                    ['Best step-size \\textbf{SGD}'],
                    bbox_to_anchor=(0, -0.15),
                    markerfirst=False,
                    loc='upper left',
                    ncol=8,
                    frameon=False
                    )

axes[0].add_artist(l1)
axes[0].add_artist(l2)

plt.savefig(join(analysis_dir, 'compare_reduction.pdf'))
plt.show()
