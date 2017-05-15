import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from os.path import expanduser
import seaborn.apionly as sns
import numpy as np

res = pd.read_csv(expanduser('~/nips/performance.csv'), index_col=0,
                  header=0)
flatui = get_cmap('Vega20').colors[1::2]
x = []
width = 0.7
center = {'Archi': 1, 'Brainomics': 4, 'CamCan': 7, 'UCLA': 10}
offset = {'Dictionary projection': -width * 2,
          'Multi-scale dictionary': - width,
          'Latent factors + dropout': 0,
          'Transfer from HCP': + width}
colors = {'Dictionary projection': flatui[0],
          'Multi-scale dictionary': flatui[1],
          'Latent factors + dropout': flatui[2],
          'Transfer from HCP': flatui[3]}

method_label = {'Dictionary projection': 'Dictionary projection',
                'Multi-scale dictionary': 'Multi-scale dictionary',
                'Latent factors + dropout': 'Latent cognitive (single-dataset)',
                'Transfer from HCP': 'Latent cognitive (multi-dataset)'}

fig, ax = plt.subplots(1, 1, figsize=(5.5015, 1.2))
fig.subplots_adjust(bottom=.15, left=.08, right=0.98, top=0.965)
for method in ['Dictionary projection', 'Multi-scale dictionary',
               'Latent factors + dropout',
               'Transfer from HCP']:
    sub_res = res.query("Method == '%s'" % method)
    y = sub_res['mean']
    std = sub_res['std']
    datasets = sub_res['Dataset']
    x = []
    for dataset in datasets:
        x.append(center[dataset] + offset[method])
    ax.bar(x, y, width=width, color=colors[method], yerr=std,
           label=method_label[method])
    for this_x, this_y in zip(x, y):
        ax.annotate("%.1f" % (this_y * 100), xy=(this_x, this_y),
                    xytext=(0, -8),
                    textcoords='offset points', ha='center', va='center',
                    xycoords='data')
ax.set_xticks(np.array(list(center.values())) - width / 2)
labels = list(center.keys())
labels[-1] = 'LA5c'
ax.set_xticklabels(labels)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0.44, 0.91)
ax.set_xlim(-1, 11)
ax.legend(loc='center left', ncol=1, bbox_to_anchor=(.6, 0.74),
          columnspacing=0,
          frameon=False)
sns.despine(ax=ax)

plt.savefig(expanduser('~/nips/ablation.pdf'))
