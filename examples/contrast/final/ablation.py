import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from os.path import expanduser
import seaborn.apionly as sns
import numpy as np

res = pd.read_csv(expanduser('~/nips/performance.csv'), index_col=0,
                  header=0)
# res = res.query("Method in ['Transfer from HCP', 'Latent factors + dropout']")
res_ext = pd.read_csv(expanduser('~/nips/performance_simple.csv'),
                      header=0)
res = pd.concat([res, res_ext], axis=0)
res = res.reset_index()
idx = res.groupby(by=['Dataset', 'Method']).aggregate('idxmax')['mean'].values
res = res.loc[idx]
flatui = get_cmap('Vega20').colors[1::2]
x = []
width = 0.75
center = {'Archi': 1, 'Brainomics': 5, 'CamCan': 9, 'UCLA': 13}
offset = {'Simple multinomial': - 2.5 * width,
          'Dictionary projection': -1.5 * width,
          'Multi-scale dictionary': - .5 * width,
          'Latent factors + dropout': width / 2,
          'Transfer from HCP': + 1.5 * width}
colors = {'Simple multinomial': flatui[0],
          'Dictionary projection': flatui[1],
          'Multi-scale dictionary': flatui[2],
          'Latent factors + dropout': flatui[3],
          'Transfer from HCP': flatui[4]}

method_label = {'Simple multinomial': 'Full multinomial',
    'Dictionary projection': 'Spatial projection',
                'Multi-scale dictionary': 'Multi-scale spatial projection',
                'Latent factors + dropout': 'Latent cognitive space (single study)',
                'Transfer from HCP': 'Latent cognitive space (multi-study)'}

fig, ax = plt.subplots(1, 1, figsize=(5.5015, 1.3))
fig.subplots_adjust(bottom=.15, left=.08, right=0.98, top=.98)
for method in ['Simple multinomial', 'Dictionary projection', 'Multi-scale dictionary',
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
ax.set_xlim(-1.25, 14.5)
h, l = ax.get_legend_handles_labels()
h_1 = [h[0]] + h[2:]
l_1 = [l[0]] + l[2:]
legend_1 = ax.legend(h_1, l_1, loc='center left', ncol=1, bbox_to_anchor=(.49, 0.74),
          columnspacing=0,
          frameon=False)
ax.legend([h[1]], [l[1]], loc='center left', ncol=1, bbox_to_anchor=(.75, 0.97),
          columnspacing=0,
          frameon=False)
ax.add_artist(legend_1)
sns.despine(ax=ax)

plt.savefig(expanduser('~/nips/ablation.pdf'))
