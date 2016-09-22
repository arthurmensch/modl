import json
import pandas as pd
from os.path import join, expanduser
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
output_dir = expanduser('~/drago_output/modl/denoise')

exps = []
for i in range(3300, 3303):
# for i in [400, 405, 406]:
    working_dir = join(output_dir, 'experiment_%i' % i)
    exp = json.load(open(join(working_dir, 'experiment.json'), 'r'))
    res = json.load(open(join(working_dir, 'callback.json'), 'r'))
    exps.append(dict(**exp, **res))
df = pd.DataFrame(exps)
n_exp = len(df)

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

labels = ['Mairal \'10', 'Proposed ($r=4$)', 'Mensch \'16 ($r = 4$)', 'Full Gram', 'Dict only',
          'Averaging (sync param)', 'Masked (ICML) (sync param)', 'Full Gram (sync param)']
# for i in range(n_exp):
#     for j in range(2):
#         ax[0, j].plot(df.ix[i, 'time'], df.ix[i, 'train_obj'][:-1], label=labels[i], marker='o', markersize=4 if j == 1 else 3)
#         # ax[0, j].plot(df.ix[i, 'time'], df.ix[i, 'test_obj'][:-1], '--', label=labels[i], marker='o', markersize=4 if j == 1 else 3)
#         ax[0, j].set_xscale('log')
#         # ax[0, j].set_yscale('log')
#         ax[0, j].set_xlabel('Time')
#         sns.despine(fig, ax[j])
#     for j in range(2):
#         ax[1, j].plot(df.ix[i, 'iter'], df.ix[i, 'train_obj'][:-1], label=labels[i], marker='o', markersize=4 if j == 1 else 3)
#         # ax[1, j].plot(df.ix[i, 'iter'], df.ix[i, 'test_obj'][:-1], '--', label=labels[i], marker='o', markersize=4 if j == 1 else 3)
#         ax[1, j].set_xscale('log')
#         ax[1, j].set_xlabel('Iter')
#         sns.despine(fig, ax[j])


# ax[0, 0].set_ylabel('Train loss')
# ax[1, 0].set_ylabel('Train loss')

# for i in range(2):
#     ax[i, 0].set_ylim([50, 105])
# ax[0, 0].set_xlim([1e0, 5e2])
# ax[1, 0].set_xlim([1e4, 3e6])
# ax[1, 0].set_xlim(1e3, 1e5)
# ax[1, 0].set_ylim([0, 250])

# ax[0, 1].legend()

fig, ax = plt.subplots(1, 1, figsize=(4, 1.8), squeeze=False)
fig.subplots_adjust(left=0.17, right=0.85)

range_extr = 2
extrap = df.ix[1, 'train_obj'][-2] + (range_extr + 10) * (df.ix[1, 'train_obj'][-2] - df.ix[1, 'train_obj'][-2 - range_extr + 1]) / range_extr
for i in range(n_exp):
    ax[0, 0].plot(df.ix[i, 'time'], (np.array(df.ix[i, 'train_obj'][:-1]) - extrap) / extrap, label=labels[i], marker='o', markersize=2)
    # ax[0, j].plot(df.ix[i, 'time'], df.ix[i, 'test_obj'][:-1], '--', label=labels[i], marker='o', markersize=4 if j == 1 else 3)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    # ax[0, j].set_yscale('log')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Train loss (rel. to lowest value)')
    ax[0, 0].xaxis.set_label_coords(1.09, 0.02)
    # ax[0, 0].annotate("64x64 patches, 20M image", xy=(.2, 1), xycoords='axes fraction')
    sns.despine(fig, ax[0])
ax[0, 0].set_ylim([2e-3, 1.1e0])
ax[0, 0].set_yticks([2e-3, 1e-2, 1e-1, 1e0])
ax[0, 0].set_yticklabels(['$0.2\\%$', '$1\\%$', '$10\\%$', '$100\\%$'])
ax[0, 0].set_xticks([1, 10, 100, 500])
ax[0, 0].set_xticklabels(['1 s', '10 s', '100 s', '500 s'])
ax[0, 0].set_xlim([1e0, 6e2])
ax[0, 0].legend(bbox_to_anchor=(0.55, 1.1), loc='upper left')

plt.savefig('opt.pdf')