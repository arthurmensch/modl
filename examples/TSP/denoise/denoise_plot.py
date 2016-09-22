import json
import pandas as pd
from os.path import join, expanduser
import matplotlib.pyplot as plt
import seaborn.apionly as sns

output_dir = expanduser('~/output/modl/denoise')

exps = []
offset = 500
for i in range(340, 343):
    working_dir = join(output_dir, 'experiment_%i' % i)
    exp = json.load(open(join(working_dir, 'experiment.json'), 'r'))
    res = json.load(open(join(working_dir, 'callback.json'), 'r'))
    exps.append(dict(**exp, **res))
df = pd.DataFrame(exps)
n_exp = len(df)

# fig, ax = plt.subplots(2, 2, figsize=(10, 6))
#
# labels = ['Full', 'Averaging', 'Masked (ICML)', 'Full Gram', 'Dict only',
#           'Averaging (sync param)', 'Masked (ICML) (sync param)', 'Full Gram (sync param)']
# for i in range(0, n_exp):
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
#
# for i in range(2):
#     ax[i, 0].set_ylim([0, 500])
# ax[0, 0].set_xlim([1e0, 5e2])
# ax[1, 0].set_xlim([1e4, 3e6])
# ax[1, 0].set_xlim(1e3, 1e5)
# ax[1, 0].set_ylim([0, 250])

# ax[0, 1].legend()

fig, ax = plt.subplots(1, 1, figsize=(4, 2), squeeze=False)
fig.subplots_adjust(left=0.15)

labels = ['Full', 'Averaging', 'Masked (ICML)']

for i in range(n_exp):
    ax[0, 0].plot(df.ix[i, 'time'], df.ix[i, 'train_obj'][:-1], label=labels[i], marker='o', markersize=2)
    # ax[0, j].plot(df.ix[i, 'time'], df.ix[i, 'test_obj'][:-1], '--', label=labels[i], marker='o', markersize=4 if j == 1 else 3)
    ax[0, 0].set_xscale('log')
    # ax[0, j].set_yscale('log')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Train loss')
    ax[0, 0].xaxis.set_label_coords(1.09, 0.02)
    ax[0, 0].annotate("8x8 patches, 1M image", xy=(.5, 1), xycoords='axes fraction')
    sns.despine(fig, ax[0])
# ax[0, 0].set_ylim([140, 180])
# ax[0, 0].set_xlim([15, 10000])
# ax[0, 0].legend()
plt.show()
# plt.savefig('reduced_.pdf')