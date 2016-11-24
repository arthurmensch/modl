# import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from modl.plotting.images import plot_single_patch

ref = np.load('components_1_49316s_841000p.npy')

reduced = np.load('components_24_179s_87400p.npy')

full = np.load('components_1_177s_3000p.npy')

ref = ref.reshape((-1,  16, 16, 223))
reduced = reduced.reshape((-1,  16, 16, 223))
full = full.reshape((-1,  16, 16, 223))

fig, axes = plt.subplots(3, 3, figsize=(252 / 72.25, 80 / 72.25))
fig.subplots_adjust(right=0.79, left=0.11, bottom=0.00, top=0.9, wspace=0.1, hspace=0.1)
names = ["""$r = 1$""",
         """$r = 24$""",
         """$r = 1$"""]
times = ['\\textbf{14h}', '\\textbf{179 s}', '\\textbf{177 s}']
patches = ["""841k patches""", """87k patches""", """3k patches"""]

order = [0, 2, 1]

for j, idx in enumerate([18, 49, 90]):
    for i, comp in enumerate([ref, full, reduced]):
        axes[i, j] = plot_single_patch(axes[i, j], comp[idx], 1, 3)
    axes[0, j].set_xlabel('Comp. %i' % (j + 1))
    axes[0, j].xaxis.set_label_coords(0.5, 1.4)
for i in range(3):
    axes[i, 2].annotate('Time: %s' % times[order[i]],
                        xycoords='axes fraction', xy=(1.1, 0.6), va='bottom', ha='left')
    axes[i, 2].annotate('%s' % patches[order[i]],
                        xycoords='axes fraction', xy=(1.1, 0.0), va='bottom', ha='left')
axes[1, 0].annotate("""\\textbf{\\textsc{OMF}}""",
                    xycoords='axes fraction', xy=(-0.5, 1.3), va='bottom', ha='left')
axes[1, 0].annotate("""$r = 1$""", xycoords='axes fraction', xy=(-0.5, 0.7), va='bottom', ha='left')
axes[2, 0].annotate("""\\textbf{\\textsc{SOMF}}""", xycoords='axes fraction', xy=(-0.5, 0.6), va='bottom', ha='left')
axes[2, 0].annotate("""$r = 24$""", xycoords='axes fraction', xy=(-0.5, 0.0), va='bottom', ha='left')
trans = blended_transform_factory(fig.transFigure, axes[1, 0].transAxes)
line = Line2D([0, 1], [-0.2, -0.2], color='black', linestyle='--', linewidth=1, transform=trans)
fig.lines.append(line)
plt.savefig('patches.pdf')
# plt.show()