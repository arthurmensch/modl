from math import sqrt
from os.path import expanduser, join

import matplotlib
from matplotlib.cm import get_cmap

from nilearn._utils import check_niimg
import matplotlib.pyplot as plt
from nilearn.image import index_img, new_img_like
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords

import numpy as np

nii_dir = expanduser('~/data/modl_data/pipeline/contrast/final/compare')
img_no_latent = check_niimg(join(nii_dir, 'no_latent.nii.gz'))
img_latent_single = check_niimg(join(nii_dir, 'latent_single.nii.gz'))
img_latent_multi = check_niimg(join(nii_dir, 'latent_multi.nii.gz'))
imgs = [img_no_latent, img_latent_single, img_latent_multi]
for img in imgs:
    img.get_data()

conditions = {# 'triangle_intention': 28,
              # 'face_trusty': 5,
              'calculaudio': 7,
              'face_control': 3}
conditions_name = {'triangle_intention': 'Moving triangles',
                   'calculaudio': 'Audio\ncalculation',
                   'face_control': 'Face'}
flatui = get_cmap('Vega20').colors[1::2]

exps = ['Multi-scale\n'
        'spatial projection', 'Latent cognitive\nspace (single)',
        'Latent cognitive\n(multi-study)']
for i, condition in enumerate(conditions):
    fig, axes = plt.subplots(1, 3, figsize=(2.55, 0.8))
    fig.subplots_adjust(top=0.99, left=0.14, right=.99, bottom=0.22, hspace=0,
                        wspace=0.1)
    idx = conditions[condition]
    if condition in ['computation', 'calculaudio', 'calculvideo', 'MATH']:
        cut_coords = [46]
    elif condition in ['OBK_FACE', '2BK_FACE', 'face_control',
                       'face_trusty', 'face_sex', 'FACES']:
        cut_coords = [-10]  # FFA
    elif condition in ['triangle_random', 'triangle_intention']:
        cut_coords = [-2]
    else:
        cut_coords = find_xyz_cut_coords(img_latent_multi,
                                         activation_threshold=vmax / 3)
    for j, img in enumerate(imgs):
        this_img = index_img(imgs[j], idx)
        data = this_img.get_data()
        vmax = np.max(np.abs(data))
        data /= vmax
        this_img = new_img_like(this_img, data=data)
        vmax = np.max(data)
        plot_stat_map(this_img, axes=axes[j], figure=fig, display_mode='z',
                      cmap=get_cmap('seismic'),
                      threshold=0.0 * vmax,
                      vmax=vmax * 0.7 if i == 0 and j >= 0 else vmax,
                      cut_coords=cut_coords, colorbar=False, annotate=False, draw_cross=False)
        axes[j].annotate(exps[j], xy=(.5, .02),
                         fontsize=7,
                         # bbox=bbox_props,
                         xycoords='axes fraction', ha='center', va='top')
    axes[0].annotate('%s\nz=%i' % (conditions_name[condition],
                                        cut_coords[0]),
                        fontsize=7,
                        xy=(-.51, .9),
                        xycoords='axes fraction', ha='left', va='top',
                        )
    plt.savefig(expanduser('~/nips/classification_%s.pdf' % condition))
