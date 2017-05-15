from os.path import expanduser
import matplotlib

from nilearn.datasets import fetch_adhd
from nilearn.image import index_img

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, plot_stat_map

for j, i in enumerate([16, 64, 256]):
    img = expanduser('~/data/modl_data/pipeline/components/hcp/%i/0.0001/components.nii.gz') % i
    plot_prob_atlas(img, view_type='filled_contours', linewidths=1, colorbar=False, draw_cross=False, annotate=False,
                    cut_coords=1,
                    display_mode='x')
    plt.savefig('components_%i_x.png' % i)
    plot_prob_atlas(img, view_type='filled_contours', linewidths=1, colorbar=False, draw_cross=False, annotate=False,
                    cut_coords=1,
                    display_mode='z')
    plt.savefig('components_%i_z.png' % i)
# for i in [16, 64, 256]:
#     img = expanduser('~/data/modl_data/pipeline/components/hcp/%i/0.0001/components.nii.gz') % i
#     plot_prob_atlas(img, view_type='contours', colorbar=False,
#                     cut_coords=1,
#                     draw_cross=False, annotate=False,
#                     display_mode='x')
#     plt.savefig('components_%i.png' % i)
#
# adhd = fetch_adhd()
# img = adhd.func[1]
#
# for i in [0, 10, 20]:
#     plot_stat_map(index_img(img, i), cut_coords=1,
#                   draw_cross=False, annotate=False,
#                   display_mode='z', colorbar=False)
#     plt.savefig('rest_%i.png' % i)
