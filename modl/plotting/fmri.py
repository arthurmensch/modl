from nilearn._utils import check_niimg
from nilearn.image import index_img
from nilearn.plotting import plot_prob_atlas, plot_stat_map


def display_maps(fig, components, index=0):
    components = check_niimg(components)
    ax = fig.add_subplot(2, 1, 1)
    plot_prob_atlas(components, view_type="filled_contours",
                    axes=ax)
    ax = fig.add_subplot(2, 1, 2)
    plot_stat_map(index_img(components, index), axes=ax, colorbar=False,
                  threshold=0)
    return fig
