from matplotlib import pyplot as plt
from nilearn.image import index_img
from nilearn.plotting import plot_prob_atlas, plot_stat_map


def display_maps(components, index=0):
    fig, axes = plt.subplots(2, 1)
    plot_prob_atlas(components, view_type="filled_contours",
                    axes=axes[0])
    plot_stat_map(index_img(components, index),
                  axes=axes[1],
                  colorbar=False,
                  threshold=0)
    return fig
