import numpy as np
import matplotlib.cm as cm


def plot_patches(fig, patches):
    if patches.ndim == 4:
        channel_step = patches.shape[3] // 3
        patches = np.concatenate([np.sum(patches[:, :, :, i * channel_step:
        (i + 1) * channel_step],
                                         axis=3)[..., np.newaxis]
                                  for i in range(3)], axis=3)
        patches = np.rollaxis(patches, 3, 2).reshape(
            (patches.shape[0], patches.shape[1], patches.shape[2] * 3))
    for i, patch in enumerate(patches[:100]):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(
            patch,
            interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())

    fig.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    return fig
