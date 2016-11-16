import numpy as np
import matplotlib.cm as cm


def plot_patches(fig, patches, shape):
    for i, patch in enumerate(patches[:100]):
        patch = patch.reshape(shape)
        step = shape[2] // 3
        ax = fig.add_subplot(10, 10, i + 1)
        if patch.ndim == 3:
            # patch = np.sum(patch, axis=2)
            # ax.imshow(
            #     np.rollaxis(patch[:, :, :3], 2, 1).reshape((patch.shape[0], patch.shape[1] * 3)),
            #     cmap=cm.gray_r,
            #     interpolation='nearest')
            ax.imshow(
                patch[:, :, :3],
                interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())

    fig.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    return fig
