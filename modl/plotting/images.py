import matplotlib.pyplot as plt
from sklearn.utils import check_array
import numpy as np

def plot_patches(patches, shape):
    fig = plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(patches[:100]):
        patch = patch.reshape(shape)
        step = shape[2] // 3
        plt.subplot(10, 10, i + 1)
        if patch.ndim == 3:
            # patch = np.concatenate([np.mean(patch[:, :,
            #                                 i * step:min(shape[2], (i + 1) * step)],
            #                                 axis=2)[:, :, np.newaxis]
            #                         for i in range(3, 0, -1)], axis=2)
            patch = np.sum(patch, axis=2)
        plt.imshow(
            patch,
            cmap=plt.cm.gray_r,
            interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary',
                 fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    return fig