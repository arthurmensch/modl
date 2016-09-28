import matplotlib.pyplot as plt


def plot_patches(patches, name):
    fig = plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(patches[:100]):
        plt.subplot(10, 10, i + 1)
        if patch.ndim == 3:
            patch = patch[:, :, :3]
        plt.imshow(
            patch,
            cmap=plt.cm.gray_r,
            interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary',
                 fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.savefig(name)
    plt.close(fig)