from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import check_random_state

from modl.dict_fact import DictFact
from math import sqrt
import seaborn.apionly as sns
class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr, n_threads=None):
        self.X_tr = X_tr
        # self.X_te = X_t
        self.obj = []
        self.times = []
        self.iter = []
        # self.R = []
        self.start_time = time()
        self.test_time = 0
        self.profile = []
        self.n_threads = n_threads

    def __call__(self, mf):
        test_time = time()
        # print(np.diag(mf.A))
        self.obj.append(mf.score(self.X_tr, n_threads=self.n_threads))
        self.test_time += time() - test_time
        self.times.append(time() - self.start_time - self.test_time)
        self.profile.append(mf.time)
        self.iter.append(mf.total_counter)


def main():
    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    face = misc.face(gray=True)
    face = face / 255

    # downsample for higher speed
    # face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    # face /= 4.0
    height, width = face.shape

    # Distort the right half of the image
    print('Distorting image...')
    distorted = face.copy()
    # distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    redundancy = 36
    tile = int(sqrt(redundancy))
    patch_size = (8, 8)
    data = extract_patches_2d(distorted[:, :width // 2], patch_size,
                              max_patches=4000, random_state=0)
    tiled_data = np.empty(
        (data.shape[0], data.shape[1] * tile, data.shape[2] * tile))
    for i in range(tile):
        for j in range(tile):
            tiled_data[:, i::tile, j::tile] = data
    data = tiled_data
    patch_size = (patch_size[0] * tile, patch_size[1] * tile)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))
    random_state = check_random_state(0)
    random_state.shuffle(data)

    ###############################################################################
    # Learn the dictionary from reference patches

    print('Learning the dictionary...')

    cb = Callback(data, n_threads=2)
    n_samples = data.shape[0]
    dico = DictFact(n_components=100, alpha=1,
                    l1_ratio=0,
                    pen_l1_ratio=.9,
                    batch_size=50,
                    learning_rate=.9,
                    sample_learning_rate=None,
                    reduction=5,
                    verbose=1,
                    G_agg='average',
                    Dx_agg='average',
                    AB_agg='full',
                    proj='partial',
                    subset_sampling='random',
                    dict_reduction='follow',
                    callback=cb,
                    n_threads=1,
                    n_samples=n_samples,
                    lasso_tol=1e-2,
                    # purge_tol=1e-3,
                    random_state=42,
                    n_epochs=20)
    # warmup = 1 * n_samples
    # t0 = time()
    # reduction = dico.reduction
    # dico.set_params(reduction=1)s
    # warmup_epochs = warmup // n_samples
    # for _ in range(warmup_epochs):
    #     dico.partial_fit(data)
    # warmup_rem = warmup % n_samples
    # if warmup_rem != 0:
    #     dico.partial_fit(data[:warmup_rem], np.arange(warmup, dtype='i4'))
    #     dico.set_params(reduction=reduction, purge_tol=1e-1)
    #     dico.partial_fit(data[warmup_rem:],
    #                      np.arange(warmup, n_samples, dtype='i4'))
    # else:
    #     dico.set_params(reduction=reduction, purge_tol=1e-1)
    # for i in range(dico.n_epochs - warmup_epochs):
    #     dico.partial_fit(data)
    dico.fit(data)
    V = dico.components_
    dt = cb.times[-1] if dico.callback != None else time() - t0
    print('done in %.2fs., test time: %.2fs' % (dt, cb.test_time))

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from face patches\n' +
                 'Train time %.1fs on %d patches' % (dt, len(data)),
                 fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(right=0.8)

    profile = np.array(cb.profile)
    iter = np.array(cb.iter)
    obj = np.array(cb.obj)

    axes[0].plot(iter[1:], obj[1:], marker='o')
    axes[0].set_ylabel('Function value')

    profile = profile[:, [0, 3, 2, 1, 4]]
    labels = np.array(
        ['', 'Dx time', 'Agg time', 'Code time', 'G time',
         'BCD time'])
    average_time = np.zeros((profile.shape[0] - 1, profile.shape[1] + 1))
    average_time[:, 1:] = (profile[1:] - profile[:-1]) \
                          / (iter[1:] - iter[:-1])[:, np.newaxis]
    average_time = np.cumsum(average_time, axis=1)

    palette = sns.color_palette("deep", 5)
    for i in range(1, 6):
        axes[1].fill_between(iter[1:], average_time[:, i],
                             average_time[:, i - 1],
                             facecolor=palette[i - 1], label=labels[i])
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(reversed(handles), reversed(labels), loc='upper left',
                    bbox_to_anchor=(1, 1),)
    axes[1].set_ylabel('Average time')
    # axes[1].set_yscale('Log')
    axes[0].set_ylabel('Function value')
    axes[1].set_ylim([0, 0.002])
    plt.show()


if __name__ == '__main__':
    main()
