from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils import check_random_state

from modl.new.dict_fact import DictMF


# from modl.dict_fact import DictMF


class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr):
        self.X_tr = X_tr
        # self.X_te = X_t
        self.obj = []
        self.times = []
        self.iter = []
        # self.R = []
        self.start_time = time()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time()
        self.obj.append(mf.score(self.X_tr))
        # R = (mf.B_ - mf.A_.dot(mf.D_))
        # scale = np.diag(mf.A_).copy()[:, np.newaxis]
        # scale[scale == 0] = 1
        # R /= scale
        # self.R.append(np.sum(R ** 2))
        self.test_time += time() - test_time
        self.times.append(time() - self.start_time - self.test_time)
        self.iter.append(mf.n_iter_[0])


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
    tile = 4
    patch_size = (8, 8)
    data = extract_patches_2d(distorted[:, :width // 2], patch_size,
                              max_patches=2000, random_state=0)
    tiled_data = np.empty(
        (data.shape[0], data.shape[1] * tile, data.shape[2] * tile))
    for i in range(tile):
        for j in range(tile):
            tiled_data[:, i::tile, j::tile] = data
    data = tiled_data
    patch_size = (8 * tile, 8 * tile)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))
    random_state = check_random_state(0)
    random_state.shuffle(data)

    ###############################################################################
    # Learn the dictionary from reference patches

    print('Learning the dictionary...')

    cb = Callback(data[:500])
    dico = DictMF(n_components=100, alpha=1,
                  l1_ratio=0,
                  pen_l1_ratio=0.9,
                  batch_size=10,
                  learning_rate=.8,
                  sample_learning_rate=None,
                  reduction=6,
                  verbose=1,
                  solver='gram',
                  weights='sync',
                  subset_sampling='random',
                  dict_subset_sampling='independent',
                  callback=cb,
                  n_threads=2,
                  backend='c',
                  tol=1e-2,
                  random_state=42,
                  n_epochs=100)
    t0 = time()
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

    fig, axes = plt.subplots(1, 1, sharex=True)
    axes.plot(cb.iter[1:], np.array(cb.obj[1:]))
    axes.set_ylabel('Function value')
    #
    plt.show()


if __name__ == '__main__':
    main()
