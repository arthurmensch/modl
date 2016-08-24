from time import time

import matplotlib.pyplot as plt
import numpy as np
from modl.dict_fact import DictMF
from scipy import misc
from sklearn.feature_extraction.image import extract_patches_2d

class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr):
        self.X_tr = X_tr
        # self.X_te = X_t
        self.obj = []
        self.times = []
        self.sparsity = []
        self.iter = []
        self.beta_var = []
        self.B_var = []
        self.G_var = []
        self.R = []
        self.start_time = time()
        self.test_time = 0

    def __call__(self, mf, sample_subset=None):
        test_time = time()
        self.obj.append(mf.score(self.X_tr))
        beta = self.X_tr.dot(mf.components_.T)
        self.beta_var.append(np.sum((beta - mf.beta_) ** 2))
        self.B_var.append(np.sum((mf.full_B_ - mf.B_) ** 2))
        R = (mf.B_ - mf.A_.dot(mf.D_))
        scale = np.diag(mf.A_).copy()[:, np.newaxis]
        scale[scale == 0] = 1
        R /= scale
        self.R.append(np.sum(R ** 2))
        G = mf.D_.dot(mf.D_.T)
        self.G_var.append(np.sum((G - mf.G_beta_[sample_subset]) ** 2) / np.sum(G ** 2) / sample_subset.shape[0])
        self.sparsity.append(np.sum(mf.components_ != 0) / mf.components_.size)
        self.test_time += time() - test_time
        self.times.append(time() - self.start_time - self.test_time)
        self.iter.append(mf.n_iter_[0] + 1)


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
    tile = 1
    patch_size = (8, 8)
    data = extract_patches_2d(distorted[:, :width // 2], patch_size,
                              max_patches=2000, random_state=0)
    tiled_data = np.empty((data.shape[0], data.shape[1] * tile, data.shape[2] * tile))
    for i in range(tile):
        for j in range(tile):
            tiled_data[:, i::tile, j::tile] = data
    data = tiled_data
    patch_size = (8 * tile, 8 * tile)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))

    ###############################################################################
    # Learn the dictionary from reference patches

    print('Learning the dictionary...')
    t0 = time()

    cb = Callback(data)
    dico = DictMF(n_components=100, alpha=1,
                  l1_ratio=0,
                  pen_l1_ratio=0.9,
                  batch_size=10,
                  learning_rate=1,
                  reduction=6,
                  verbose=1,
                  projection='partial',
                  solver='masked',
                  replacement=True,
                  coupled_subset=False,
                  backend='python',
                  n_samples=2000,
                  full_B=True,
                  callback=cb,
                  random_state=0)
    for i in range(100):
        dico.partial_fit(data)
    V = dico.components_
    dt = time() - t0
    print('done in %.2fs.' % dt)

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

    fig, axes = plt.subplots(5, 1, sharex=True)
    axes[0].plot(cb.iter[1:], cb.obj[1:])
    axes[0].legend()
    axes[0].set_ylabel('Function value')
    axes[0].set_xscale('log')
    axes[1].plot(cb.iter[1:], cb.beta_var[1:])
    axes[1].set_ylabel('beta variance')
    # axes[1].set_yscale('log')
    axes[2].plot(cb.iter[1:], cb.B_var[1:])
    axes[2].set_ylabel('B variance')
    # axes[2].set_yscale('log')
    axes[3].plot(cb.iter[1:], cb.R[1:])
    axes[3].set_xlabel('Iter')
    axes[3].set_ylabel('Residual')

    axes[4].plot(cb.iter[1:], cb.G_var[1:])
    axes[4].set_xlabel('Iter')
    axes[4].set_ylabel('G variance')

    plt.show()

if __name__ == '__main__':
    main()