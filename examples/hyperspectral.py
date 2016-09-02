from time import time

import gc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import \
    reconstruct_from_patches_2d, extract_patches_2d
from sklearn.utils import check_random_state, gen_batches

from modl._utils.hyper import *
from modl.dict_fact import DictFact

n_components = 100

def load_aviris(test=False, max_patches=None, random_state=None):
    patch_size = (8, 8)
    full_img = fetch_aviris()
    img = full_img[::4, ::4, ::2].copy()
    del full_img

    n_channels = img.shape[2]
    height, width = img.shape[:-1]

    train_patches = extract_patches_2d(img[:, :width // 2, :], patch_size,
                                       max_patches=max_patches,
                                       random_state=random_state)
    if test:
        test_patches = extract_patches_2d(img[:, width // 2:, :], patch_size,
                                          max_patches=max_patches,
                                          random_state=random_state)
        return train_patches, test_patches
    return train_patches

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
        self.test_time += time() - test_time
        self.times.append(time() - self.start_time - self.test_time)
        self.iter.append(mf.total_counter)

def run(n_jobs=1):
    output_dir = expanduser('~/output/modl/hyperspectral')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mem = Memory(cachedir=expanduser('~/cache'))

    train_patches = mem.cache(load_aviris)(test=False, max_patches=10000,
                                           random_state=0)

    n_samples, height, width, n_channels = train_patches.shape
    X_train = train_patches.reshape(n_samples, -1)

    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)

    X_test = X_train[:1000]

    cb = Callback(X_test)
    dico = DictFact(n_components=n_components, alpha=0.1,
                    batch_size=100,
                    pen_l1_ratio=0.9,
                    l1_ratio=0,
                    n_threads=1,
                    n_epochs=100, verbose=2,
                    solver='gram',
                    weights='sync',
                    reduction=10,
                    callback=cb,
                    random_state=0)

    t0 = time()
    dico.fit(X_train)
    V = dico.components_.reshape(n_components,
                                 height,
                                 width, n_channels)
    dt = cb.times[-1] if dico.callback != None else time() - t0
    print('done in %.2fs., test time: %.2fs' % (dt, cb.test_time))

    fig = plt.figure(figsize=(4.2, 8))
    for channel in range(2):
        for i, comp in enumerate(V[:100]):
            ax = fig.add_subplot(20, 10, 2 * i + 1 + channel)
            ax.imshow(comp[:, :, channel], cmap=plt.cm.gray_r,
                       interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
            plt.suptitle('Dictionary learned from face patches\n' +
                         'Train time %.1fs on %d patches' % (dt, len(X_train)),
                         fontsize=16)
            fig.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.savefig(join(output_dir, 'components.pdf'))
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, sharex=True)
    obj = np.array(cb.obj)
    iter = np.array(cb.obj)
    ax.plot(iter, obj)
    ax.set_ylim([0.9 * np.min(obj[1:]), 1.1 * np.max(obj[1:])])
    ax.set_ylabel('Function value')
    plt.savefig(join(output_dir, 'learning_curve.pdf'))

if __name__ == '__main__':
    run(16)
