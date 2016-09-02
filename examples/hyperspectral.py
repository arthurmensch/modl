from time import time

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import \
    reconstruct_from_patches_2d, extract_patches_2d
from sklearn.utils import check_random_state, gen_batches

from modl._utils.hyper import *
from modl.dict_fact import DictFact


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


patch_size = (8, 8)
n_components = 100

mem = Memory(cachedir=expanduser('~/cache'))
# data = mem.cache(fetch_data)()
# train_img = data[2]['sp']

full_img = mem.cache(fetch_aviris)()
train_img = full_img[::4, ::4, ::2].copy()
del full_img

n_channels = train_img.shape[2]
height, width = train_img.shape[:-1]

distorted = train_img.copy()
distorted[:, width // 2:, :] += 0.0 * np.random.randn(height, width // 2,
                                                      n_channels)

patches = extract_patches_2d(distorted[:, :width // 2, :], patch_size,
                             # max_patches=100000,
                             random_state=0)

X_train = patches.reshape(patches.shape[0], -1)

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

random_state = check_random_state(0)
X_test = X_train[:1000]
cb = Callback(X_test)
dico = DictFact(n_components=n_components, alpha=0.1,
                batch_size=100,
                pen_l1_ratio=0.9,
                l1_ratio=0,
                n_threads=16,
                n_epochs=30, verbose=2,
                solver='gram',
                weights='sync',
                reduction=10,
                callback=cb,
                random_state=0)

t0 = time()
dico.fit(X_train)
V = dico.components_.reshape(n_components,
                             patch_size[0],
                             patch_size[1], n_channels)
dt = cb.times[-1] if dico.callback != None else time() - t0
print('done in %.2fs., test time: %.2fs' % (dt, cb.test_time))

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp[:, :, 0], cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(X_train)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

fig, axes = plt.subplots(1, 1, sharex=True)
axes.plot(cb.iter[1:], np.array(cb.obj[1:]))
axes.set_ylabel('Function value')
#
output_dir = expanduser('~/output/modl/hyperspectral')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(join(output_dir, 'learning_curve.pdf'))


# noisy_patches = extract_patches_2d(distorted[:, width // 2:], patch_size)
#
# X_test = noisy_patches.reshape(noisy_patches.shape[0], -1)
#
# code = np.empty((X_test.shape[0], n_components))
# batches = gen_batches(code.shape[0], code.shape[0] // 10)
# rec = np.empty_like(X_test)
# print('Transform')
# for i, batch in enumerate(batches):
#     print(i)
#     # code[batch] = dict_learning.transform(X_test[batch]).T
#     code[batch], D = dict_learning.transform(X_test[batch])
#     rec[batch] = code[batch].dot(D)
#
# rec = rec.reshape(-1, patch_size[0], patch_size[1], n_channels)
#
# train_img[:, width // 2:, :] = reconstruct_from_patches_2d(rec, (height,
#                                                                  width - width // 2,
#                                                                  n_channels))
#
# display_sp_img(distorted[:, :, :4])
# display_sp_img(train_img[:, :, :4])
#
# components = dict_learning.components_.reshape(n_components,
#                                                patch_size[0],
#                                                patch_size[1], n_channels)
#
# display_sp_img(components[0, :, :, :16])
# display_sp_img(components[1, :, :, :16])
# display_sp_img(components[2, :, :, :16])
#
# plt.show()
