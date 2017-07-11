# Author: Arthur Mensch
# License: BSD
import time
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from modl.datasets.image import load_image
from modl.decomposition.image import ImageDictFact
from modl.feature_extraction.image import LazyCleanPatchExtractor
from modl.plotting.image import plot_patches


class DictionaryScorer:
    def __init__(self, test_data):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        self.time = []
        self.score = []
        self.iter = []

    def __call__(self, dict_fact):
        test_time = time.clock()
        score = dict_fact.score(self.test_data)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        self.time.append(this_time)
        self.score.append(score)
        self.iter.append(dict_fact.n_iter_)

batch_size = 400
learning_rate = 0.92
reduction = 10
alpha = 0.08
n_epochs = 1
n_components = 400
test_size = 4000
max_patches = 10000
patch_size = (32, 32)
n_threads = 2
verbose = 0
method = 'gram'
step_size = 0.1
setting = 'dictionary learning'
source = 'aviris'
gray = False
scale = 1

print('Loading data')
image = load_image(source, scale=scale, gray=gray)
print('Done')
width, height, n_channel = image.shape
patch_extractor = LazyCleanPatchExtractor(patch_size=patch_size,
                                          max_patches=test_size,
                                          random_state=1)
test_data = patch_extractor.transform(image[:, :height // 2, :])
cb = DictionaryScorer(test_data)
dict_fact = ImageDictFact(method=method,
                          setting=setting,
                          alpha=alpha,
                          step_size=step_size,
                          n_epochs=n_epochs,
                          random_state=1,
                          n_components=n_components,
                          learning_rate=learning_rate,
                          max_patches=max_patches,
                          batch_size=batch_size,
                          patch_size=patch_size,
                          reduction=reduction,
                          callback=cb,
                          verbose=verbose,
                          n_threads=n_threads,
                          )
dict_fact.fit(image[:, height // 2:, :])
score = dict_fact.score(test_data)

fig = plt.figure()
patches = dict_fact.components_
plot_patches(fig, patches)
fig.suptitle('Dictionary')

fig, ax = plt.subplots(1, 1)
ax.plot(cb.time, cb.score, marker='o')
ax.legend()
ax.set_xscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Test objective value')

plt.show()