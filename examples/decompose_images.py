# Author: Arthur Mensch
# License: BSD
import time
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from modl.datasets.image import load_image
from modl.decomposition.image import ImageDictFact, DictionaryScorer
from modl.feature_extraction.image import LazyCleanPatchExtractor
from modl.plotting.image import plot_patches


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
source = 'lisboa'
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