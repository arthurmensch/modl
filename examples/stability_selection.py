# Author: Arthur Mensch
# License: BSD
from os.path import expanduser

from sklearn.utils import check_random_state

from modl.datasets.image import load_image
from modl.decomposition.image import ImageDictFact
from modl.decomposition.stability import mean_amari_discrepency
from modl.feature_extraction.image import LazyCleanPatchExtractor

from joblib import Parallel, delayed, Memory

import numpy as np

from modl.plotting.image import plot_patches

batch_size = 400
learning_rate = 0.92
reduction_ = 2
alpha = 0.08
n_epochs = 5
n_components = 100
test_size = 4000
max_patches = 1000
patch_size = (32, 32)
n_threads = 2
verbose = 100
method = 'masked'
step_size = 0.1
setting = 'dictionary learning'
source = 'lisboa'
gray = False
scale = 1

n_jobs = 4
components_list = [5, 10, 20, 25, 40, 60, 80]
random_states = check_random_state(0).randint(0, 10000, size=10)

print('Loading data')
image = load_image(source, scale=scale, gray=gray)
width, height, n_channel = image.shape
patch_extractor = LazyCleanPatchExtractor(patch_size=patch_size,
                                          max_patches=test_size,
                                          random_state=1)
test_data = patch_extractor.transform(image[:, :height // 2, :])

mem = Memory(location=expanduser('~/cache'))


def single_fit(image, test_data, n_components, random_state):
    dict_fact = ImageDictFact(method=method,
                              setting=setting,
                              alpha=alpha,
                              step_size=step_size,
                              n_epochs=n_epochs,
                              n_components=n_components,
                              learning_rate=learning_rate,
                              max_patches=max_patches,
                              batch_size=batch_size,
                              patch_size=patch_size,
                              reduction=reduction_,
                              callback=None,
                              verbose=verbose,
                              n_threads=n_threads,
                              random_state=random_state
                              )
    dict_fact.fit(image[:, height // 2:, :])
    score = dict_fact.score(test_data)
    return dict_fact.components_, score


dictionaries = Parallel(n_jobs=n_jobs)(
    delayed(mem.cache(single_fit))(image, test_data,
                                   n_components, random_state)
    for n_components in components_list
    for random_state in random_states)

dictionaries_ = {}
scores = {}
for n_components in components_list:
    dictionaries_[n_components] = []
    scores[n_components] = []
    for random_state in random_states:
        (this_dict, this_score), dictionaries = (dictionaries[0],
                                                 dictionaries[1:])
        dictionaries_[n_components].append(this_dict)
        scores[n_components].append(this_score)

discrepencies = np.array([mean_amari_discrepency([this_dict.reshape((this_dict.shape[0], -1))
                                                  for this_dict in dictionaries_[n_components]])
                         for n_components in components_list])
best_n_components = components_list[np.argmin(discrepencies)]
best_dictionary_idx = np.argmin(np.array(scores[best_n_components]))
best_dictionary = dictionaries_[best_n_components][best_dictionary_idx]


import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure()
plot_patches(fig, best_dictionary)
fig.suptitle('Dictionary')

fig, ax = plt.subplots(1, 1)
ax.plot(components_list, discrepencies, marker='o')
ax.set_xlabel('Number of components')
ax.set_ylabel('Mean Amari discrepency')
sns.despine(fig)
fig.suptitle('Stability selection using DL')

plt.show()
