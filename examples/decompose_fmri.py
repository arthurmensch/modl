# Author: Arthur Mensch
# License: BSD
import time

from modl.utils.nifti import monkey_patch_nifti_image

monkey_patch_nifti_image()

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.datasets import fetch_adhd
from modl.datasets.fmri import load_atlas_init
from modl.fmri import fMRIDictFact
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs


class rfMRIDictionaryScorer:
    def __init__(self, test_data):
        self.start_time = time.perf_counter()
        self.test_data = test_data
        self.test_time = 0
        self.score = []
        self.iter = []
        self.time = []

    def __call__(self, dict_fact):
        test_time = time.perf_counter()
        test_imgs, test_confounds = zip(*self.test_data)
        score = dict_fact.score(test_imgs, confounds=test_confounds)
        self.test_time += time.perf_counter() - test_time
        this_time = time.perf_counter() - self.start_time - self.test_time
        self.score.append(score)
        self.time.append(this_time)
        self.iter.append(dict_fact.n_iter_)


def main():
    n_components = 20
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    verbose = 15
    n_jobs = 4
    warmup = True
    smoothing_fwhm = 6

    dict_init = load_atlas_init('smith', n_components=n_components)

    dataset = fetch_adhd(n_subjects=40)
    data = dataset.rest.values
    train_data, test_data = train_test_split(data, test_size=1, random_state=0)
    train_imgs, train_confounds = zip(*train_data)
    mask = dataset.mask
    memory = Memory(cachedir=get_cache_dirs()[0],
                    verbose=2)

    cb = rfMRIDictionaryScorer(test_data)
    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             method=method,
                             mask=mask,
                             memory=memory,
                             memory_level=2,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=1,
                             n_components=n_components,
                             dict_init=dict_init,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             callback=cb,
                             warmup=warmup,
                             )
    dict_fact.fit(train_imgs, train_confounds)

    dict_fact.components_.to_filename('components.nii.gz')
    fig = plt.figure()
    display_maps(fig, dict_fact.components_)
    fig, ax = plt.subplots(1, 1)
    ax.plot(cb.time, cb.score, marker='o')
    plt.show()

if __name__ == '__main__':
    main()
