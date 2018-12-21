# Author: Arthur Mensch
# License: BSD
import warnings

from nilearn.input_data import NiftiMasker

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Memory, dump
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from modl.datasets import fetch_adhd
from modl.decomposition.fmri import fMRIDictFact
from modl.decomposition.stability import mean_amari_discrepency
from modl.plotting.fmri import display_maps

from nilearn.datasets import fetch_atlas_smith_2009

from modl.utils.system import get_cache_dirs

batch_size = 200
learning_rate = .92
method = 'masked'
step_size = 0.01
reduction_ = 8
alpha = 1e-3
n_epochs = 4
verbose = 15
n_jobs = 70
smoothing_fwhm = 6
components_list = [20, 40, 80, 120, 200, 300, 500]
n_runs = 20

dict_init = fetch_atlas_smith_2009().rsn20

dataset = fetch_adhd(n_subjects=40)
data = dataset.rest.values
train_data, test_data = train_test_split(data, test_size=2, random_state=0)
train_imgs, train_confounds = zip(*train_data)
test_imgs, test_confounds = zip(*test_data)
mask = dataset.mask
mem = Memory(location=get_cache_dirs()[0])
masker = NiftiMasker(mask_img=mask).fit()


def fit_single(train_imgs, test_imgs, n_components, random_state):
    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             method=method,
                             step_size=step_size,
                             mask=mask,
                             memory=mem,
                             memory_level=2,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=1,
                             random_state=random_state,
                             n_components=n_components,
                             positive=True,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction_,
                             alpha=alpha,
                             callback=None,
                             )
    dict_fact.fit(train_imgs, confounds=train_confounds)
    score = dict_fact.score(test_imgs)
    return dict_fact.components_, score


def fit_many_runs(train_imgs, test_imgs, components_list, n_runs=10, n_jobs=1):
    random_states = check_random_state(0).randint(0, int(1e7), size=n_runs)
    cached_fit = mem.cache(fit_single)

    res = Parallel(n_jobs=n_jobs)(delayed(cached_fit)(
        train_imgs, test_imgs, n_components, random_state)
                                  for n_components in components_list
                                  for random_state in random_states
                                  )
    components, scores = zip(*res)
    shape = (len(components_list), len(random_states))
    components = np.array(components).reshape(shape).tolist()
    scores = np.array(scores).reshape(shape).tolist()

    discrepencies = []
    var_discrepencies = []
    best_components = []
    for n_components, these_components, these_scores in zip(components_list,
                                                            components,
                                                            scores):
        discrepency, var_discrepency = mean_amari_discrepency(
            these_components)
        best_estimator = these_components[np.argmin(these_scores)]
        discrepencies.append(var_discrepency)
        var_discrepencies.append(var_discrepency)
        best_components.append(best_estimator)

    discrepencies = np.array(discrepencies)
    var_discrepencies = np.array(var_discrepencies)
    best_components = np.array(best_components)
    components = best_components[np.argmin(discrepencies)]

    return discrepencies, var_discrepencies, components


output_dir = expanduser('~/output_drago4/modl/fmri_stability2')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

discrepencies, var_discrepencies, components = fit_many_runs(
    train_imgs, test_imgs,
    components_list,
    n_jobs=n_jobs,
    n_runs=n_runs)

components_img = masker.inverse_transform(components)
components_img.to_filename(
    join(output_dir, 'components.nii.gz'))
dump((components_list, discrepencies, var_discrepencies),
     join(output_dir, 'discrepencies.pkl'))

fig = plt.figure()
display_maps(fig, components_img)
plt.savefig(join(output_dir, 'components.pdf'))
fig, ax = plt.subplots(1, 1)
ax.fill_between(components_list, discrepencies - var_discrepencies,
                discrepencies + var_discrepencies, alpha=0.5)
ax.plot(components_list, discrepencies, marker='o')
ax.set_xlabel('Number of components')
ax.set_ylabel('Mean Amari discrepency')
sns.despine(fig)
fig.suptitle('Stability selection using DL')
plt.savefig(join(output_dir, 'discrepencies.pdf'))
