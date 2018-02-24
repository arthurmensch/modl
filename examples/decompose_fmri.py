# Author: Arthur Mensch
# License: BSD

import matplotlib as mpl
mpl.use('Qt5Agg')

from nilearn.datasets import fetch_atlas_smith_2009
from modl.input_data.fmri.fixes import monkey_patch_nifti_image

monkey_patch_nifti_image()

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.datasets import fetch_adhd
from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs


n_components = 20
batch_size = 50
learning_rate = .92
method = 'masked'
step_size = 0.01
reduction = 12
alpha = 1e-3
n_epochs = 2
verbose = 15
n_jobs = 2
smoothing_fwhm = 6

dict_init = fetch_atlas_smith_2009().rsn20

dataset = fetch_adhd(n_subjects=40)
data = dataset.rest.values
train_data, test_data = train_test_split(data, test_size=1, random_state=0)
train_imgs, train_confounds = zip(*train_data)
test_imgs, test_confounds = zip(*test_data)
mask = dataset.mask
memory = Memory(cachedir=get_cache_dirs()[0],
                verbose=2)

cb = rfMRIDictionaryScorer(test_imgs, test_confounds=test_confounds)
dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                         method=method,
                         step_size=step_size,
                         mask=mask,
                         memory=memory,
                         memory_level=2,
                         verbose=verbose,
                         n_epochs=n_epochs,
                         n_jobs=n_jobs,
                         random_state=1,
                         n_components=n_components,
                         dict_init=dict_init,
                         positive=True,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         reduction=reduction,
                         alpha=alpha,
                         callback=cb,
                         )
dict_fact.fit(train_imgs, confounds=train_confounds)

fig = plt.figure()
display_maps(fig, dict_fact.components_img_)
fig, ax = plt.subplots(1, 1)
ax.plot(cb.time, cb.score, marker='o')
plt.show()
