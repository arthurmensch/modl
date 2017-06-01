# Author: Arthur Mensch
# License: BSD
import os
from os.path import join

from nilearn.datasets import fetch_atlas_smith_2009

from modl.input_data.fmri.fixes import monkey_patch_nifti_image
from modl.input_data.fmri.rest import get_raw_rest_data

monkey_patch_nifti_image()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_output_dir

n_components = 20
batch_size = 200
learning_rate = 0.92
method = 'masked'
reduction = 10
alpha = 1e-3
n_epochs = 1
verbose = 15
n_jobs = 5
smoothing_fwhm = 6

dict_init = fetch_atlas_smith_2009().rsn20

artifact_dir = join(get_output_dir(), 'unmasked', 'hcp')

masker, data = get_raw_rest_data(artifact_dir)

train_imgs, test_imgs = train_test_split(data, test_size=1, random_state=0)
train_imgs = train_imgs['filename'].values[:40]
test_imgs = test_imgs['filename'].values

cb = rfMRIDictionaryScorer(test_imgs)
dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                         method=method,
                         mask=masker,
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
                         )
dict_fact.fit(train_imgs)
output_dir = join(get_output_dir(), 'components', 'hcp')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dict_fact.components_img_.to_filename(join(output_dir, 'components.nii.gz'))
fig = plt.figure()
display_maps(fig, dict_fact.components_img_)
plt.savefig(join(output_dir, 'components.png'))

fig, ax = plt.subplots(1, 1)
ax.plot(cb.time, cb.score, marker='o')
plt.savefig(join(output_dir, 'score.png'))
