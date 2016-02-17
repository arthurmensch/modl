# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

import time
from os.path import expanduser

from modl import datasets
from modl._utils.masking import DummyMasker
from modl._utils.system.openblas import num_threads
from modl.spca_fmri import SpcaFmri

adhd_dataset = datasets.fetch_hcp_rest(data_dir='/storage/data', n_subjects=10)

func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

# Apply our decomposition estimator with reduction
n_components = 70
n_jobs = 20

masker = DummyMasker(data_dir='/storage/data/HCP_unmasked',
                     mmap_mode='r')

with num_threads(1):
    dict_fact = SpcaFmri(mask=masker,
                         shelve=False,
                         n_components=n_components,
                         reduction=12,
                         alpha=0.001,
                         random_state=0,
                         n_epochs=1,
                         memory=expanduser("~/nilearn_cache"), memory_level=2,
                         verbose=4,
                         n_jobs=n_jobs,
                         )

print('[Example] Learning maps')
t0 = time.time()
with num_threads(n_jobs):
    dict_fact.fit(func_filenames)
print('[Example] Dumping results')
# Decomposition estimator embeds their own masker
masker = dict_fact.masker_
components_img = masker.inverse_transform(dict_fact.components_)
components_img.to_filename('components.nii.gz')
time = time.time() - t0
print('[Example] Run in %.2f s' % time)
# Show components from both methods using 4D plotting tools
from nilearn.plotting import plot_prob_atlas, show

print('[Example] Displaying')

plot_prob_atlas(components_img, view_type="filled_contours",
                title="Reduced sparse PCA",colorbar=False)
show()