# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

import time
from os.path import expanduser

from nilearn.datasets import fetch_atlas_smith_2009

from modl import datasets
from modl.spca_fmri import SpcaFmri

hcp_dataset = datasets.fetch_hcp_rest(data_dir='/storage/data', n_subjects=10)
mask = '/storage/data/HCP_mask/mask_img.nii.gz'

func_filenames = hcp_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      hcp_dataset.func[0])  # 4D data

# Apply our decomposition estimator with reduction
n_components = 70
n_jobs = 20

# mask = DummyMasker(data_dir='/storage/data/HCP_unmasked',
#                    mask_img='/storage/data/HCP_mask/mask_img.nii.gz',
#                    mmap_mode='r')

dict_fact = SpcaFmri(mask=mask,
                     smoothing_fwhm=3,
                     shelve=True,
                     n_components=n_components,
                     dict_init=fetch_atlas_smith_2009().rsn70,
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
                title="Reduced sparse PCA", colorbar=False)
show()
