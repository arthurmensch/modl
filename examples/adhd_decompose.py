# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import os
import time
from os.path import expanduser, join

from nilearn import datasets

from modl.spca_fmri import SpcaFmri


class Callback(object):
    def __init__(self, train_data, test_data):
        self.train = train_data
        self.test_data = test_data
        self.test_obj = []
        self.train_obj = []
        self.time = []

        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, spca_fmri):
        test_time = time.clock()
        test_obj = spca_fmri.objective_function(self.test_data)
        train_obj = spca_fmri.objective_function(self.train_data)
        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)

adhd_dataset = datasets.fetch_adhd(n_subjects=40)

func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

# Apply our decomposition estimator with reduction
n_components = 20
n_jobs = 3

trace_folder = expanduser('~/output/modl/adhd')
try:
    os.makedirs(trace_folder)
except OSError:
    pass

dict_fact = SpcaFmri(n_components=n_components, smoothing_fwhm=6.,
                     memory=expanduser("~/nilearn_cache"), memory_level=2,
                     reduction=3,
                     projection='partial',
                     var_red='combo',
                     verbose=10,
                     alpha=0.001,
                     random_state=0,
                     learning_rate=.875,
                     replacement=True,
                     coupled_subset=True,
                     batch_size=50,
                     offset=0,
                     n_epochs=2,
                     backend='python',
                     # trace_folder=trace_folder,
                     n_jobs=n_jobs,
                     )

print('[Example] Learning maps')
t0 = time.time()
dict_fact.fit(func_filenames)
print('[Example] Dumping results')
# Decomposition estimator embeds their own masker
masker = dict_fact.masker_
components_img = masker.inverse_transform(dict_fact.components_)
components_img.to_filename(join(trace_folder, 'components.nii.gz'))
time = time.time() - t0
print('[Example] Run in %.2f s' % time)
# Show components from both methods using 4D plotting tools
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show
from nilearn.image import index_img

print('[Example] Displaying')
fig, axes = plt.subplots(2, 1)
plot_prob_atlas(components_img, view_type="filled_contours",
                axes=axes[0])
plot_stat_map(index_img(components_img, 0),
              axes=axes[1],
              colorbar=False,
              threshold=0)
plt.savefig(join(trace_folder, 'components.pdf'))
show()
