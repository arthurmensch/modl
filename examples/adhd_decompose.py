    # Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

# Load ADDH
import os
import time
from os.path import expanduser, join

import numpy as np
from modl.spca_fmri import SpcaFmri
from nilearn import datasets

import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show
from nilearn.image import index_img
from sklearn.cross_validation import train_test_split


class Callback(object):
    def __init__(self, train_data,
                 trace_folder=None,
                 test_data=None):
        self.train_data = train_data
        self.test_data = test_data
        self.trace_folder = trace_folder
        if self.test_data is not None:
            self.test_obj = []
        self.train_obj = []
        self.time = []
        self.iter = []

        self.start_time = time.clock()
        self.test_time = 0

        self.profile = []

    def __call__(self, spca_fmri):
        test_time = time.clock()
        train_obj = spca_fmri.score(self.train_data)
        self.train_obj.append(train_obj)
        if self.test_data is not None:
            test_obj = spca_fmri.score(self.test_data)
            self.test_obj.append(test_obj)
        if self.trace_folder is not None:
            spca_fmri.components_.to_filename(join(self.trace_folder,
                                                   "record_"
                                                   "%s.nii.gz"
                                                   % spca_fmri.n_iter))
        self.iter.append(spca_fmri.n_iter)
        self.profile.append(spca_fmri.time)
        self.test_time += time.clock() - test_time
        self.time.append(time.clock() - self.start_time - self.test_time)


def display_maps(index=None):
    name = 'components_' + index + '.png'
    fig, axes = plt.subplots(2, 1)
    plot_prob_atlas(spca_fmri.components_, view_type="filled_contours",
                    axes=axes[0])
    plot_stat_map(index_img(spca_fmri.components_, 0),
                  axes=axes[1],
                  colorbar=False,
                  threshold=0)
    plt.savefig(join(trace_folder, 'components.png'))

adhd_dataset = datasets.fetch_adhd(n_subjects=40)

data = adhd_dataset.func  # list of 4D nifti files for each subject
train_data, test_data = train_test_split(data, test_size=0.1, random_state=0)

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

# Apply our decomposition estimator with reduction
n_components = 20
n_jobs = 1
trace_folder = expanduser('~/output/modl/adhd_reduced')

try:
    os.makedirs(trace_folder)
except OSError:
    pass

cb = Callback(train_data, test_data=test_data,
              trace_folder=trace_folder)
spca_fmri = SpcaFmri(n_components=n_components, smoothing_fwhm=6.,
                     memory=expanduser("~/nilearn_cache"), memory_level=2,
                     reduction=3,
                     verbose=10,
                     alpha=0.001,
                     random_state=0,
                     learning_rate=.8,
                     batch_size=20,
                     AB_agg='full',
                     G_agg='full',
                     Dx_agg='full',
                     offset=0,
                     shelve=True,
                     n_epochs=1,
                     callback=cb,
                     # trace_folder=trace_folder,
                     n_jobs=n_jobs,
                     )

print('[Example] Learning maps')
t0 = time.time()
spca_fmri.fit(train_data)
print('[Example] Dumping results')
# Decomposition estimator embeds their own masker
masker = spca_fmri.masker_
spca_fmri.components_.to_filename(join(trace_folder, 'components.nii.gz'))
time = time.time() - t0
print('[Example] Run in %.2f s' % time)
# Show components from both methods using 4D plotting tools

print('[Example] Displaying')
fig, axes = plt.subplots(2, 1)
plot_prob_atlas(spca_fmri.components_, view_type="filled_contours",
                axes=axes[0])
plot_stat_map(index_img(spca_fmri.components_, 0),
              axes=axes[1],
              colorbar=False,
              threshold=0)
plt.savefig(join(trace_folder, 'components.png'))

fig, axes = plt.subplots(2, 1, sharex=True)

profile = np.array(cb.profile)
iter = np.array(cb.iter)
obj = np.array(cb.train_obj)
average_time = np.zeros_like(profile)
average_time[1:] = (profile[1:] - profile[:-1])\
                   / (iter[1:] - iter[:-1])[:, np.newaxis]

axes[0].plot(iter[1:], obj[1:], marker='o')
axes[1].plot(iter[1:], average_time[1:], marker='o')
axes[1].legend(['Dx time', 'G time', 'Code time', 'Agg time',
                'BCD time', 'Total', 'IO time'])
axes[1].set_ylabel('Average time')
axes[1].set_yscale('Log')
axes[0].set_ylabel('Function value')

plt.savefig(join(trace_folder, 'loss_profile.png'))
