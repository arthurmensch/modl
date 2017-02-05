import itertools
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory

from modl.datasets import fetch_adhd
from modl.utils.system import get_cache_dirs

import matplotlib.pyplot as plt

data = fetch_adhd(n_subjects=3)
rest = data.rest
mask = data.mask

mem = Memory(cachedir=get_cache_dirs()[0])
# mem = Memory(cachedir=None)

masker = MultiNiftiMasker(smoothing_fwhm=6,
                          standardize=True,
                          detrend=True,
                          mask_img=mask,
                          memory=mem,
                          memory_level=1,
                          n_jobs=1).fit()
iter_df = rest.loc[:, ['filename', 'confounds']].iterrows()

data_1 = []

for subject, (img, confounds) in iter_df:
    data_1.append(masker.transform(img, confounds))

imgs = rest.loc[:, 'filename']
confounds = rest.loc[:, 'confounds']

data = masker.transform(imgs, confounds)

for this_data in data:
    plot_stat_map(index_img(masker.inverse_transform(this_data), 0))

for this_data in data_1:
    plot_stat_map(index_img(masker.inverse_transform(this_data), 0))

plt.show()