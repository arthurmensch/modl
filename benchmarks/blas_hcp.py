# Author: Arthur Mensch
# License: BSD
# Adapted from nilearn example

import time
from os.path import expanduser

import numpy as np
from nilearn.datasets import fetch_atlas_smith_2009

from modl._utils.system.mkl import num_threads
from modl.datasets.hcp import get_hcp_data
from modl.spca_fmri import SpcaFmri

mask, func_filenames = get_hcp_data(data_dir='/storage/data')

func_filenames = func_filenames[:2]

# Apply our decomposition estimator with reduction
n_components = 70
n_jobs = 20
raw = True
init = True

dict_fact = SpcaFmri(mask=mask,
                     smoothing_fwhm=3,
                     shelve=not raw,
                     n_components=n_components,
                     dict_init=fetch_atlas_smith_2009().rsn70 if init else None,
                     reduction=12,
                     alpha=0.001,
                     random_state=0,
                     n_epochs=1,
                     memory=expanduser("~/nilearn_cache"), memory_level=2,
                     verbose=4,
                     n_jobs=1,
                     )

print('[Example] Learning maps')
timings = np.zeros(20)
for n_jobs in range(1, 21):
    with num_threads(n_jobs):
        t0 = time.time()
        dict_fact.fit(func_filenames, raw=raw)
        timings[n_jobs - 1] = time.time() - t0

print(timings)
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.plot(np.arange(1, 21), timings)
plt.savefig('bench_hcp_blas.pdf')