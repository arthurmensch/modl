import os

from os.path import join

from modl.datasets import get_data_dirs
from modl.datasets.hcp import fetch_hcp, INTERESTING_CONTRASTS
from nilearn.input_data import MultiNiftiMasker
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import gen_batches
from sklearn.externals.joblib import dump

import numpy as np

n_jobs = 10
batch_size = 1200
dataset = fetch_hcp()
imgs = dataset.contrasts
mask = dataset.mask

dump_dir = join(get_data_dirs()[0], 'raw', 'hcp', 'task')
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# Selection of contrasts
interesting_con = list(INTERESTING_CONTRASTS.keys())
imgs = imgs.loc[(slice(None), slice(None), interesting_con), :]

contrast_labels = imgs.index.get_level_values(2).values
label_encoder = LabelEncoder()
contrast_labels = label_encoder.fit_transform(contrast_labels)
imgs = imgs.assign(label=contrast_labels)

masker = MultiNiftiMasker(smoothing_fwhm=0, mask_img=mask,
                          n_jobs=n_jobs).fit()

batches = gen_batches(len(imgs), batch_size)

np.save(join(dump_dir, 'labels.npy'), contrast_labels)
dump(label_encoder, join(dump_dir, 'label_encoder.pkl'))
dump(imgs.index, join(dump_dir, 'index.pkl'))

data = np.lib.format.open_memmap(join(dump_dir,
                                      'z_maps.npy'),
                                 mode='w+',
                                 shape=(len(imgs),
                                        masker.mask_img_.get_data().sum()),
                                 dtype=np.float32)

for i, batch in enumerate(batches):
    print('Batch %i' % i)
    this_data = masker.transform(imgs['z_map'].values[batch],
                                 )
    data[batch] = this_data
