from os.path import join, expanduser

from keras.models import load_model
from nilearn._utils import check_niimg

from modl.datasets import fetch_hcp, get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax
from modl.classification import Reconstructer
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, plot_prob_atlas
from sklearn.externals.joblib import Memory

import matplotlib.pyplot as plt

import numpy as np

# dictionary_penalty = 1e-4
# n_components_list = [16, 64, 256]
# scale_importance = 'sqrt'
#
# artifact_dir = expanduser('~/data/modl_data/pipeline/contrast/'
#                           'prediction_hierarchical/None')
# model = load_model(join(artifact_dir, 'model.keras'),
#                    custom_objects={'HierarchicalLabelMasking':
#                                        HierarchicalLabelMasking,
#                                    'PartialSoftmax': PartialSoftmax})
#
# memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
# print('Fetch data')
# mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
# masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()
# print('Retrieve components')
# bases = memory.cache(retrieve_components)(dictionary_penalty, masker,
#                                           n_components_list)
# for i, basis in enumerate(bases):
#     S = np.std(basis, axis=1)
#     S[S == 0] = 0
#     basis = basis / S[:, np.newaxis]
#     bases[i] = basis
# reconstructer = Reconstructer(bases=bases,
#                               scale_importance=scale_importance)
# weights = model.get_layer('latent').get_weights()[0].T
# n_components = weights.shape[0]
# imgs = reconstructer.fit_transform(weights)
# imgs = masker.inverse_transform(imgs)
#
# imgs.to_filename('components.nii.gz')

imgs = check_niimg('components.nii.gz')
n_components = imgs.shape[3]
data = imgs.get_data()
vmax = np.max(np.abs(data))
fig, axes = plt.subplots(25, 1, figsize=(8, 50))
axes = np.ravel(axes)
for i in range(n_components):
    img = index_img(imgs, i)
    plot_stat_map(img, figure=fig, axes=axes[i], threshold=0,
                  colorbar=True, vmax=vmax)
plt.savefig('components.pdf')
# plot_prob_atlas(imgs)
# plt.show()