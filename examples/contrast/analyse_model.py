import os
from os.path import join

from keras.models import load_model
from numpy.linalg import lstsq
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import ridge_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import matplotlib

from examples.contrast import bhtsne
from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory, load
import numpy as np

from sklearn.decomposition import PCA

import pandas as pd

from seaborn.apionly import husl_palette

from jinja2 import Environment, PackageLoader, select_autoescape

# Load model
# y =X_full W_g D^-1 W_e W_s
memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

dictionary_penalty = 1e-4
n_components_list = [16, 64, 256]

artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                    'prediction_hierarchical', 'good')

analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                    'prediction_hierarchical', 'analysis')

if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()
standard_scaler = load(join(artifact_dir, 'standard_scaler.pkl'))  # D
# Expect shape (?, 336)
labels = load(join(artifact_dir, 'labels.pkl'))

X = load(join(artifact_dir, 'X.pkl'))
y_pred = load(join(artifact_dir, 'y_pred_depth_1.pkl'))
y_pred = pd.DataFrame(y_pred, index=X.index)

model = load_model(join(artifact_dir, 'model.keras'),
                   custom_objects={'HierarchicalLabelMasking':
                                       HierarchicalLabelMasking,
                                   'PartialSoftmax': PartialSoftmax})

weight_latent = model.get_layer('latent').get_weights()[0]  # W_e
# Shape (336, 50)
weight_supervised, bias = model.get_layer('supervised_depth_1').get_weights()  # W_s
# Shape (50, 97)

# weight_latent, r = np.linalg.qr(weight_latent)
# weight_supervised = r.dot(weight_supervised)

# Latent factors
bases = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                          n_components_list)  # W_g
for i, basis in enumerate(bases):
    S = np.std(basis, axis=1)
    S[S == 0] = 0
    basis = basis / S[:, np.newaxis]
    bases[i] = basis
bases = np.concatenate(bases, axis=0).T
# Shape (200000, 336)

# Forward
classification_vectors = bases.dot(standard_scaler.inverse_transform(weight_latent.dot(weight_supervised).T).T).T

# Backward
latent_loadings = ridge_regression(weight_latent.T, np.eye(weight_latent.shape[1]), 0)  # (50, 336)
latent_loadings = standard_scaler.transform(latent_loadings).T  # (336, 50)
latent_imgs = ridge_regression(bases.T, latent_loadings, 0)

latent_freqs = weight_supervised
latent_freqs = np.exp(latent_freqs)
latent_freqs /= np.sum(latent_freqs, axis=1, keepdims=True)

# Save
latent_imgs_nii = masker.inverse_transform(latent_imgs)
latent_imgs_nii.to_filename(join(analysis_dir, 'latent_imgs.nii.gz'))

classification_vectors_nii = masker.inverse_transform(classification_vectors)
classification_vectors_nii.to_filename(join(analysis_dir,
                                            'classification_vectors.nii.gz'))

# CCA + PCA
# data = pd.concat([X, y_pred], names=['type'], keys=['input', 'output'],
#                  axis=1)
# data_sub = []
# min_samples = data.iloc[:, 0].groupby(level='dataset').aggregate(
#     'count').min()
# random_state = check_random_state(0)
# # Same number of samples for all datasets
# # TODO: Use sample weights instead
# for dataset, sub_data in data.groupby(level='dataset'):
#     indices = random_state.permutation(sub_data.shape[0])[:min_samples]
#     sub_data = sub_data.iloc[indices]
#     data_sub.append(sub_data)
# data = pd.concat(data_sub)
#
# X = data['input']  # X_full W_g
# y_pred = data['output']
# X_proj = standard_scaler.transform(X)  # X_full W_g D^-1
# X_proj = X.values.dot(weight_latent)  # X_full W_g D^-1 W_e
# proj_std_scaler = StandardScaler(with_std=False).fit(X_proj)
# X_proj = proj_std_scaler.transform(X_proj)
#
# n_components = 10
# cca = CCA(n_components=n_components, scale=False)
# cca.fit(X_proj, y_pred)
# cca_loadings = proj_std_scaler.inverse_transform(cca.x_loadings_.T)
#
# pca = PCA(n_components=n_components, whiten=False)
# pca.fit(X_proj)
# pca_loadings = proj_std_scaler.inverse_transform(pca.components_)
#
# cca_imgs = reconstructer.transform(cca_loadings.dot(weight_latent_inv))
# cca_imgs_nii = masker.inverse_transform(cca_imgs)
# cca_imgs_nii.to_filename(join(analysis_dir, 'cca_imgs.nii.gz'))
#
# pca_imgs = reconstructer.transform(pca_loadings.dot(weight_latent_inv))
# pca_imgs_nii = masker.inverse_transform(pca_imgs)
# pca_imgs_nii.to_filename(join(analysis_dir, 'pca_imgs.nii.gz'))

# cca_freqs = cca_loadings.dot(weight_supervised)
# cca_freqs = np.exp(cca_freqs)
# cca_freqs /= np.sum(cca_freqs, axis=1, keepdims=True)
#
# pca_freqs = pca_loadings.dot(weight_supervised)
# pca_freqs = np.exp(pca_freqs)
# pca_freqs /= np.sum(pca_freqs, axis=1, keepdims=True)

# Display
n_components = 50

env = Environment(loader=PackageLoader('modl', 'templates'),
                  autoescape=select_autoescape(['html', 'xml']))

template = env.get_template('analyse_model.html')
template.render()

for imgs, freqs, name in zip([# cca_imgs_nii, pca_imgs_nii,
                              latent_imgs_nii],
                             [# cca_freqs, pca_freqs,
                              latent_freqs],
                             [# 'cca', 'pca',
                              'latent']):
    label_freqs = ['_'.join(label.split('_')[1:]) for label in labels]
    for i in range(n_components):
        img = index_img(imgs, i)
        vmax = np.max(img.get_data())
        these_freqs = {label: freq for label, freq in zip(label_freqs,
                                                          freqs[i])}
        fig, axes = plt.subplots(1, 2,
                                 figsize=(12, 5))
        bbox = axes[1].get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        width = int(width)
        height = int(height)
        wc = WordCloud(width=width, height=height, relative_scaling=1,
                       background_color='white')
        wc.generate_from_frequencies(these_freqs)
        axes[1].imshow(wc)
        axes[1].set_axis_off()
        plot_stat_map(img, figure=fig, axes=axes[0], threshold=vmax / 3,
                      colorbar=True)
        plt.savefig(join(analysis_dir, '%s_%i.png' % (name, i)))


data = classification_vectors_nii.get_data()
vmax = np.max(np.abs(data))
labels_clean = ['_'.join(label.split('_')[1:]) for label in labels]
n_vectors = 97
for i, label in enumerate(labels_clean[:n_vectors]):
    img = index_img(classification_vectors_nii, i)
    vmax = np.max(img.get_data())
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_stat_map(index_img(classification_vectors_nii, i),
                  axes=ax[i], figure=fig, colorbar=True,
                  threshold=vmax / 3,
                  title=label)
    fig.savefig(join(analysis_dir, 'components_%i.png' %i))
    plt.close(fig)