import os
from os.path import join

import matplotlib
from sklearn.decomposition import PCA

matplotlib.use('agg')

from keras.models import load_model
from sklearn.cross_decomposition import CCA
from sklearn.utils import check_random_state
from wordcloud import WordCloud

from examples.contrast import bhtsne
from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax
from modl.classification import Reconstructer
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory, load
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from seaborn.apionly import husl_palette


def get_imgs(artifact_dir, scale_importance, dictionary_penalty,
             n_components_list):
    model = load_model(join(artifact_dir, 'model.keras'),
                       custom_objects={'HierarchicalLabelMasking':
                                           HierarchicalLabelMasking,
                                       'PartialSoftmax': PartialSoftmax})
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    print('Fetch data')
    mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()
    print('Retrieve components')
    bases = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                              n_components_list)
    for i, basis in enumerate(bases):
        S = np.std(basis, axis=1)
        S[S == 0] = 0
        basis = basis / S[:, np.newaxis]
        bases[i] = basis
    reconstructer = Reconstructer(bases=bases,
                                  scale_importance=scale_importance)
    weights = model.get_layer('latent').get_weights()[0].T
    standard_scaler = load(join(artifact_dir, 'standard_scaler.pkl'))
    weights = standard_scaler.inverse_transform(weights)
    imgs = reconstructer.fit_transform(weights)
    return imgs


def run():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]
    scale_importance = None

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'good')

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'analysis')
    mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    imgs = get_imgs(artifact_dir, scale_importance,
                    dictionary_penalty, n_components_list)
    np.save(join(analysis_dir, 'imgs'), imgs)
    model = load_model(join(artifact_dir, 'model.keras'),
                       custom_objects={'HierarchicalLabelMasking':
                                           HierarchicalLabelMasking,
                                       'PartialSoftmax': PartialSoftmax})

    components_imgs = masker.inverse_transform(imgs)
    components_imgs.to_filename(join(analysis_dir, 'components.nii.gz'))

    X = load(join(artifact_dir, 'X.pkl'))
    y_pred = load(join(artifact_dir, 'y_pred.pkl'))
    y_pred = y_pred[1]
    labels = load(join(artifact_dir, 'labels.pkl'))
    y_pred = pd.DataFrame(y_pred, index=X.index)
    data = pd.concat([X, y_pred], names=['kind'], keys=['input', 'output'],
                     axis=1)
    data_sub = []
    min_samples = data.iloc[:, 0].groupby(level='dataset').aggregate(
        'count').min()
    random_state = check_random_state(0)
    for dataset, data_dataset in data.groupby(level='dataset'):
        indices = random_state.permutation(data_dataset.shape[0])[:min_samples]
        data_dataset = data_dataset.iloc[indices]
        data_sub.append(data_dataset)
    data = pd.concat(data_sub)
    X = data['input']
    y_pred = data['output']
    weights = model.get_layer('latent').get_weights()[0]
    X_proj = X.values.dot(weights)

    # CCA
    n_components = 10
    pca = PCA(n_components=n_components)
    cca = CCA(n_components=n_components)

    pca.fit(X_proj)
    cca.fit(X_proj, y_pred)

    principal_imgs = pca.components_
    principal_imgs = cca.x_loadings_.T


    weights_supervised, bias = model.get_layer(
        'supervised_depth_1').get_weights()
    freqs = principal_imgs.dot(weights_supervised)  # + bias[np.newaxis, :]
    freqs = np.exp(freqs)
    freqs /= np.sum(freqs, axis=1, keepdims=True)

    components_imgs = masker.inverse_transform(principal_imgs.dot(imgs))
    components_imgs.to_filename(join(analysis_dir, 'components_principal.'
                                                   'nii.gz'))
    data = components_imgs.get_data()
    vmax = np.max(np.abs(data))

    wc = WordCloud(width=800, height=400, relative_scaling=1,
                   background_color='white')
    fig, axes = plt.subplots(n_components, 2, figsize=(14, 40))
    labels = ['_'.join(label.split('_')[1:]) for label in labels]
    for i, ax in enumerate(axes):
        component = index_img(components_imgs, i)
        these_freqs = {label: freq for label, freq in zip(labels, freqs[i])}
        wc.generate_from_frequencies(these_freqs)
        ax[1].imshow(wc)
        ax[1].set_axis_off()
        plot_stat_map(component, figure=fig, axes=ax[0], threshold=1e-5,
                      colorbar=True, vmax=vmax)
    plt.savefig('principal.pdf')

    weights, bias = model.get_layer('supervised_depth_1').get_weights()
    vectors = weights.T.dot(imgs)
    components_imgs = masker.inverse_transform(vectors)
    components_imgs.to_filename(join(analysis_dir, 'vectors.'
                                                   'nii.gz'))
    data = components_imgs.get_data()
    vmax = np.max(np.abs(data))
    labels = ['_'.join(label.split('_')[1:]) for label in labels]
    fig, ax = plt.subplots(len(labels), 1)
    for i, label in enumerate(labels):
        print(label)
        plot_stat_map(index_img(components_imgs, i),
                      axes=ax[i], figure=fig, colorbar=True, vmax=vmax,
                      title=label)
    plt.savefig('vectors.pdf')

    # T-SNE
    X_proj = pd.DataFrame(data=X_proj, index=X.index)
    X_proj.sort_index(inplace=True)
    markers = ['o', 'x', '+', 'v']
    datasets = X_proj.index.get_level_values('dataset').unique().values
    markers = {dataset: marker for dataset, marker in zip(datasets, markers)}

    seed = random_state.randint(100000)
    Xt = bhtsne.run_bh_tsne(X_proj.values, initial_dims=10, verbose=True,
                            randseed=seed)
    Xt = pd.DataFrame(data=Xt, index=X.index)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.subplots_adjust(right=0.5)
    for dataset, Xt_dataset in Xt.groupby(level='dataset'):
        labels = Xt_dataset.index.get_level_values('contrast').unique().values
        colors = husl_palette(n_colors=len(labels))
        colors = {label: colors[i] for i, label in enumerate(labels)}
        for contrast, Xt_task in Xt_dataset.groupby(level='contrast'):
            ax.scatter(Xt_task.iloc[:, 0], Xt_task.iloc[:, 1],
                       marker=markers[dataset],
                       color=colors[contrast],
                       label='_'.join(contrast.split('_')[1:]))
    ax.legend(ncol=3, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.savefig('tsne.pdf')


if __name__ == '__main__':
    run()
