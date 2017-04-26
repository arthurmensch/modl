import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import select_autoescape, PackageLoader, Environment
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from numpy.linalg import lstsq, pinv
from sklearn.externals.joblib import Memory, load, Parallel, delayed
from wordcloud import WordCloud

from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs


def plot_classification_vector(classification_vectors_nii, labels_clean,
                               assets_dir, depth, i, overwrite=False):
    src = 'classification_depth_%i_%i.png' % (depth, i)
    label = labels_clean[i]
    target = join(assets_dir, src)
    if overwrite or not os.path.exists(target):
        img = index_img(classification_vectors_nii, i)
        vmax = np.max(img.get_data())
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_stat_map(index_img(classification_vectors_nii, i),
                      axes=ax, figure=fig, colorbar=True,
                      threshold=vmax / 3,
                      title=label)
        fig.savefig(join(assets_dir, src))
        plt.close(fig)
    return {'src': src, 'label': label}


def plot_latent_vector(latent_imgs_nii, labels_clean, freqs, assets_dir, i,
                       overwrite=False):
    img = index_img(latent_imgs_nii, i)
    vmax = np.max(img.get_data())
    freq_dict = {label: freq for label, freq in zip(labels_clean, freqs[i])}
    src = 'latent_%i.png' % i
    title = 'Latent component %i' % i
    target = join(assets_dir, src)
    if overwrite or not os.path.exists(target):
        fig, axes = plt.subplots(1, 2,
                                 figsize=(16, 5))
        bbox = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.
                                                       inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        width = int(width)
        height = int(height)
        wc = WordCloud(width=width, height=height, relative_scaling=1,
                       background_color='white')
        wc.generate_from_frequencies(freq_dict)
        axes[1].imshow(wc)
        axes[1].set_axis_off()
        plot_stat_map(img, figure=fig, axes=axes[0], threshold=vmax / 3,
                      colorbar=True)
        plt.savefig(target)
        plt.close(fig)
    return {'src': src, 'title': title, 'freq': repr(freq_dict)}


def run(overwrite=False):
    # Load model
    # y =X_full W_g D^-1 W_e W_s
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'good')

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'analysis')
    assets_dir = join(analysis_dir, 'assets')

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    n_components = 10
    n_vectors = 3

    # HTML output
    env = Environment(loader=PackageLoader('modl', 'templates'),
                      autoescape=select_autoescape(['html', 'xml']))
    template = env.get_template('analyse_model.html')
    latent_array = []
    classification_array = []
    gram_array = []

    print('Load model')
    mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()
    standard_scaler = load(join(artifact_dir, 'standard_scaler.pkl'))  # D
    # Expected shape (?, 336)
    labels = load(join(artifact_dir, 'labels.pkl'))

    X = load(join(artifact_dir, 'X.pkl'))
    y_pred = load(join(artifact_dir, 'y_pred_depth_1.pkl'))
    y_pred = pd.DataFrame(y_pred, index=X.index)

    from keras.models import load_model

    model = load_model(join(artifact_dir, 'model.keras'),
                       custom_objects={'HierarchicalLabelMasking':
                                           HierarchicalLabelMasking,
                                       'PartialSoftmax': PartialSoftmax})

    weight_latent = model.get_layer('latent').get_weights()[0]  # W_e
    # Shape (336, 50)

    weight_supervised = {}
    for depth in [1, 2]:
        weight_supervised[depth], _ = model.get_layer(
            'supervised_depth_%i' % depth).get_weights()  # W_s
    # Shape (50, 97)

    # Latent factors
    bases = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                              n_components_list)  # W_g
    for i, basis in enumerate(bases):
        S = np.std(basis, axis=1)
        S[S == 0] = 0
        basis = basis / S[:, np.newaxis]
        bases[i] = basis
    bases = np.concatenate(bases, axis=0).T

    labels_clean = ['_'.join(label.split('_')[1:]) for label in labels]
    # Shape (200000, 336)

    print('Forward analysis')
    # Forward
    # for depth in [1, 2]:
    #     classification_vectors = bases.dot(
    #         standard_scaler.inverse_transform(
    #             weight_latent.dot(weight_supervised[depth]).T).T).T
    #     np.save(join(analysis_dir, 'classification'),
    #             classification_vectors)
    #     np.save(join(analysis_dir, 'weight_supervised_depth_%i' % depth),
    #             weight_supervised[depth])
    #     classification_vectors_nii = masker.inverse_transform(
    #         classification_vectors)
    #     classification_vectors_nii.to_filename(
    #         join(analysis_dir,
    #              'classification_vectors_depth_%i.nii.gz' % depth))
    #     # Display
    #     print('Plotting')
    #     this_classification_array = \
    #         Parallel(n_jobs=2, verbose=10)(
    #             delayed(plot_classification_vector)(
    #                 classification_vectors_nii, labels_clean, assets_dir,
    #                 depth, i, overwrite=overwrite)
    #             for i in range(n_vectors))
    #     classification_array.append(this_classification_array)
    #
    #     gram = classification_vectors.dot(classification_vectors.T)
    #     np.save(join(analysis_dir, 'gram'), gram)
    #     fig, ax = plt.subplots(1, 1)
    #     cax = ax.imshow(gram)
    #     fig.colorbar(cax)
    #     gram_src = 'gram_depth_%i.png' % depth
    #     gram = {'src': gram_src, 'title': 'Gram at depth %i' % depth}
    #     gram_array.append(gram)
    #     fig.savefig(join(assets_dir, gram_src))
    #     plt.close(fig)
    # classification_array = list(zip(*classification_array))

    print('Backward analysis')
    # Backward
    weight_latent, r = np.linalg.qr(weight_latent)
    weight_supervised = weight_supervised[1]
    weight_supervised = r.dot(weight_supervised)

    latent_loadings = pinv(weight_latent.T)
    latent_loadings = standard_scaler.transform(latent_loadings.T).T
    latent_imgs = lstsq(bases.T, latent_loadings)[0].T

    freqs = weight_supervised
    freqs = np.exp(freqs)
    freqs /= np.sum(freqs, axis=1, keepdims=True)

    # Save
    latent_imgs_nii = masker.inverse_transform(latent_imgs)
    latent_imgs_nii.to_filename(join(analysis_dir, 'latent_imgs.nii.gz'))

    print('Plotting')
    latent_array = Parallel(n_jobs=2, verbose=10)(
        delayed(plot_latent_vector)(
            latent_imgs_nii, labels_clean, freqs, assets_dir, i,
            overwrite=overwrite)
        for i in range(n_components))

    html_file = template.render(classification_array=classification_array,
                                gram_array=gram_array,
                                latent_array=latent_array)
    with open(join(analysis_dir, 'result.html'), 'w+') as f:
        f.write(html_file)


if __name__ == '__main__':
    run(overwrite=False)
