import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import select_autoescape, PackageLoader, Environment
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from numpy.linalg import lstsq, svd
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory, load, Parallel, delayed
from wordcloud import WordCloud

from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax, \
    make_projection_matrix
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs


def plot_classification_vector(classification_vectors_nii, label_index,
                               assets_dir, depth, i, overwrite=False):
    src = 'classification_depth_%i_%i.png' % (depth, i)
    label = 'd: %s - t: %s - c: %s' % label_index.values[i]
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


def plot_latent_vector(latent_imgs_nii, freqs, assets_dir, i,
                       negative=False,
                       overwrite=False):
    img = index_img(latent_imgs_nii, i)
    freqs = freqs[i]
    freqs = freqs.sort_values(ascending=False)
    lim = np.where(freqs < 1e-2)[0]
    if len(lim) == 0:
        lim = - 1
    else:
        lim = lim[0]
    freqs = freqs.iloc[:lim]
    print(i, freqs.iloc[0])
    vmax = np.max(img.get_data())
    label_index = freqs.index
    tasks = label_index.get_level_values('task')
    conditions = label_index.get_level_values('condition')
    labels = ['-'.join([task, condition])
              for task, condition in zip(tasks, conditions)]
    freq_dict = {label: freq for label, freq in zip(labels, freqs.values)}
    src = 'latent_%i_%s.png' % (i, negative)
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
        wc = WordCloud(width=width, height=height, relative_scaling=.5,
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
                        'prediction_hierarchical', 'None')

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'analysis')
    assets_dir = join(analysis_dir, 'assets')

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    n_components = 50
    n_vectors = 97

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
    bias_supervised = {}
    for depth in [1, 2]:
        weight_supervised[depth], bias_supervised[depth] = model.get_layer(
            'supervised_depth_%i' % depth).get_weights()  # W_s
    # Shape (50, 97)

    # Latent factors
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)  # W_g

    # bases = np.load(join(analysis_dir, 'bases.npy'))
    label_index = []
    for label in labels:
        split = label.split('_')
        label_index.append((split[0], split[1], '_'.join(split[2:])))
    label_index = pd.MultiIndex.from_tuples(
        label_index, names=('dataset', 'task', 'condition'))
    # Shape (200000, 336)

    print('Forward analysis')
    # proj = memory.cache(make_projection_matrix)(components,
    #                                             scale_bases=True,
    #                                             inverse=True)
    # scaled_proj = proj / standard_scaler.scale_[np.newaxis, :]
    # latent_proj = scaled_proj.dot(weight_latent)
    # for depth in [1, 2]:
    #     classification_vectors = latent_proj.dot(weight_supervised[depth]).T
    #     classification_vectors = pd.DataFrame(data=classification_vectors,
    #                                           index=label_index)
    #     if depth == 1:
    #         mean = classification_vectors.groupby(level='dataset').\
    #             transform('mean')
    #     else:
    #         mean = classification_vectors.groupby(
    #             level=['dataset', 'task']).transform('mean')
    #     # We can translate classification vectors
    #     classification_vectors -= mean
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
    #                 classification_vectors_nii, label_index, assets_dir,
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
    le_dict = load(join(artifact_dir, 'le_dict.pkl'))
    i = le_dict['dataset'].transform(['archi'])[0]
    print(i)
    # Backward
    weight_supervised = weight_supervised[1]
    weight_latent, r = np.linalg.qr(weight_latent)
    weight_supervised = r.dot(weight_supervised)

    sample_weight = X.iloc[:, 0].groupby(level='dataset').transform('count')
    sample_weight = 1 / sample_weight
    sample_weight /= sample_weight.min()
    sample_weight = sample_weight.values

    # Manual sample weight
    sample_weight = X.iloc[:, 0].groupby(level='dataset').aggregate('count')
    max_sample_weight = sample_weight.max()
    repeat = max_sample_weight / sample_weight
    repeat = repeat.astype('int')
    new_X = []
    keys = []
    for dataset, sub_X in X.groupby(level='dataset'):
        new_X += [sub_X] * repeat[dataset]
        keys.append(dataset)
    X = pd.concat(new_X, keys=keys, names=['dataset'])

    X = X.values
    X = standard_scaler.transform(X)
    X = X.dot(weight_latent)

    # ICA
    print('ICA')
    fast_ica = FastICA(whiten=True, n_components=10, max_iter=10000)
    fast_ica.fit(X)
    V = fast_ica.components_

    # Sample weight PCA
    # mean = np.sum(X * sample_weight[:, np.newaxis], axis=0) / np.sum(sample_weight)
    # X -= mean[np.newaxis, :]
    # G = (X.T * sample_weight).dot(X) / (np.sum(sample_weight))
    # _, S, V = svd(G)

    latent_loadings = lstsq(weight_latent.T, V.T)[0].T
    latent_loadings *= standard_scaler.scale_
    bases = memory.cache(make_projection_matrix)(components, scale_bases=True,
                                                 inverse=False)
    proj = memory.cache(make_projection_matrix)(components, scale_bases=True,
                                                inverse=True)
    latent_imgs = latent_loadings.dot(bases)
    latent_array = {}
    for negative in [False]:
        if negative:
            latent_imgs *= -1
            V *= -1
        input = latent_imgs.dot(proj)
        input = standard_scaler.transform(input)
        labels = np.zeros((input.shape[0], 3))
        labels[:, 0] = i
        freqs = model.predict([input, labels])[1]
        freqs = pd.DataFrame(data=freqs.T, index=label_index)
        # Save
        latent_imgs_nii = masker.inverse_transform(latent_imgs)
        latent_imgs_nii.to_filename(join(analysis_dir, 'latent_imgs_%s.nii.gz'
                                         % negative))

        print('Plotting')
        latent_array[negative] = Parallel(n_jobs=2, verbose=10)(
            delayed(plot_latent_vector)(
                latent_imgs_nii, freqs, assets_dir, i,
                negative=negative,
                overwrite=overwrite)
            for i in range(10))
    # latent_array = list(zip(*[latent_array[True], latent_array[False]]))
    latent_array = latent_array[False]
    html_file = template.render(classification_array=classification_array,
                                gram_array=gram_array,
                                latent_array=latent_array)
    with open(join(analysis_dir, 'result.html'), 'w+') as f:
        f.write(html_file)


if __name__ == '__main__':
    run(overwrite=True)
