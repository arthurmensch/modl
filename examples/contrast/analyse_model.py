import os
from os.path import join

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from jinja2 import select_autoescape, PackageLoader, Environment
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from numpy.linalg import lstsq, svd
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory, load, Parallel, delayed
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax, \
    make_projection_matrix
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs


def plot_confusion_matrix(conf_arr, labels):
    fig = plt.figure(figsize=(20, 20))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    S = np.sum(conf_arr, axis=1, keepdims=True)
    S[S == 0] = 1
    conf_arr = conf_arr / S

    res = ax.imshow(conf_arr,
                    interpolation='nearest')

    width, height = conf_arr.shape

    plt.xticks(range(width))
    plt.yticks(range(height))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels, rotation=90)
    fig.colorbar(res)


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
                       overwrite=False):
    img = index_img(latent_imgs_nii, i)
    freqs = freqs[i]
    vmax = np.max(img.get_data())
    src = 'latent_%i.png' % i
    title = 'Latent component %i' % i
    target = join(assets_dir, src)
    if overwrite or not os.path.exists(target):
        fig = plt.figure(figsize=(24, 5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])
        for i, (dataset, sub_freqs) in enumerate(freqs.groupby(level='dataset')):
            ax = plt.subplot(gs[i + 1])
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.
                                                           inverted())
            width, height = bbox.width, bbox.height
            width *= fig.dpi
            height *= fig.dpi
            width = int(width)
            height = int(height)
            wc = WordCloud(width=width, height=height, relative_scaling=.5,
                           background_color='white', random_state=0)
            sub_freqs = sub_freqs.sort_values(ascending=False)
            # Remove labels below chance level
            lim = np.where(sub_freqs < 1 / sub_freqs.shape[0])[0][0]
            sub_freqs = sub_freqs.iloc[:lim]
            freq_dict = {label[2]: freq for label, freq in sub_freqs.iteritems()}
            title += str(freq_dict)
            wc.generate_from_frequencies(freq_dict)
            ax.imshow(wc)
            ax.set_axis_off()
        ax = plt.subplot(gs[0])
        plot_stat_map(img, figure=fig, axes=ax, threshold=vmax / 3,
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
    n_vectors = 96

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
    lbin = load(join(artifact_dir, 'lbin.pkl'))
    labels = lbin.classes_

    X = load(join(artifact_dir, 'X.pkl'))
    # y_pred = load(join(artifact_dir, 'y_pred_depth_1.pkl'))
    # y_pred = pd.DataFrame(y_pred, index=X.index)

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
        label_index.append(label.split('__'))
    label_index = pd.MultiIndex.from_tuples(
        label_index, names=('dataset', 'task', 'condition'))
    # Shape (200000, 336)

    print('Prediction')
    for i in range(3):
        prediction = pd.read_csv(
            join(artifact_dir, 'prediction_depth_%i.csv' % i))
        prediction = prediction.set_index(
            ['fold', 'dataset', 'subject', 'task', 'contrast', ])
        match = prediction['true_label'] == prediction['predicted_label']
        prediction = prediction.assign(match=match)
        prediction.sort_index(inplace=True)
        train_conf = confusion_matrix(prediction.loc['train', 'true_label'],
                                      prediction.loc[
                                          'train', 'predicted_label'],
                                      labels=labels)
        test_conf = confusion_matrix(prediction.loc['test', 'true_label'],
                                     prediction.loc['test', 'predicted_label'],
                                     labels=labels)
        plt.figure()
        plot_confusion_matrix(train_conf, labels)
        plt.savefig(join(analysis_dir, 'train_conf_depth_%i.png' % i))
        plot_confusion_matrix(test_conf, labels)
        plt.savefig(join(analysis_dir, 'test_conf_depth_%i.png' % i))
    #
    # print('Forward analysis')
    # proj = memory.cache(make_projection_matrix)(components,
    #                                             scale_bases=True,
    #                                             forward=True)
    # scaled_proj = proj / standard_scaler.scale_[np.newaxis, :]
    # latent_proj = scaled_proj.dot(weight_latent)
    # for depth in [1, 2]:
    #     classification_vectors = latent_proj.dot(weight_supervised[depth]).T
    #     classification_vectors = pd.DataFrame(data=classification_vectors,
    #                                           index=label_index)
    #     if depth == 1:
    #         mean = classification_vectors.groupby(level='dataset'). \
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
    # Backward
    weight_supervised = weight_supervised[1]  # Dataset level
    n_labels = weight_supervised.shape[1]
    weight_latent, r = np.linalg.qr(weight_latent)
    # weight_supervised = r.dot(weight_supervised)

    # ICA
    # # Manual sample weight
    sample_weight = X.iloc[:, 0].groupby(level=['dataset']).aggregate('count')
    max_sample_weight = sample_weight.max()
    repeat = max_sample_weight / sample_weight
    repeat = repeat.astype('int')
    new_X = []
    keys = []
    for dataset, sub_X in X.groupby(level=['dataset']):
        new_X += [sub_X] * repeat[dataset]
        keys.append(dataset)
    X = pd.concat(new_X, keys=keys, names=['dataset'])

    X = X.values
    X = standard_scaler.transform(X)
    X = X.dot(weight_latent)
    #
    #
    #
    print('ICA')
    fast_ica = FastICA(whiten=True, n_components=10, max_iter=10000)
    fast_ica.fit(X)
    mean_load = np.sqrt(np.sum(fast_ica.transform(X) ** 2, axis=0) / X.shape[0])
    VT = mean_load[:, np.newaxis] * fast_ica.mixing_.T + fast_ica.mean_
    VT_neg = - mean_load[:, np.newaxis] * fast_ica.mixing_.T + fast_ica.mean_
    VT = np.concatenate([VT, VT_neg], axis=0)

    # # PCA
    # Sample weight
    # sample_weight = X.iloc[:, 0].groupby(level=['dataset']).transform('count')
    # sample_weight = 1 / sample_weight
    # sample_weight /= sample_weight.min()
    # sample_weight = sample_weight.values
    #
    # X = X.values
    # X = standard_scaler.transform(X)
    # X = X.dot(weight_latent)
    #
    # # Sample weight PCA
    # mean = np.sum(X * sample_weight[:, np.newaxis], axis=0) / np.sum(sample_weight)
    # X -= mean[np.newaxis, :]
    # G = (X.T * sample_weight).dot(X) / (np.sum(sample_weight))
    # _, S, VT = svd(G)
    # VT = VT[:n_components]
    # S = S[:n_components]
    # VT = S[:, np.newaxis] * VT + mean
    # VT_neg = - S[:, np.newaxis] * VT + mean
    # VT = np.concatenate([VT, VT_neg])

    proj = memory.cache(make_projection_matrix)(components,
                                                scale_bases=True,
                                                forward=True)
    inv_proj = memory.cache(make_projection_matrix)(components,
                                                    scale_bases=True,
                                                    forward=False)
    latent_loadings = lstsq(weight_latent.T, VT.T)[0].T
    latent_loadings = standard_scaler.inverse_transform(latent_loadings)
    latent_imgs = latent_loadings.dot(inv_proj)
    model_input = latent_imgs.dot(proj)
    model_input = standard_scaler.transform(model_input)
    freqs = np.zeros((VT.shape[0], n_labels))

    count = pd.DataFrame(data=np.ones(label_index.shape[0]),
                         index=label_index).groupby(
        level='dataset').count().values[:, 0]
    idx = 0
    chance_level = np.ones(n_labels)
    for c in count:
        print(c)
        labels = np.ones((model_input.shape[0], 1)) * idx
        freqs += model.predict([model_input, labels])[1]
        chance_level[idx:idx + c] = 1 / c
        idx += c
    freqs = pd.DataFrame(data=freqs.T, index=label_index)
    # Save
    latent_imgs_nii = masker.inverse_transform(latent_imgs)
    latent_imgs_nii.to_filename(join(analysis_dir, 'latent_imgs.nii.gz'))

    print('Plotting')
    latent_array = Parallel(n_jobs=2, verbose=10)(
        delayed(plot_latent_vector)(
            latent_imgs_nii, freqs, assets_dir, i,
            overwrite=overwrite)
        for i in range(VT.shape[0]))
    html_file = template.render(classification_array=classification_array,
                                gram_array=gram_array,
                                latent_array=latent_array)
    with open(join(analysis_dir, 'result.html'), 'w+') as f:
        f.write(html_file)


if __name__ == '__main__':
    run(overwrite=True)
