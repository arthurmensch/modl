import os
from math import sqrt
from os.path import join

import matplotlib
from matplotlib.cm import get_cmap

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from jinja2 import select_autoescape, PackageLoader, Environment
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords, \
    plot_glass_brain
from numpy.linalg import lstsq
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals.joblib import Memory, load, Parallel, delayed
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax, \
    make_projection_matrix
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs
from keras.models import load_model

from PIL import ImageColor

idx = pd.IndexSlice


def plot_confusion_matrix(conf_arr, labels, normalize=True):
    fig = plt.figure(figsize=(20, 20))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if normalize:
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
        plot_stat_map(img,
                      axes=ax, figure=fig, colorbar=True,
                      threshold=vmax / 3,
                      title=label)
        fig.savefig(join(assets_dir, src))
        plt.close(fig)
    return {'src': src, 'label': label}


def plot_both_classification_vector(classification_vectors_nii, label_index,
                                    assets_dir, i, overwrite=False):
    src = 'classification_depth_%i.png' % (i)
    label = 'd: %s - t: %s - c: %s' % label_index.values[i]
    target = join(assets_dir, src)
    if overwrite or not os.path.exists(target):
        img = index_img(classification_vectors_nii['no_latent'], i)
        img_2 = index_img(classification_vectors_nii['latent_single'], i)
        img_3 = index_img(classification_vectors_nii['latent_multi'], i)
        vmax = np.max(img.get_data())
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        condition = label_index.values[i][2]
        if condition in ['computation', 'calculaudio', 'calculvideo', 'MATH']:
            cut_coords = (-28, -62, 50)
        elif condition in ['OBK_FACE', '2BK_FACE', 'face_control',
                           'face_trusty', 'face_sex', 'FACES']:
            cut_coords = (-38, -50, -22)  # FFA
        elif condition in ['triangle_random', 'triangle_intention']:
            cut_coords = (-54, -58, 6)
        else:
            cut_coords = find_xyz_cut_coords(img_3,
                                             activation_threshold=vmax / 2)
        plot_stat_map(img,
                      axes=axes[0], figure=fig, colorbar=True,
                      threshold=vmax / 5,
                      cut_coords=cut_coords,
                      title=label)
        vmax = np.max(img_2.get_data())
        plot_stat_map(img_2,
                      axes=axes[1], figure=fig, colorbar=True,
                      threshold=vmax / 5,
                      cut_coords=cut_coords,
                      title=label)
        vmax = np.max(img_3.get_data())
        plot_stat_map(img_3,
                      axes=axes[2], figure=fig, colorbar=True,
                      threshold=vmax / 5,
                      cut_coords=cut_coords,
                      title=label)
        fig.savefig(join(assets_dir, src))
        plt.close(fig)


def convert_tuple_to_rgb(x):
    r = int(x[0] * 256)
    g = int(x[1] * 256)
    b = int(x[2] * 256)
    return 'rgb(%i,%i,%o)' % (r, g, b)


def plot_latent_vector(latent_imgs_nii, freqs, assets_dir, i,
                       overwrite=False):
    img = index_img(latent_imgs_nii, i)
    freqs = freqs[i]
    vmax = np.max(img.get_data())
    src = 'latent_%i.png' % i
    title = 'Latent component %i' % i
    target = join(assets_dir, src)
    if overwrite or not os.path.exists(target):
        fig = plt.figure(figsize=(2.55 * 6, .6 * 6))
        fig.subplots_adjust(left=0, right=1, wspace=0, top=1, bottom=0)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax = plt.subplot(gs[1])
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.
                                                  inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        width = int(width)
        height = int(height)

        flatui = get_cmap('Vega20').colors[0::2]

        def color_func(word, font_size, position, orientation, **kwargs):
            if word in freqs.loc['hcp'].index.get_level_values(
                    'condition').unique().values:
                return convert_tuple_to_rgb(flatui[0])
            elif (word in freqs.loc['archi'].index.get_level_values(
                    'condition').unique().values or
                          word in ['R_buttonpress_audio_cue', 'L_buttonpress_audio_cue',
                                   'R_buttonpress_video_cue', 'L_buttonpress_video_cue',
                                   'audiocalculation', 'videocalculation',
                                   'V_checkerboard', 'H_checkerboard',
                                   'face_gender']):
                return convert_tuple_to_rgb(flatui[1])
            elif word in freqs.loc['camcan'].index.get_level_values(
                    'condition').unique().values or word in ['VideoOnly', 'AudioOnly']:
                return convert_tuple_to_rgb(flatui[2])
            elif word in freqs.loc['brainomics'].index.get_level_values(
                    'condition').unique().values:
                return convert_tuple_to_rgb(flatui[3])

        wc = WordCloud(width=width, height=height, relative_scaling=1,
                       prefer_horizontal=1,
                       background_color='white', random_state=0,
                       color_func=color_func)

        freq_dict = {}
        for i, (dataset, sub_freqs) in enumerate(
                freqs.groupby(level='dataset')):
            sub_freqs = sub_freqs.sort_values(ascending=False)
            # Remove labels below chance level
            lim = np.where(sub_freqs < 1 / sub_freqs.shape[0])[0][0]
            sub_freqs = sub_freqs.iloc[:lim]
            max_freq = sub_freqs.max()
            for label, freq in sub_freqs.iteritems():
                condition = label[2]
                if condition == 'clicDaudio':
                    condition = 'R_buttonpress_audio_cue'
                if condition == 'clicGaudio':
                    condition = 'L_buttonpress_audio_cue'
                if condition == 'clicDvideo':
                    condition = 'R_buttonpress_video_cue'
                if condition == 'clicGvideo':
                    condition = 'L_buttonpress_video_cue'
                if condition == 'calculaudio':
                    condition = 'audiocalculation'
                if condition == 'calculvideo':
                    condition = 'videocalculation'
                if condition == 'face_sex':
                    condition = 'face_gender'
                if condition == 'damier_V':
                    condition = 'V_checkerboard'
                if condition == 'damier_H':
                    condition = 'H_checkerboard'
                if condition == 'VidOnly':
                    condition = 'VideoOnly'
                if condition == 'AudOnly':
                    condition = 'AudioOnly'
                freq_dict[condition] = freq / max_freq / pow(len(condition), .05)
        title = str(freq_dict)
        wc.generate_from_frequencies(freq_dict)
        ax.imshow(wc)
        ax.set_axis_off()
        ax = plt.subplot(gs[0])
        plot_stat_map(img, figure=fig, axes=ax, threshold=vmax / 3,
                      colorbar=False)
        plt.savefig(target)
        plt.close(fig)
    return {'src': src, 'title': title, 'freq': repr(freq_dict)}


def compare():
    # Load model
    # y =X_full W_g D^-1 W_e W_s
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)
    unmask_dir = join(get_data_dirs()[0], 'pipeline', 'unmask', 'contrast')

    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    artifact_dir = {
        'latent_multi': join(get_data_dirs()[0], 'pipeline', 'contrast',
                             'prediction_hierarchical_introspect',
                             'latent'),
        'no_latent': join(get_data_dirs()[0], 'pipeline',
                          'contrast',
                          'prediction_hierarchical_introspect',
                          'no_latent_single'),
        'latent_single': join(get_data_dirs()[0], 'pipeline', 'contrast',
                              'prediction_hierarchical_introspect',
                              'latent_single')}

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'final', 'compare')
    assets_dir = join(analysis_dir, 'assets')

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()

    components = memory.cache(retrieve_components)(dictionary_penalty,
                                                   masker,
                                                   n_components_list)  # W_g

    proj, inv_proj, rec = memory.cache(make_projection_matrix)(components,
                                                               scale_bases=True, )

    print('Load model')
    classification_vectors_nii = {}
    for exp in ['no_latent', 'latent_single', 'latent_multi']:
        # Expected shape (?, 336)
        this_artifact_dir = artifact_dir[exp]
        label_index = []
        lbin = load(join(artifact_dir[exp], 'lbin.pkl'))
        labels = lbin.classes_
        for label in labels:
            label_index.append(label.split('__'))
        label_index = pd.MultiIndex.from_tuples(
            label_index, names=('dataset', 'task', 'condition'))
        model = load_model(join(this_artifact_dir, 'model.keras'),
                           custom_objects={'HierarchicalLabelMasking':
                                               HierarchicalLabelMasking,
                                           'PartialSoftmax': PartialSoftmax})
        if exp != 'no_latent':
            weight_latent = model.get_layer('latent').get_weights()[0]  # W_e
            latent_proj = proj.dot(weight_latent)
        else:
            latent_proj = proj
        # Shape (336, 50)
        weight_supervised, _ = model.get_layer(
            'supervised_depth_1').get_weights()  # W_s
        classification_vectors = latent_proj.dot(weight_supervised).T
        # gram = classification_vectors.T.dot(classification_vectors)
        # plot_confusion_matrix(gram, labels, normalize=False)
        # gram_src = 'gram_%s.png' % 'latent' if latent else 'no_latent'
        # plt.savefig(join(assets_dir, gram_src))

        classification_vectors = pd.DataFrame(data=classification_vectors,
                                              index=label_index)
        classification_vectors.sort_index(inplace=True)
        datasets = classification_vectors.index.get_level_values(
            'dataset').unique().values
        for dataset in datasets:
            mean = np.mean(classification_vectors.loc[dataset], axis=0).values
            classification_vectors.loc[dataset] = classification_vectors.loc[
                                                      dataset].values - mean
        # We can translate classification vectors
        classification_vectors_nii[exp] = masker.inverse_transform(
            classification_vectors.values)
        classification_vectors_nii[exp].to_filename(join(analysis_dir,
                                                         '%s.nii.gz' % exp))

    lbin = load(join(artifact_dir['latent_single'], 'lbin.pkl'))
    labels = lbin.classes_
    label_index = []
    for label in labels:
        label_index.append(label.split('__'))
    label_index = pd.MultiIndex.from_tuples(
        label_index, names=('dataset', 'task', 'condition'))
    for i, value in enumerate(label_index.values):
        print(i, value)
    # Display
    print('Plotting')
    Parallel(n_jobs=10, verbose=10)(
        delayed(plot_both_classification_vector)(
            classification_vectors_nii, label_index, assets_dir,
            i, overwrite=True)
        for i in range(30))


def run(overwrite=True):
    # Load model
    # y =X_full W_g D^-1 W_e W_s
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical_full', 'latent')

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical_full', 'analysis')
    assets_dir = join(analysis_dir, 'assets')

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    n_components = 200
    n_vectors = 76

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
    # standard_scaler = load(join(artifact_dir, 'standard_scaler.pkl'))
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

    # print('Prediction')
    # for i in range(3):
    #     prediction = pd.read_csv(
    #         join(artifact_dir, 'prediction_depth_%i.csv' % i))
    #     prediction = prediction.set_index(
    #         ['fold', 'dataset', 'subject', 'task', 'contrast', ])
    #     match = prediction['true_label'] == prediction['predicted_label']
    #     prediction = prediction.assign(match=match)
    #     prediction.sort_index(inplace=True)
    #     train_conf = confusion_matrix(prediction.loc['train', 'true_label'],
    #                                   prediction.loc[
    #                                       'train', 'predicted_label'],
    #                                   labels=labels)
    #     test_conf = confusion_matrix(prediction.loc['test', 'true_label'],
    #                                  prediction.loc['test', 'predicted_label'],
    #                                  labels=labels)
    #     plt.figure()
    #     plot_confusion_matrix(train_conf, labels)
    #     plt.savefig(join(analysis_dir, 'train_conf_depth_%i.png' % i))
    #     plot_confusion_matrix(test_conf, labels)
    #     plt.savefig(join(analysis_dir, 'test_conf_depth_%i.png' % i))
    #
    # print('Forward analysis')
    # proj, inv_proj, rec = memory.cache(make_projection_matrix)(components,
    #                                                            scale_bases=True,
    #                                                            )
    # if latent:
    #     latent_proj = proj.dot(weight_latent)
    # else:
    #     latent_proj = proj
    # for depth in [1]:
    #     classification_vectors = latent_proj.dot(weight_supervised[depth]).T
    #     classification_vectors = pd.DataFrame(data=classification_vectors,
    #                                           index=label_index)
    #     mean = classification_vectors.groupby(level='dataset').transform(
    #         'mean')
    #     # We can translate classification vectors
    #     # classification_vectors -= mean
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
    #     gram = classification_vectors.dot(classification_vectors.T).values
    #     np.save(join(analysis_dir, 'gram'), gram)
    #     fig = plt.figure()
    #     plot_confusion_matrix(gram, labels)
    #     gram_src = 'gram_depth_%i.png' % depth
    #     gram = {'src': gram_src, 'title': 'Gram at depth %i' % depth}
    #     gram_array.append(gram)
    #     fig.savefig(join(assets_dir, gram_src))
    #     plt.close(fig)
    # classification_array = classification_array[0]

    print('Backward analysis')
    # Backward
    weight_supervised = weight_supervised[1]  # Dataset level
    n_labels = weight_supervised.shape[1]
    weight_latent, r = np.linalg.qr(weight_latent)
    # weight_supervised = r.dot(weight_supervised)

    # ICA
    sample_weight = X.iloc[:, 0].groupby(level=['dataset']).aggregate('count')
    max_sample_weight = sample_weight.max()
    repeat = max_sample_weight / sample_weight
    repeat = repeat.astype('int')
    new_X = []
    for dataset, sub_X in X.groupby(level=['dataset']):
        new_X += [sub_X] * repeat[dataset]
    X = pd.concat(new_X)
    X = X.values
    X = X.dot(weight_latent)

    kmeans = MiniBatchKMeans(n_clusters=200, random_state=0)
    kmeans.fit(X)
    VT = kmeans.cluster_centers_[:n_components]

    proj, inv_proj, rec = make_projection_matrix(components, scale_bases=True)

    # Least square in dictionary span
    full_proj = rec.dot(proj).dot(weight_latent)
    latent_geometric = lstsq(full_proj.T, VT.T)[0].T
    latent_imgs = latent_geometric.dot(rec)
    model_input = latent_imgs.dot(proj)

    freqs = np.zeros((VT.shape[0], n_labels))

    count = pd.DataFrame(data=np.ones(label_index.shape[0]),
                         index=label_index).groupby(
        level='dataset').count().values[:, 0]
    idx = 0
    for c in count:
        print(c)
        labels = np.ones((model_input.shape[0], 1)) * idx
        freqs += model.predict([model_input, labels])[1]
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


def plot_gram_matrix():
    # Load model
    # y =X_full W_g D^-1 W_e W_s
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'None')

    analysis_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'prediction_hierarchical', 'analysis_latent')
    assets_dir = join(analysis_dir, 'assets')

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # HTML output

    print('Load model')
    # Expected shape (?, 336)
    lbin = load(join(artifact_dir, 'lbin.pkl'))
    labels = lbin.classes_
    mask = join(get_data_dirs()[0], 'mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=0).fit()
    components = memory.cache(retrieve_components)(dictionary_penalty, masker,
                                                   n_components_list)  # W_g

    model = load_model(join(artifact_dir, 'model.keras'),
                       custom_objects={'HierarchicalLabelMasking':
                                           HierarchicalLabelMasking,
                                       'PartialSoftmax': PartialSoftmax})
    weight_latent = model.get_layer('latent').get_weights()[0]  # W_e
    # Shape (336, 50)
    weight_supervised, _ = model.get_layer(
        'supervised_depth_1').get_weights()  # W_s

    proj, inv_proj, rec = memory.cache(make_projection_matrix)(components,
                                                               scale_bases=True,
                                                               )
    latent_proj = proj.dot(weight_latent)
    classification_vectors = latent_proj.dot(weight_supervised).T
    S = np.sqrt(np.sum(classification_vectors ** 2, axis=1, keepdims=True))
    classification_vectors /= S
    gram = classification_vectors.dot(classification_vectors.T)
    np.save(join(analysis_dir, 'gram'), gram)
    plot_confusion_matrix(gram, labels, normalize=False)
    gram_src = 'gram_depth_1.png'
    plt.savefig(join(assets_dir, gram_src))


if __name__ == '__main__':
    run()
