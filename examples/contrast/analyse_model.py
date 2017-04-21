from os.path import join, expanduser

from keras.models import load_model
from nilearn._utils import check_niimg
from wordcloud import WordCloud

from modl.datasets import get_data_dirs
from modl.hierarchical import HierarchicalLabelMasking, PartialSoftmax
from modl.classification import Reconstructer
from modl.input_data.fmri.unmask import retrieve_components
from modl.utils.system import get_cache_dirs
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, plot_prob_atlas
from sklearn.externals.joblib import Memory, load
import matplotlib.pyplot as plt

import numpy as np

def get_model(artifact_dir, scale_importance, dictionary_penalty,
              n_components_list):
    artifact_dir = expanduser('~/data/modl_data/pipeline/contrast/'
                              'prediction_hierarchical/good')
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
    imgs = reconstructer.fit_transform(weights)
    return masker, imgs, model


def run():
    dictionary_penalty = 1e-4
    n_components_list = [16, 64, 256]
    scale_importance = 'sqrt'

    artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                        'predict_hierarchical', 'good')

    masker, imgs, model = get_model(artifact_dir, scale_importance,
                                    dictionary_penalty, n_components_list)

    np.save('imgs', imgs)
    components_imgs = masker.inverse_transform(imgs)
    components_imgs.to_filename('components.nii.gz')

    imgs /= np.sqrt(np.sum(imgs ** 2, axis=1, keepdims=True))
    gram = imgs.dot(imgs.T)
    plt.imshow(gram)
    plt.colorbar()
    plt.savefig('gram.pdf')

    weights, bias = model.get_layer('supervised_depth_1').get_weights()

    labels = load(join(artifact_dir, 'labels.pkl'))

    # q, r = np.linalg.qr(imgs.T)
    # imgs = q.T
    # components_imgs = masker.inverse_transform(imgs)
    # components_imgs.to_filename('components_orth.nii.gz')
    # weight = r.dot(weights)

    #
    #
    components_imgs = check_niimg('components_orth.nii.gz')
    n_components = components_imgs.shape[3]
    data = components_imgs.get_data()
    vmax = np.max(np.abs(data))

    wc = WordCloud(width=800, height=400, relative_scaling=1,
                   background_color='white')
    fig, axes = plt.subplots(25, 2, figsize=(14, 40))
    labels = ['_'.join(label.split('_')[1:]) for label in labels]
    weights = np.abs(weights)
    for i, ax in enumerate(axes):
        component = index_img(components_imgs, i)
        freqs = weights[i]
        freqs = {label: freq for label, freq in zip(labels, freqs)}
        wc.generate_from_frequencies(freqs)
        ax[1].imshow(wc)
        ax[1].set_axis_off()
        plot_stat_map(component, figure=fig, axes=ax[0], threshold=2e-3,
                      colorbar=True, vmax=vmax)
    plt.savefig('figure.pdf')
