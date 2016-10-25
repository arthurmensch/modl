from joblib import Memory
from modl.datasets import get_data_dirs
from os.path import join

from scipy.misc import face
from skimage.io import imread
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state
from spectral import open_image

import numpy as np


def load_images(source,
              scale=1,
              gray=False,
              memory=Memory(cachedir=None)):
    data_dir = get_data_dirs()[0]
    if source == 'face':
        image = face(gray=gray)
        image = image / 255
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if scale != 1:
            image = memory.cache(rescale)(image, scale=scale)
        return image
    elif source == 'lisboa':
        image = imread(join(data_dir, 'images', 'lisboa.jpg'), as_grey=gray)
        image = image / 255
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if scale != 1:
            image = memory.cache(rescale)(image, scale=scale)
        return image
    elif source == 'aviris':
        image = open_image(
            join(data_dir,
                 'aviris',
                 'f100826t01p00r05rdn_b/'
                 'f100826t01p00r05rdn_b_sc01_ort_img.hdr'))
        image = image.open_memmap()
        return image
    else:
        raise ValueError('Data source is not known')


def get_num_patches(image, patch_shape=(8,)):
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, )
    if len(patch_shape) == 1:
        patch_shape = (patch_shape, patch_shape)
    if len(patch_shape) == 2:
        n_channel = image.shape[2]
        patch_shape = (patch_shape[0], patch_shape[1], n_channel)
    patches = extract_patches(image, patch_shape=patch_shape)

    return patches.shape[0] * patches.shape[1] * patches.shape[2]


def gen_patch_batches(image,
                      patch_shape=(8,),
                      batch_size=10,
                      random_state=None):
    if len(patch_shape) == 1:
        patch_shape = (patch_shape, patch_shape)
    if len(patch_shape) == 2:
        n_channel = image.shape[2]
        patch_shape = (patch_shape[0], patch_shape[1], n_channel)
    patches = extract_patches(image, patch_shape=patch_shape)
    random_state = check_random_state(random_state)
    n_samples = patches.shape[0] * patches.shape[1] * patches.shape[2]
    permutation = random_state.permutation(n_samples).astype('i4')
    patch_indices = list(np.ndindex(patches.shape[:3]))
    batch = np.zeros((batch_size, *patch_shape))
    indices = np.zeros(batch_size, dtype='i4')
    ii = 0
    for i in permutation:
        patch = patches[patch_indices[i]]
        if np.any(patch == -50):
            continue
        batch[ii % batch_size] = patch
        if image.dtype == np.dtype('>i2'):
            batch[ii % batch_size] /= 65535
        indices[ii % batch_size] = i
        ii += 1
        if not ii % batch_size:
            if image.dtype == np.dtype('>i2'):
                batch /= 65535
            yield np.array(batch), np.array(indices)
