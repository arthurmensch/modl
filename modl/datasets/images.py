from joblib import Memory
from modl.datasets import get_data_dirs
from os.path import join

from scipy.misc import face
from skimage.io import imread
from skimage.transform import rescale
from spectral import open_image

import numpy as np


def load_images(source,
                scale=1,
                gray=False,
                normalize=False,
                center=False,
                memory=Memory(cachedir=None)):
    data_dir = get_data_dirs()[0]
    if source == 'face':
        image = face(gray=gray)
        image = image / 255
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if scale != 1:
            image = memory.cache(rescale)(image, scale=scale)
        if center:
            image -= image.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]
        if normalize:
            std = image.std(axis=(0, 1))
            std[std == 0] = 1
            image /= std[np.newaxis, np.newaxis, :]
        return image
    elif source == 'lisboa':
        image = imread(join(data_dir, 'images', 'lisboa.jpg'), as_grey=gray)
        image = image / 255
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if scale != 1:
            image = memory.cache(rescale)(image, scale=scale)
        if center:
            image -= image.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]
        if normalize:
            std = image.std(axis=(0, 1))
            std[std == 0] = 1
            image /= std[np.newaxis, np.newaxis, :]
        return image
    elif source == 'aviris':
        image = open_image(
            join(data_dir,
                 'aviris',
                 'f100826t01p00r05rdn_b/'
                 'f100826t01p00r05rdn_b_sc01_ort_img.hdr'))
        image = np.array(image.open_memmap(), dtype=float)
        good_bands = list(range(image.shape[2]))
        good_bands.remove(110)
        image = image[:, :, good_bands]
        indices = image == -50
        image[indices] = -1
        image[~indices] -= np.min(image[~indices])
        image[~indices] /= np.max(image[~indices])
        if center:
            image -= image.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]
        if normalize:
            std = image.std(axis=(0, 1))
            std[std == 0] = 1
            image /= std[np.newaxis, np.newaxis, :]
        return image
    else:
        raise ValueError('Data source is not known')