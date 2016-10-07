from tempfile import NamedTemporaryFile, TemporaryFile

import atexit
from joblib import Memory
from modl.datasets import get_data_dirs
import os
from os.path import join, expanduser
from skimage.io import imread
from scipy.misc import face
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches_2d
from spectral import open_image

from joblib import dump, load

import numpy as np

from sacred.ingredient import Ingredient

data_ing = Ingredient('data')
patch_ing = Ingredient('patches', ingredients=[data_ing])

@patch_ing.config
def config():
    patch_size = (32, 32)
    max_patches = 10000
    test_size = 2000
    normalize_per_channel = False
    pickle = True


@data_ing.config
def config():
    source = 'aviris'
    scale = 1
    gray = False
    in_memory = True


@data_ing.capture
def load_data(source,
              scale,
              gray,
              in_memory,
              memory=Memory(cachedir=None)):
    data_dirs = get_data_dirs()
    data_dir = join(data_dirs[0], 'modl_data')
    if source == 'face':
        image = face(gray=gray)
        image = image / 255
        if scale != 1:
            image = memory.cache(rescale)(image, scale=scale)
        return image
    elif source == 'lisboa':
        image = imread(join(data_dir, 'images', 'lisboa.jpg'), as_grey=gray)
        image = image / 255
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
        if in_memory:
            image = np.array(image)
        else:
            f = NamedTemporaryFile()
            dump(image, f.name)
            image = load(f.name, mmap_mode='readwrite')

        image -= image.min()
        image = image / (256 * 256 - 1)
        return image
    else:
        raise ValueError('Data source is not known')


@patch_ing.capture
def make_patches(patch_size, max_patches, test_size,
                 normalize_per_channel,
                 pickle,
                 _run, _seed, data):
    if pickle and data['source'] == 'aviris':
        train_data, test_data = load(join(get_data_dirs()[0],
                                          'modl_data', 'aviris.pkl'),
                                     mmap_mode='r')
        test_data = np.array(test_data)
        _run.info['data_shape'] = (patch_size[0], patch_size[1], 224)
        train_data = train_data[:max_patches]
        return train_data, test_data

    img = load_data()
    if img.ndim == 3:
        height, width, n_channels = img.shape
        _run.info['data_shape'] = (patch_size[0], patch_size[1], n_channels)
    else:
        height, width = img.shape
        _run.info['data_shape'] = (patch_size[0], patch_size[1])
        img = img[:, :, np.newaxis]
    train_data = extract_patches_2d(img[:, :width // 2, :], patch_size,
                                    max_patches=max_patches,
                                    random_state=_seed)
    test_data = extract_patches_2d(img[:, width // 2:, :], patch_size,
                                   max_patches=test_size,
                                   random_state=_seed)
    if train_data.ndim == 3:
        train_data = train_data[..., np.newaxis]
        test_data = test_data[..., np.newaxis]
    if normalize_per_channel:
        train_mean = train_data.mean(axis=(1, 2))
        train_data -= train_mean[:, np.newaxis, np.newaxis, :]
        train_std = train_data.std(axis=(1, 2))
        train_std[train_std == 0] = 1
        train_data /= train_std[:, np.newaxis, np.newaxis, :]
        train_data = train_data.reshape((train_data.shape[0], -1))

        test_mean = test_data.mean(axis=(1, 2))
        test_data -= test_mean[:, np.newaxis, np.newaxis, :]
        test_std = test_data.std(axis=(1, 2))
        test_std[test_std == 0] = 1
        test_data /= test_std[:, np.newaxis, np.newaxis, :]
        test_data = test_data.reshape((test_data.shape[0], -1))
    else:
        train_data = train_data.reshape((train_data.shape[0], -1))
        test_data = test_data.reshape((test_data.shape[0], -1))
        train_mean = train_data.mean(axis=1)
        test_mean = test_data.mean(axis=1)
        train_data -= train_mean[:, np.newaxis]
        test_data -= test_mean[:, np.newaxis]

        train_std = train_data.std(axis=1)
        train_std[train_std == 0] = 1
        train_data /= train_std[:, np.newaxis]

        test_std = test_data.std(axis=1)
        test_std[test_std == 0] = 1
        test_data /= test_std[:, np.newaxis]

    return train_data, test_data