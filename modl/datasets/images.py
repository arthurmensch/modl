from modl.datasets import get_data_dirs
from os.path import join
from skimage.io import imread
from scipy.misc import face
from skimage.transform import rescale
from spectral import open_image

import numpy as np


def get_image(data_dir=None,
              source='aviris',
              scale=1,
              gray=False, in_memory=False):
    data_dirs = get_data_dirs(data_dir=data_dir)
    data_dir = data_dirs[0]
    if source == 'face':
        image = face(gray=gray)
        image = image / 255
        if scale != 1:
            image = rescale(image, scale=scale)
        return image
    elif source == 'lisboa':
        image = imread(join(data_dir, 'lisboa.jpg'))
        image = image / 255
        if scale != 1:
            image = rescale(image, scale=scale)
        return image
    elif source == ' aviris':
        image = open_image(
            join(data_dir,
                 'f100826t01p00r05rdn_b/'
                 'f100826t01p00r05rdn_b_sc01_ort_img.hdr'))
        image = image.open_memmap()
        if in_memory:
            image = np.array(image)
        image -= image.min()
        image = image / (256 * 256 - 1)
        return image
    else:
        raise ValueError('Data source is not known')