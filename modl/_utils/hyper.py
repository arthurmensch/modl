import os
from os.path import join, expanduser

import numpy as np
from skimage.io import imread
from joblib import Memory

import matplotlib.pyplot as plt

from math import ceil, sqrt


def fetch_data(data_dir=expanduser('~/data/complete_ms_data')):
    results = []
    for subdir in os.listdir(data_dir):
        current_dir = join(data_dir, subdir, subdir)
        name = subdir
        sp_img = []
        for img_file in os.listdir(current_dir):
            name, extension = os.path.splitext(img_file)
            if extension in ['.png', '.bmp']:
                img = imread(join(current_dir, img_file))
                if extension == '.png':
                    sp_img.append(img[..., np.newaxis] / (256 * 256))
                else:
                    rgb_img = img
        sp_img = np.concatenate(sp_img, 2)
        this_result = {'name': name,
                       'sp': sp_img,
                       'rgb': rgb_img}
        results.append(this_result)
    return results


def fetch_aviris():
    from spectral import open_image
    img = open_image(
        '/home/arthur/data/'
        'f100826t01p00r05rdn_b/f100826t01p00r05rdn_b_sc01_ort_img.hdr')
    img = img.open_memmap()
    img = np.array(img)
    img -= img.min()
    img = img / (256 * 256)
    return img


def display_sp_img(sp_img, rgb_img=None, name=None):
    if isinstance(sp_img, dict):
        rgb_img = sp_img['rgb']
        name = sp_img['name']
        sp_img = sp_img['sp']
    n_channels = sp_img.shape[2]
    m = ceil(sqrt(n_channels + 1))
    fig, axes = plt.subplots(m, m, figsize=(10, 10))
    axes = np.ravel(axes)
    if rgb_img is not None:
        axes[0].imshow(rgb_img)
        axes[0].axis('off')
        axes[0].set_title('RGB')
    for i, img in enumerate(np.rollaxis(sp_img, 2)):
        axes[i + 1].axis('off')
        axes[i + 1].imshow(img, cmap='gray')
        axes[i + 1].set_title('SP %i' % i)
    for j in range(i + 2, m ** 2):
        axes[j].axis('off')
    if name is not None:
        fig.suptitle(name)


if __name__ == '__main__':
    mem = Memory(cachedir=expanduser('~/cache'))
    img_list = mem.cache(fetch_data)()
    display_sp_img(img_list[0])
