from math import ceil, sqrt
from os.path import join, expanduser

import matplotlib.pyplot as plt
import numpy as np
from spectral import open_image


def fetch_aviris(datadir=expanduser('~/data/aviris')):
    img = open_image(
        join(datadir,
             'f100826t01p00r05rdn_b/f100826t01p00r05rdn_b_sc01_ort_img.hdr'))
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
