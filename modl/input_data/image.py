import numpy as np
from math import sqrt

def scale_patches(X, with_mean=True, with_std=True, channel_wise=True, copy=True):
    if copy:
        X = X.copy()
    if with_mean:
        if channel_wise:
            X -= np.mean(X, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        else:
            X -= np.mean(X, axis=(1, 2, 3))[:,
                 np.newaxis, np.newaxis, np.newaxis]
    if with_std:
        if channel_wise:
            n_channel = X.shape[3]
            std = np.sqrt(np.sum(X ** 2, axis=(1, 2)))
            std[std == 0] = 1
            X /= std[:, np.newaxis, np.newaxis, :] * sqrt(n_channel)
        else:
            std = np.sqrt(np.sum(X ** 2, axis=(1, 2, 3)))
            std[std == 0] = 1
            X /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return X

from .image_fast import clean_mask
from .image_fast import fill