"""
social sparsity: approximate overlapping group lasso

"""
# Author: Elvis Dohmatob, Gael Varoquaux
# License: simplified BSD

from math import sqrt, log
from numbers import Number
import numpy as np
from scipy import ndimage


def _fwhm2sigma(fwhm, voxel_size):
    """Convert a FWHM value to sigma in a Gaussian kernel.
    """
    fwhm = np.asanyarray(fwhm)
    return fwhm / (sqrt(8. * log(2.)) * voxel_size)


def _prox_l21(img, alpha, grp_norms_squared=None):
    # To avoid side effects, assign the raw img values on the side
    if grp_norms_squared is None:
        grp_norms_squared = img ** 2
    grp_norms = np.sqrt(grp_norms_squared, out=grp_norms_squared)
    shrink = np.zeros(img.shape)
    img_nz = (grp_norms > 1e-10)
    shrink[img_nz] = (1 - alpha / grp_norms[img_nz]).clip(0)
    return img * shrink


def _grp_norms_squared(img, fwhm, affine=None, voxel_size=None,
                       side_weights=.7, kernel="gaussian", mode="constant",
                       cval=0.):
    """Social sparsity as defined by eq 4 of Kowalski et al, 'Social Sparsity...'

    Parameters
    ----------
    side_weight: nonnegative float, optional (default 0.7)
        Weights of sides of neigborhood relative to center. A value of 1
        corresponds to the classical overlapping group-Lasso shrinkage
        operator.

    fwhm: int, optional (default 1)
        Size of neigbourhoods to consider, measured in mm's. This is a
        field-of-view (FOV) parameter and plays a rule similar to the fwhm in
        Gaussian kernels. The larger the radius, the smoother the prox.

    """
    if isinstance(fwhm, Number):
        fwhm = (fwhm,) * img.ndim
    fwhm = np.asanyarray(fwhm)
    if voxel_size is None:
        if affine is None:
            raise ValueError(
                "voxel_size not provided, you must specify the affine")
        affine = affine[:3, :3]
        voxel_size = np.sqrt(np.sum(affine ** 2, axis=0))
    sigma = _fwhm2sigma(fwhm, voxel_size)
    grp_norms_squared = img ** 2

    if kernel == "gaussian":
        for axis, s in enumerate(sigma):
            ndimage.gaussian_filter1d(grp_norms_squared, s, axis=axis,
                                      output=grp_norms_squared,
                                      mode=mode, cval=cval)
    elif kernel == "pyramid":
        radius = np.floor(2 * sigma).astype(np.int)  # the "blur radius"
        diameter = 2 * radius + 1
        if side_weights == 1.:
            # use scipy's optimized rectangular uniform filter
            ndimage.uniform_filter(grp_norms_squared, size=diameter,
                                   output=grp_norms_squared, mode=mode,
                                   cval=cval)
        else:
            social_filter = np.full(diameter, 1.)
            social_filter *= side_weights

            # adjust weight at center of filter
            if img.ndim == 1:
                social_filter[radius] = 1.
            elif img.ndim == 2:
                social_filter[radius[0], radius[1]] = 1.
            elif img.ndim == 3:
                social_filter[radius[0], radius[1], radius[2]] = 1.
            else:
                raise RuntimeError("WTF! img.ndim is %i." % img.ndim)

            # normalize filter weights to sum to 1
            social_filter /= social_filter.sum()

            # the actual convolution
            ndimage.filters.convolve(grp_norms_squared, social_filter,
                                     output=grp_norms_squared,
                                     mode="constant")
    else:
        raise ValueError("Unknown kernel: %s" % kernel)
    # else:
    #     # use ninja code from @gael
    #     grp_norms_squared = _neighboorhood_norms_squared(
    #         img, side_weights=side_weights)
    return grp_norms_squared


def _prox_social_sparsity(img, alpha, fwhm, affine=None, voxel_size=None,
                          side_weights=.7, kernel="gaussian", mode="constant",
                          cval=0.):
    """Social sparsity as defined by eq 4 of Kowalski et al, 'Social Sparsity...'

    Parameters
    ----------
    side_weight: nonnegative float, optional (default 0.7)
        Weights of sides of neigborhood relative to center. A value of 1
        corresponds to the classical overlapping group-Lasso shrinkage
        operator.

    fwhm: int, optional (default 1)
        Size of neigbourhoods to consider, measured in mm's. This is a
        field-of-view (FOV) parameter and plays a rule similar to the fwhm in
        Gaussian kernels. The larger the radius, the smoother the prox.

    """
    grp_norms_squared = _grp_norms_squared(img, fwhm, affine=affine,
                                           voxel_size=voxel_size,
                                           side_weights=side_weights,
                                           kernel=kernel,
                                           mode=mode, cval=cval)
    return _prox_l21(img, alpha, grp_norms_squared=grp_norms_squared)


def _social_sparsity_alpha_grid(grad_loss, mask, fwhm, affine=None,
                                n_alphas=10, kernel="gaussian",
                                voxel_size=None, side_weights=.7,
                                mode="constant", cval=0., eps=1e-3):
    """Computes grid of regularization parameters for social sparsity.

    Parameters
    ----------
    grad_loss: ndarray, shape (n_targets, n_features)
        Gradient of loss function at zero.

    mask: ndarray, shape (d1, d2, ...,)
        Contains n_features non-zero values.
    """
    imgs = np.zeros((len(grad_loss),) + mask.shape, dtype=grad_loss.dtype)
    imgs[:, mask] = grad_loss
    grp_norms_squared = [_grp_norms_squared(img, fwhm, affine=affine,
                                            voxel_size=voxel_size,
                                            side_weights=side_weights,
                                            kernel=kernel, mode=mode,
                                            cval=cval)
                         for img in imgs]
    alpha_max = np.sqrt(np.max(grp_norms_squared))

    if n_alphas == 1:
        return np.array([alpha_max])
    alpha_min = alpha_max * eps
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                       num=n_alphas)[::-1]
