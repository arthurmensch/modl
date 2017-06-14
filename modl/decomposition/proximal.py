from math import sqrt
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_almost_equal)
from nilearn.decoding.objective_functions import _gradient, _div, _unmask
from nilearn.decoding.fista import mfista
from nilearn.decoding.proximal_operators import _prox_l1, _prox_tvl1
from ..utils.math.enet import enet_projection


def _gram_schmidt(V, offset=0, normalize=True):
    """Computes modified Gram-Schmidt (MGS) orthogonalization of a set
    of vectors specified as rows of a 2D array V.

    Parameters
    ----------
    offset: int, optional (default 0)
    assumes that all vectors from index 0 to offset - 1 (inclusive) have
    already been processed.

    Returns
    -------
    V[offset:]
    """
    scales = {}
    for i in range(offset, len(V)):
        for j in range(i):
            if j not in scales:
                scales[j] = V[j].dot(V[j])
            if scales[j] > 0.:
                weight = V[j].dot(V[i]) / scales[j]
                V[i] -= weight * V[j]
        if normalize:
            scales[i] = V[i].dot(V[i])
            if scales[i] > 0.:
                V[i] /= sqrt(scales[i])
    return V[offset:]


def test_gram_schmidt():

    V = np.array([[3., 1.], [2., 2.]])
    assert_array_almost_equal(_gram_schmidt(V, normalize=False),
                              np.array([[3, 1], [-.4, 1.2]]))

    V = np.array([[1., 1., 1., 1.], [-1., 4., 4., 1.], [4., -2., 2., 0]])
    assert_array_almost_equal(_gram_schmidt(V, normalize=False),
                              np.array([[1., 1., 1., 1.], [-3., 2., 2., -1.],
                                        [1., -10. / 6., 7. / 3, -10. / 6.]]))

    V = np.array([[1., 1., 1., 1.], [-1., 4., 4., 1.], [4., -2., 2., 0]])
    V_ = V.copy()
    _gram_schmidt(V, offset=2, normalize=False)
    assert_array_equal(V[:2], V_[:2])


def _atomic_prox(atom, weight, output=None, mask=None, pos=True,
                 atom_init=None, which="social", l1_ratio=1., norm=None,
                 max_iter=100, check_lipschitz=False, check_grad=False,
                 tol=1e-3, verbose=2, idx=None, **kwargs):
    """
    Solves for SSODL (Smooth Sparse Dictionary Learning) dictionary update.

    \argmin_{V} \frac{1}{2} .5/n * tr(VV^TA) - (1/n)tr(VB^T) +
                n * alpha \sum_j \varphi(v^j),

    where \varphi is a regularizer that imposes structure: sparsity and
    smoothness like GraphNet, TVL1, or social sparsity.

    References
    ==========
    [1] Dohmatob et al. "Learning brain regions via large-scale online
        structured sparse dictionary-learning", NIPS 2017
    [2] Varoquaux et al. "Social-sparsity brain decoders: faster spatial
        sparsity", PRNI 2016.
    [3] Kowalski et al. "Social sparsity! neighborhood systems enrich
        structured shrinkage operators", Transactions on Signal Processing
    """
    if verbose and idx is not None:
        msg = "[SODL (%s)] updating component %02i" % (which, idx)
        msg += "+" * (80 - len(msg))
        print(msg)

    n_voxels = len(atom)
    if which == "enet":
        if norm is None:
            raise ValueError
        if pos:
            atom[atom < 0.] = 0.
        enet_projection(atom, output, norm, l1_ratio)
        atom = output
    elif which == "social":
        try:
            from .social_sparsity import _prox_social_sparsity
        except ImportError:
            raise RuntimeError("Can't do social sparsity (module not found)")
        atom = _prox_social_sparsity(_unmask(atom, mask), weight,
                                     **kwargs)[mask]
    elif which == "gram-schmidt":
        raise NotImplementedError
        # dictionary[k] = _gram_schmidt(dictionary[:k + 1], offset=k)[-1]
    elif which == "enet variational":
        l1_weight = weight * l1_ratio
        l2_weight = weight - l1_weight
        scale = sqrt(1. + l2_weight)
        atom /= scale
        if l1_weight > 0.:
            atom = _prox_l1(atom, l1_weight)
        atom /= scale
    elif which in ["tv-l1", "graph-net"]:
        # misc
        flat_mask = mask.ravel()
        l1_weight = weight * l1_ratio
        if which == "graph-net":
            spatial_weight = weight - l1_weight
        else:
            spatial_weight = 0.
        lap_lips = 4. * mask.ndim * spatial_weight
        loss_lips = 1.
        loss_lips *= 1.05
        lips = loss_lips + lap_lips

        def smooth_energy(v):
            """Smooth part of energy / cost function.
            """
            e = .5 * np.sum((v - atom) ** 2)
            if which == "graph-net":
                lap = np.sum(_gradient(_unmask(v, mask))[:, mask] ** 2)
                lap *= spatial_weight
                lap *= .5
                e += lap
            return e

        def nonsmooth_energy(v):
            """Non-smooth part of energy / cost function.
            """
            e = l1_weight * np.abs(v).sum()
            if which == "tv-l1":
                gradient = _gradient(_unmask(v, mask))
                tv_term = np.sum(np.sqrt(np.sum(gradient ** 2,
                                                axis=0)))
                tv_term *= spatial_weight
                e += tv_term
            return e

        def total_energy(v):
            """Total energy / cost function.
            """
            return smooth_energy(v) + nonsmooth_energy(v)

        def grad(v):
            """Gradient of smooth part of energy / cost function.
            """
            grad = v - atom
            if which == "graph-net":
                lap = -_div(_gradient(_unmask(v, mask)))[mask]
                lap *= spatial_weight
                grad += lap
            return grad

        def prox(v, stepsize, dgap_tol, init=None):
            """Proximal operator of non-smooth part of energy / cost function
            """
            if which == "graph-net" or l1_ratio == 1.:
                out = _prox_l1(v, stepsize * l1_weight, copy=False)
                info = dict(converged=True)
            elif which == "tv-l1":
                v = _unmask(v, mask)
                if init is not None:
                    init = _unmask(init, mask)
                out, info = _prox_tvl1(v, init=init, l1_ratio=l1_ratio,
                                       weight=weight * stepsize,
                                       dgap_tol=dgap_tol, max_iter=1000,
                                       verbose=verbose)
                out = out.ravel()[flat_mask]
            else:
                raise ValueError("Unknown value for which: %s" % (which))
            return out, info

        # for debugging
        if check_grad:
            from sklearn.utils.testing import assert_less
            from scipy.optimize import check_grad
            rng = check_random_state(42)
            x0 = rng.randn(n_voxels)
            assert_less(check_grad(smooth_energy, grad, x0), 1e-3)

        # use FISTA update atom
        init = {}
        if atom_init is not None:
            init["w"] = atom_init
        atom, _, _ = mfista(
            grad, prox, total_energy, lips, n_voxels, tol=tol,
            max_iter=max_iter, check_lipschitz=check_lipschitz,
            verbose=verbose, init=init)

    return atom
