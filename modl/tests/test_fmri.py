import nibabel
import numpy as np
import pytest
from nilearn._utils.testing import assert_less_equal
from nilearn.image import iter_img
from nilearn.input_data import MultiNiftiMasker

from modl import fMRIDictFact

methods = ['masked', 'average', 'gram', 'reducing ratio', 'dictionary only']
methods = ['masked']


# Utils function are copied from nilearn.decomposition.tests.test_canica
def _make_data_from_components(components, shape, rng=None,
                               n_subjects=8):
    data = []
    if rng is None:
        rng = np.random.RandomState(0)
    cov = rng.uniform(-1, 1, size=(4, 4))
    cov.flat[::5] = 1
    cov *= 10
    mean = np.zeros(4)
    for _ in range(n_subjects):
        loadings = rng.randn(40, 4)
        this_data = np.dot(loadings, components)
        this_data += 10 * rng.normal(size=this_data.shape)
        data.append(this_data)
    return data


def _make_components(shape):
    # Create two images with "activated regions"
    component1 = np.zeros(shape)
    component1[:5, :10] = 1
    component1[5:10, :10] = -1

    component2 = np.zeros(shape)
    component2[:5, -10:] = 1
    component2[5:10, -10:] = -1

    component3 = np.zeros(shape)
    component3[-5:, -10:] = 1
    component3[-10:-5, -10:] = -1

    component4 = np.zeros(shape)
    component4[-5:, :10] = 1
    component4[-10:-5, :10] = -1

    return np.vstack((component1.ravel(), component2.ravel(),
                      component3.ravel(), component4.ravel()))


def _make_test_data(rng=None, n_subjects=8, noisy=False):
    if rng is None:
        rng = np.random.RandomState(0)
    shape = (20, 20, 1)
    components = _make_components(shape)
    if noisy:  # Creating noisy non positive data
        components[rng.randn(*components.shape) > .8] *= -5.
        components_masked = components.get_data()
        for component in components_masked:
            assert_less_equal(component.max(), -component.min())  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(components, shape, rng=rng,
                                      n_subjects=n_subjects)
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8),
                                   affine)
    masker = MultiNiftiMasker(mask_img).fit()
    components = masker.inverse_transform(components)
    data = masker.inverse_transform(data)

    return data, mask_img, components, rng


@pytest.mark.parametrize("method", methods)
def test_sparse_pca(method):
    data, mask_img, components, rng = _make_test_data(n_subjects=10)
    sparse_pca = fMRIDictFact(n_components=4, random_state=0,
                              mask=mask_img,
                              dict_init=components,
                              method=method,
                              reduction=2,
                              smoothing_fwhm=0., n_epochs=0, alpha=0.1)
    sparse_pca.fit(data)
    maps = np.rollaxis(sparse_pca.components_.get_data(), 3, 0)
    components = np.rollaxis(components.get_data(), 3, 0)
    maps = maps.reshape((maps.shape[0], -1))
    components = components.reshape((components.shape[0], -1))

    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    S = np.sqrt(np.sum(maps ** 2, axis=1))
    S[S == 0] = 1
    maps /= S[:, np.newaxis]

    G = np.abs(components.dot(maps.T))
    # Hard
    # if var_red:
    #     recovered_maps = min(np.sum(np.any(G > 0.5, axis=1)),
    #                          np.sum(np.any(G > 0.5, axis=0)))
    # else:
    #     recovered_maps = min(np.sum(np.any(G > 0.95, axis=1)),
    #                  np.sum(np.any(G > 0.95, axis=0)))

    recovered_maps = np.sum(G > 0.95)
    print(G)
    assert (recovered_maps >= 4)

    # Smoke test n_epochs > 1
    sparse_pca = fMRIDictFact(n_components=4, random_state=0,
                              mask=mask_img,
                              method=method,
                              smoothing_fwhm=0., n_epochs=2, alpha=1)
    sparse_pca.fit(data)

    # Smoke test reduction_ratio < 1
    sparse_pca = fMRIDictFact(n_components=4, random_state=0,
                              reduction=2,
                              mask=mask_img,
                              method=method,
                              smoothing_fwhm=0., n_epochs=1, alpha=1)
    sparse_pca.fit(data)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_test_data(n_subjects=2, noisy=True)

    sparse_pca = fMRIDictFact(n_components=4, random_state=rng,
                              mask=mask_img,
                              smoothing_fwhm=0.)
    sparse_pca.fit(data)
    for mp in iter_img(sparse_pca.components_):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))
