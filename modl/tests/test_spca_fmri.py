import nibabel
import numpy as np
import pytest
from nilearn._utils.testing import assert_less_equal
from nilearn.image import iter_img

from modl.spca_fmri import SpcaFmri

backends = ['c', 'python']
var_reds = ['weight_based', 'sample_based']


# Utils function are copied from nilearn.decomposition.tests.test_canica
def _make_data_from_components(components, affine, shape, rng=None,
                               n_subjects=8):
    data = []
    if rng is None:
        rng = np.random.RandomState(0)
    cov = rng.uniform(-1, 1, size=(4, 4))
    cov.flat[::5] = 1
    cov *= 10
    mean = np.zeros(4)
    for _ in range(n_subjects):
        loadings = rng.multivariate_normal(mean, cov, size=40)
        this_data = np.dot(loadings, components)
        this_data += .01 * rng.normal(size=this_data.shape)
        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40,) + shape)
        this_data = np.rollaxis(this_data, 0, 4)
        data.append(nibabel.Nifti1Image(this_data, affine))
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
    affine = np.eye(4)
    components = _make_components(shape)
    if noisy:  # Creating noisy non positive data
        components[rng.randn(*components.shape) > .8] *= -5.

    for mp in components:
        assert_less_equal(mp.max(), -mp.min())  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(components, affine, shape, rng=rng,
                                      n_subjects=n_subjects)
    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8), affine)
    return data, mask_img, components, rng


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_sparse_pca(backend, var_red):
    data, mask_img, components, rng = _make_test_data(n_subjects=10)
    sparse_pca = SpcaFmri(n_components=4, random_state=0,
                          mask=mask_img,
                          backend=backend,
                          var_red=var_red,
                          reduction=2 if var_red else 1,
                          smoothing_fwhm=0., n_epochs=3, alpha=0.01)
    sparse_pca.fit(data)
    maps = sparse_pca.masker_. \
        inverse_transform(sparse_pca.components_).get_data()
    maps = np.reshape(np.rollaxis(maps, 3, 0), (4, 400))

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
    if var_red:
        recovered_maps = np.sum(G > 0.7)
    else:
        recovered_maps = np.sum(G > 0.95)
    assert(recovered_maps >= 4)

    # Smoke test n_epochs > 1
    sparse_pca = SpcaFmri(n_components=4, random_state=0,
                          mask=mask_img,
                          var_red=var_red,
                          smoothing_fwhm=0., n_epochs=2, alpha=1)
    sparse_pca.fit(data)

    # Smoke test reduction_ratio < 1
    sparse_pca = SpcaFmri(n_components=4, random_state=0,
                          reduction=2,
                          mask=mask_img,
                          var_red=var_red,
                          smoothing_fwhm=0., n_epochs=1, alpha=1)
    sparse_pca.fit(data)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_test_data(n_subjects=2, noisy=True)
    for mp in components:
        assert_less_equal(-mp.min(), mp.max())

    sparse_pca = SpcaFmri(n_components=4, random_state=rng,
                          mask=mask_img,
                          smoothing_fwhm=0.)
    sparse_pca.fit(data)
    for mp in iter_img(sparse_pca.masker_.inverse_transform(
            sparse_pca.components_)):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))
