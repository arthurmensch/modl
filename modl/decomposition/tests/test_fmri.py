import nibabel
import numpy as np
import pytest
from nilearn.image import iter_img
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory

from modl.decomposition import fMRIDictFact
from modl.utils.system import get_cache_dirs

methods = ['masked', 'average', 'gram', 'reducing ratio', 'dictionary only']
memories = [False, True]

# Utils function are copied from nilearn.decomposition.tests.test_canica
def _make_data_from_components(components,
                               n_subjects=8):
    data = []
    rng = np.random.RandomState(0)
    cov = rng.uniform(-1, 1, size=(4, 4))
    cov.flat[::5] = 1
    cov *= 10
    mean = np.zeros(4)
    for _ in range(n_subjects):
        loadings = rng.randn(40, 4)
        this_data = np.dot(loadings, components)
        this_data += 0.01 * rng.normal(size=this_data.shape)
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


def _make_test_data(n_subjects=8, noisy=False):
    rng = np.random.RandomState(0)
    shape = (20, 20, 1)
    components = _make_components(shape)
    if noisy:  # Creating noisy non positive data
        components[rng.randn(*components.shape) > .8] *= -5.
        for component in components:
            assert(component.max() <= -component.min())  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(components, n_subjects=n_subjects)
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8),
                                   affine)
    masker = MultiNiftiMasker(mask_img).fit()
    init = components + 1 * rng.randn(*components.shape)
    components = masker.inverse_transform(components)
    init = masker.inverse_transform(init)
    data = masker.inverse_transform(data)

    return data, mask_img, components, init


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("memory", memories)
def test_dict_fact(method, memory):
    if memory:
        memory = Memory(cachedir=get_cache_dirs()[0])
        memory_level = 2
    else:
        if method != 'masked':
            pytest.skip()
        memory = Memory(cachedir=None)
        memory_level = 0
    data, mask_img, components, init = _make_test_data(n_subjects=10)
    dict_fact = fMRIDictFact(n_components=4, random_state=0,
                             memory=memory,
                             memory_level=memory_level,
                             mask=mask_img,
                             dict_init=init,
                             method=method,
                             reduction=2,
                             smoothing_fwhm=None, n_epochs=2, alpha=1)
    dict_fact.fit(data)
    maps = np.rollaxis(dict_fact.components_img_.get_data(), 3, 0)
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

    recovered_maps = np.sum(G > 0.95)
    assert (recovered_maps >= 4)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of pipelining in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, init = _make_test_data(n_subjects=2,
                                                       noisy=True)

    dict_fact = fMRIDictFact(n_components=4, random_state=0,
                             mask=mask_img,
                             smoothing_fwhm=None)
    dict_fact.fit(data)
    for mp in iter_img(dict_fact.components_img_):
        mp = mp.get_data()
        assert(np.sum(mp[mp <= 0]) <= np.sum(mp[mp > 0]))

def test_verbose():
    pass

def test_score():
    pass

def test_transform():
    pass