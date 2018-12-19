from sklearn.utils import check_random_state

from modl.decomposition.stability import mean_amari_discrepency, amari_discrepency

from numpy.testing import assert_almost_equal


def test_amari_discrepency():
    n_components, n_features = 50, 100
    rng = check_random_state(23)
    dictionaries = [rng.randn(n_components, n_features) for _ in range(2)]
    assert amari_discrepency(dictionaries[0], dictionaries[1]) >= 0
    assert_almost_equal(amari_discrepency(dictionaries[0], dictionaries[0]), 0)


def test_mean_amari_discrepency():
    n_components, n_features, n_dictionaries = 50, 100, 20
    rng = check_random_state(23)
    dictionaries = [rng.randn(n_components, n_features) for _ in range(n_dictionaries)]
    discrepency, var_discrepency = mean_amari_discrepency(dictionaries)
    print(discrepency)
    print(var_discrepency)