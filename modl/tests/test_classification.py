from numpy.testing import assert_array_almost_equal
from sklearn import datasets

import pytest
from modl.classification import OurLogisticRegressionCV


@pytest.mark.parametrize("solver", ['cd', 'saga', 'sag_sklearn'])
@pytest.mark.parametrize("penalty", ['l1', 'l2'])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("refit", [True, False])
def test_our_logistic_regression_cv(penalty, standardize, refit, solver):
    digits = datasets.load_digits()

    from sklearn.model_selection import train_test_split

    X, y = digits.data, digits.target
    X = X[::5]
    y = y[::5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=.1)
    classifier = OurLogisticRegressionCV(
        alphas=[0.01, .1, 1],
        refit=refit,
        standardize=standardize,
        solver=solver,
        tol=0.001,
        max_iter=100, cv=4, penalty=penalty,
        verbose=0,
        random_state=0)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    assert (score > 0.8)


@pytest.mark.parametrize("penalty", ['l1', 'l2'])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("refit", [True])
def test_solver_same_result(penalty, standardize, refit, ):
    digits = datasets.load_digits()

    from sklearn.model_selection import train_test_split

    X, y = digits.data, digits.target
    X = X[::5]
    y = y[::5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=.1)
    classifier_cd = OurLogisticRegressionCV(
        alphas=[.1, 1, 10, 100, 1000],
        refit=refit,
        standardize=standardize,
        solver='cd',
        tol=1e-10,
        max_iter=1000, cv=2, penalty=penalty,
        verbose=0,
        random_state=0)
    classifier_saga = OurLogisticRegressionCV(
        alphas=[.1, 1, 10, 100, 1000],
        refit=refit,
        standardize=standardize,
        solver='saga',
        tol=1e-10,
        max_iter=1000, cv=2, penalty=penalty,
        verbose=0,
        random_state=0)
    for classifier in [classifier_cd, classifier_saga]:
        classifier.fit(X_train, y_train)
    if not standardize:
        assert_array_almost_equal(classifier_cd.estimator_.coef_,
                                  classifier_saga.estimator_.coef_, 2)
    else:
        assert_array_almost_equal(classifier_cd.estimator_.steps[1][1].coef_,
                                  classifier_saga.estimator_.steps[1][1].coef_,
                                  1)
