from numpy.testing import assert_array_almost_equal
from sklearn import datasets

import pytest
from modl.classification import OurLogisticRegressionCV


@pytest.mark.parametrize("solver", ['sgd_sklearn'])
@pytest.mark.parametrize("penalty", ['l1', 'l2'])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("fit_intercept", [True])
def test_our_logistic_regression_cv(penalty, refit, solver, fit_intercept):
    if solver == 'sag_sklearn' and penalty == 'l1':
        pytest.skip()
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
        solver=solver,
        tol=0.001,
        max_iter=100, cv=4, penalty=penalty,
        fit_intercept=fit_intercept,
        verbose=0,
        random_state=0)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    assert (score > 0.8)


@pytest.mark.parametrize("penalty", ['l1', 'l2'])
@pytest.mark.parametrize("refit", [True])
def test_solver_same_result(penalty, refit, ):
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
        solver='cd',
        tol=1e-10,
        max_iter=1000, cv=2, penalty=penalty,
        verbose=0,
        random_state=0)
    classifier_saga = OurLogisticRegressionCV(
        alphas=[.1, 1, 10, 100, 1000],
        refit=refit,
        solver='saga',
        tol=1e-10,
        max_iter=1000, cv=2, penalty=penalty,
        verbose=0,
        random_state=0)
    for classifier in [classifier_cd, classifier_saga]:
        classifier.fit(X_train, y_train)
    assert_array_almost_equal(classifier_cd.coef_,
                              classifier_saga.coef_, 2)
