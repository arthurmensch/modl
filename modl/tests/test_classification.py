import pytest
from modl.classification import OurLogisticRegressionCV

from sklearn import datasets


@pytest.mark.parametrize("loss", ['l1', 'l2'])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("refit", [True, False])
def test_our_logistic_regression_cv(loss, standardize, refit):
    digits = datasets.load_digits()

    from sklearn.model_selection import train_test_split

    X, y = digits.data, digits.target
    X = X[::5]
    y = y[::5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=.1)
    classifier = OurLogisticRegressionCV(
        Cs=[.1, 1, 10],
        refit=refit,
        standardize=standardize,
        tol=0.01,
        max_iter=10, cv=2, loss=loss)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    assert (score > 0.8)
