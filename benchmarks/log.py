import time

import numpy as np
from lightning.impl.primal_cd import CDClassifier
from lightning.impl.sag import SAGAClassifier

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import SAGClassifier
from sklearn.linear_model import LogisticRegression

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

alpha = 1e-3
n_samples = X.shape[0]

sag = SAGClassifier(eta='auto',
                    loss='log',
                    alpha=alpha,
                    tol=1e-10,
                    max_iter=1000,
                    verbose=1,
                    random_state=0)
saga = SAGAClassifier(eta='auto',
                      loss='log',
                      alpha=alpha,
                      tol=1e-10,
                      max_iter=1000,
                      verbose=1,
                      random_state=0)
cd_classifier = CDClassifier(loss='log',
                             alpha=alpha / 2,
                             C=1 / n_samples,
                             tol=1e-10,
                             max_iter=100,
                             verbose=1,
                             random_state=0)
sklearn_sag = LogisticRegression(tol=1e-10, max_iter=1000,
                                 verbose=2, random_state=0,
                                 C=1. / (n_samples * alpha),
                                 solver='sag',
                                 penalty='l2',
                                 fit_intercept=False)

classifiers = [{'name': 'Lightning SAG', 'estimator': sag},
               {'name': 'Lightning SAGA', 'estimator': saga},
               {'name': 'Sklearn SAG', 'estimator': sklearn_sag},
               {'name': 'Lightning CD', 'estimator': cd_classifier},
               ]

start = time.time()
for classifier in classifiers:
    print(classifier['name'])
    clf = classifier['estimator']
    clf.fit(X, y)

    print("Training time", time.time() - start)
    print("Accuracy", np.mean(clf.predict(X) == y))
    n_nz = np.sum(np.sum(clf.coef_ != 0, axis=0, dtype=bool))
    n_nz /= clf.coef_.size
    print(clf.coef_)
    print('Non-zero', n_nz)
