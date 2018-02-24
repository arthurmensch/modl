import time

import numpy as np
from lightning.impl.primal_cd import CDClassifier
from lightning.impl.sag import SAGAClassifier

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import SAGClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

alpha = 1e-3
n_samples = X.shape[0]

# sag = SAGClassifier(eta='auto',
#                     loss='log',
#                     alpha=alpha,
#                     penalty='l1',
#                     tol=1e-10,
#                     max_iter=1000,
#                     verbose=1,
#                     random_state=0)
saga = SAGAClassifier(eta='auto',
                      loss='log',
                      penalty='l1',
                      alpha=0,
                      beta=alpha,
                      tol=1e-10,
                      max_iter=20,
                      verbose=1,
                      random_state=0)
cd_classifier = CDClassifier(loss='log',
                             penalty='l1',
                             alpha=alpha,
                             C=1 / n_samples,
                             tol=1e-10,
                             max_iter=20,
                             verbose=1,
                             random_state=0)
sklearn_sag = LogisticRegression(tol=1e-10, max_iter=1000,
                                 verbose=2, random_state=0,
                                 C=1. / (n_samples * alpha),
                                 solver='liblinear',
                                 penalty='l1',
                                 dual=False,
                                 fit_intercept=False)
sklearn_sgd = SGDClassifier(loss='log', penalty='l1', alpha=alpha,
                            n_iter=15)

classifiers = [
    # {'name': 'Lightning SAG', 'estimator': sag},
               {'name': 'Lightning SAGA', 'estimator': saga},
    #            {'name': 'Sklearn SAG', 'estimator': sklearn_sag},
               {'name': 'Lightning CD', 'estimator': cd_classifier},
               # {'name': 'Sklearn SGD', 'estimator': sklearn_sgd},
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
