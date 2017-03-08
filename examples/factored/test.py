import warnings
from tempfile import NamedTemporaryFile

from io import BytesIO
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV

from modl.factored_logistic import FactoredLogistic

import numpy as np

from sklearn.datasets import load_digits


lr = FactoredLogistic(optimizer='adagrad', latent_dim=20, max_iter=20,
                      activation='relu',
                      alpha=0.1,
                      batch_size=200, )

alphas = np.logspace(-3, 3, 7)

grid_search = GridSearchCV(lr,
                           {'alpha': alphas},
                           cv=2,
                           refit=True,
                           verbose=1,
                           n_jobs=1)

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

grid_search.fit(X_train, y_train)

dump(grid_search, 'test')
grid_search = load('test')
y_pred = grid_search.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(accuracy)
