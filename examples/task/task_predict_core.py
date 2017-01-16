from functools import reduce
from operator import add

import numpy as np
from modl import fMRIDictFact
from modl.utils.system import get_cache_dirs
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def get_encodings(components, mask_img,
                  task_dataset, test, train, n_jobs):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=0)
    dict_fact = fMRIDictFact(smoothing_fwhm=0,
                             mask=mask_img,
                             detrend=False,
                             standardize=False,
                             memory_level=2,
                             memory=memory,
                             n_jobs=n_jobs,
                             dict_init=components,
                             ).fit()
    train_task_data = np.array(task_dataset.func)[train].tolist()
    train_task_contrasts = np.array(task_dataset.contrast)[train].tolist()
    train_task_data = reduce(add, train_task_data)
    train_task_contrasts = reduce(add, train_task_contrasts)

    X_train = np.concatenate(dict_fact.transform(train_task_data))
    y_train = LabelEncoder().fit_transform(train_task_contrasts)

    test_task_data = np.array(task_dataset.func)[test].tolist()
    test_task_contrasts = np.array(task_dataset.contrast)[test].tolist()
    test_task_data = reduce(add, test_task_data)
    test_task_contrasts = reduce(add, test_task_contrasts)

    X_test = np.concatenate(dict_fact.transform(test_task_data))
    y_test = LabelEncoder().fit_transform(test_task_contrasts)

    return X_test, X_train, y_test, y_train


def fit_lr(X_train, y_train):
    lr = LogisticRegression(multi_class='multinomial',
                            solver='sag', max_iter=1000, verbose=2)
    lr.fit(X_train, y_train)
    return lr


def get_components(alpha, batch_size, learning_rate, mask_img,
                   n_components, n_epochs, n_jobs, reduction, smoothing_fwhm,
                   method,
                   train_rest_data,
                   random_state,
                   verbose):
    memory = Memory(cachedir=get_cache_dirs()[0], verbose=0)
    dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                             mask=mask_img,
                             memory=memory,
                             method=method,
                             memory_level=2,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             n_jobs=n_jobs,
                             random_state=random_state,
                             n_components=n_components,
                             dict_init=None,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             reduction=reduction,
                             alpha=alpha,
                             )
    dict_fact.fit(train_rest_data)
    components = dict_fact.components_
    return components