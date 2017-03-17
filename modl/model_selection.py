from pandas import Series
from sklearn.model_selection._split import BaseShuffleSplit, GroupShuffleSplit

import numpy as np


class StratifiedGroupShuffleSplit(BaseShuffleSplit):
    def __init__(self, test_size=.1, train_size=None, random_state=None,
                 stratify_name='dataset', group_name='subject', n_splits=3):
        super().__init__(n_splits, test_size, train_size, random_state)

        self.stratify_name = stratify_name
        self.group_name = group_name

    def _iter_indices(self, X, y=None, groups=None):
        index_series = Series(index=X.index, data=np.arange(X.shape[0]))
        splitters = []
        indices = []
        for idx, serie in index_series.groupby(level=self.stratify_name):
            cv = GroupShuffleSplit(n_splits=self.n_splits,
                                   test_size=self.test_size,
                                   train_size=self.train_size,
                                   random_state=self.random_state)
            groups = serie.index.get_level_values(self.group_name).values
            splitter = cv.split(serie, groups=groups)
            index = serie.values
            indices.append(index)
            splitters.append(splitter)

        has_next = True
        while has_next:
            train, test = [], []
            for splitter, index in zip(splitters, indices):
                try:
                    this_train, this_test = next(splitter)
                    this_train, this_test = index[this_train], index[this_test]
                except StopIteration:
                    has_next = False
                    break
                train.append(this_train)
                test.append(this_test)
            if has_next:
                train = np.concatenate(train)
                test = np.concatenate(test)
                yield train, test
