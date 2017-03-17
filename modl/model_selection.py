from pandas import Series
from sklearn.model_selection._split import BaseShuffleSplit, GroupShuffleSplit

import numpy as np


class StratifiedGroupShuffleSplit(BaseShuffleSplit):
    def __init(self, test_size=.1, train_size=None, random_state=None,
               stratify_name='dataset',
               group_name='subject',
               n_splits=3):
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits

        self.stratify_name = stratify_name
        self.group_name = group_name

    def _iter_indices(self, X, y=None, groups=None):
        index_series = Series(index=X.index, data=np.arange(X.shape[0]))
        splitters = []
        for idx, serie in index_series.groupby(level=self.stratify_name):
            cv = GroupShuffleSplit(n_splits=self.n_splits,
                                   test_size=self.test_size,
                                   train_size=self.train_size,
                                   random_state=self.random_state)
            splitter = cv.split(serie.values,
                                groups=serie.index.
                                get_level_values(self.group_name).values)
            splitters.append(splitter)

        has_next = True
        while has_next:
            train, test = [], []
            for splitter in zip(splitters):
                try:
                    this_train, this_test = next(splitter)
                except StopIteration:
                    has_next = False
                    break
                train.append(this_train)
                test.append(this_test)
            if has_next:
                train = np.concatenate(train)
                test = np.concatenate(test)
                yield train, test
