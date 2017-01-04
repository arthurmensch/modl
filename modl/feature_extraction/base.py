from sklearn.utils import check_array, gen_batches
import numpy as np
from sklearn.utils import check_random_state


class BaseBatcher(object):
    def __init__(self, batch_size=10, random_state=None):
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, data_source):
        pass

    def partial_transform(self):
        return
        yield

    def generate_single(self):
        return next(self.partial_transform())


class NumpyBatcher(BaseBatcher):
    def __init__(self, batch_size=10, random_state=None, copy=False,
                 max_samples=None):
        BaseBatcher.__init__(self,
                             batch_size=batch_size,
                             random_state=random_state)
        self.copy = copy
        self.max_samples = max_samples

    def fit(self, data_source):
        self.data_ = check_array(data_source)
        n_samples = self.data_.shape[0]
        self.random_state_ = check_random_state(self.random_state_)
        selection = self.random_state_.permutation(n_samples)[
                    :self.max_samples]
        self.n_samples_ = selection.shape[0]
        self.indices_ = np.arange(self.n_samples_)
        if self.copy:
            self.data_ = self.data_[selection]
        else:
            self.base_indices_ = selection

    def partial_transform(self):
        batches = gen_batches(self.n_samples_, self.batch_size)
        for batch in batches:
            if self.copy:
                yield self.data_[batch], self.indices_[batch]
            else:
                yield self.data_[self.base_indices_[batch]], \
                      self.indices_[batch]

    def shuffle(self, permutation=None):
        if permutation is None:
            permutation = self.random_state_.permutation(self.n_samples_)
        self.indices_ = self.indices_[permutation]
        if self.copy:
            self.data_ = self.data_[permutation]
        else:
            self.base_indices_ = self.base_indices_[permutation]
