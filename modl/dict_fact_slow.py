from sklearn.utils import gen_batches, check_random_state
import numpy as np

class DictFact:
    def __init__(self, n_components, n_epochs,
                 reduction=1,
                 sample_learning_rate,
                 random_state=None):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.reduction = reduction

        self.sample_learning_rate = sample_learning_rate

    def _prepare(self, X):
        n_samples, n_features = X.shape
        self.beta_average = np.zeros((n_samples, self.n_components))
        self.G_average = np.zeros((n_samples, self.n_components, self.n_components))
        self.sample_counter = np.zeros(n_samples)
        self.sample_idx = np.arange(n_samples)

        self.beta = np.zeros(self.n_components)
        self.D = np.zeros(self.n_components, n_features)
        self.G = np.zeros(self.n_components, self.n_components)

        self.feature_idx = np.arange(n_features)
        self.feature_counter = np.zeros(n_features)



    def _reg_stats(self, this_X, feature_cut):
        beta = this_X[:, :feature_cut].dot(self.D[:, :feature_cut].T)
        G = self.D[:, :feature_cut].dot(self.D[:, :feature_cut].T)
        return G, beta


    def _compute_code(self):

    def _update_surrogate_stats(self):

    def _minimize_surrogate(self):


    def fit(self, X):
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        self._prepare(X)
        for n in range(self.n_epochs):
            permutation = random_state.permutation(n_samples)
            X = X[permutation]
            self.sample_idx = self.sample_idx[permutation]
            self.sample_counter = self.sample_counter[permutation]
            self.G_average = self.G_average[permutation]
            self.beta_average = self.beta_average[permutation]

            batches = gen_batches(n_samples)

            for batch in batches:
                this_X = X[batch]
                batch_size, _ = this_X.shape
                self.sample_counter[batch] += 1
                permutation = random_state.permutation(n_features)
                this_X = this_X[:, permutation]
                self.feature_idx = self.feature_idx[permutation]
                self.feature_counter = self.feature_counter[permutation]
                self.D = self.D[:, permutation]
                feature_cut = n_features // self.reduction
                self.feature_counter[:feature_cut] += batch_size

                G, beta = self._reg_stats(this_X, feature_cut)

                w_sample = np.power(self.sample_counter[batch],
                                  -self.sample_learning_rate)
                self.beta_average[batch] *= 1 - w_sample
                self.beta_average[batch] += (1 - w_sample / batch_size) * beta

                self.G_average[batch] *= 1 - w_sample
                self.G_average[batch] += np.outer(1 - w_sample / batch_size, G)

