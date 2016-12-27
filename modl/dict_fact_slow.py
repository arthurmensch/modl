import numpy as np
import scipy
from numpy import linalg
from sklearn.utils import check_array
from sklearn.utils import check_random_state, gen_batches

from sklearn.linear_model import cd_fast

from modl._utils.enet_proj_fast import enet_norm_fast, enet_projection_fast

from modl._utils.randomkit.random_fast import Sampler

from modl._utils.enet_proj import enet_scale


class DictFactSlow:
    def __init__(self,
                 reduction=1,
                 learning_rate=1,
                 sample_learning_rate=0.76,
                 BC_agg='async',
                 Dx_agg='masked',
                 G_agg='masked',
                 code_alpha=1,
                 code_l1_ratio=1,
                 n_epochs=1,
                 n_components=10,
                 batch_size=10,
                 code_pos=False,
                 D_pos=False,
                 mask_sampling=1,
                 random_state=None,
                 D_l1_ratio=0):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_learning_rate = sample_learning_rate
        self.BC_agg = BC_agg
        self.Dx_agg = Dx_agg
        self.G_agg = G_agg
        self.code_l1_ratio = code_l1_ratio
        self.code_alpha = code_alpha
        self.reduction = reduction
        self.D_l1_ratio = D_l1_ratio

        self.mask_sampling = mask_sampling

        self.D_pos = D_pos
        self.code_pos = code_pos

        self.n_components = n_components
        self.n_epochs = n_epochs

        self.random_state = random_state

    def fit(self, X):
        X = check_array(X, order='F', dtype='float')

        self.prepare(X)

        # Main loop
        n_rows, n_cols = X.shape
        for _ in range(self.n_epochs):
            permutation = self.random_state_.permutation(n_rows)
            X = X[:, permutation]
            if self.G_agg == 'average':
                self.G_average = self.G_average[:, :, permutation]
            if self.Dx_agg == 'average':
                self.Dx_average = self.Dx_average[:, permutation]
                self.code_ = self.code_[:, permutation]
            self.col_counter_ = self.col_counter_[permutation]

            batches = gen_batches(self.batch_size, n_cols)
            for batch in batches:
                len_batch = batch.stop - batch.start
                self.counter_ += len_batch
                self.col_counter_[batch] += 1
                this_X = X[:, batch]
                this_G_average = self.G_average_[batch] if self.G_agg  == 'average' else None
                this_Dx_average = self.Dx_average_[
                    batch] if self.Dx_agg == 'average' else None
                this_code = self.code_[:, batch]
                this_col_counter = self.col_counter_[batch]

                self.batch_fit(this_Dx_average, this_G_average,
                                 this_X, this_code, this_col_counter)

    def prepare(self, X):
        if isinstance(X, tuple):
            n_rows, n_cols = X
        else:
            n_rows, n_cols = X.shape

        # Regression statistics
        if self.G_agg == 'average':
            self.G_average_ = np.zeros((self.n_components, self.n_components,
                                        n_cols), order='F')
        if self.Dx_agg == 'average':
            self.Dx_average_ = np.zeros((self.n_components, self.n_components,
                                         n_cols), order='F')
        # Dictionary statistics
        self.C_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((n_rows, self.n_components), order='F')
        self.gradient_ = np.zeros((n_rows, self.n_components), order='F')

        self.random_state_ = check_random_state(self.random_state)
        if isinstance(X, tuple):
            self.D_ = self.random_state_.randn(n_rows, self.n_components).T
        else:
            random_idx = self.random_state_.permutation(n_cols)[
                         :self.n_components]
            self.D_ = X[:, random_idx].T.copy()
        self.D_ = enet_scale(self.D_, l1_ratio=self.D_l1_ratio, radius=1).T

        self.code_ = np.empty((self.n_components, n_cols), order='F')
        self.code_[:] = 1. / self.n_components

        self.norm_ = np.zeros(self.n_components)

        if self.G_agg == 'full':
            self.G_ = self.D_.t.dot(self.D_)
        else:
            self.G_ = np.zeros((self.n_components, self.n_components),
                               order='F')
        self.counter_ = 0
        self.col_counter_ = np.zeros(n_cols, dtype='int')
        self.random_state_ = check_random_state(self.random_state)
        random_seed = self.random_state_.randint(np.iinfo(np.uint32).max)
        self.row_sampler_ = Sampler(n_rows, self.reduction,
                                    self.mask_sampling, random_seed)

    # def partial_fit(self, X, sample_indices=None):
    #     """Compatibility"""
    #     X = check_array(X, dtype='float', order='F')
    #     n_cols = X.shape[1]
    #     batches = gen_batches(self.batch_size, n_cols)
    #     for batch in batches:
    #         len_batch = batch.stop - batch.start
    #         self.counter_ += len_batch
    #         self.col_counter_[batch] += 1
    #         this_X = X[:, batch]
    #         this_G_average = self.G_average_[sample_indices[batch]] if self.G_agg == 'average' else None
    #         this_Dx_average = self.Dx_average_[sample_indices[batch]] if self.Dx_agg  == 'average' else None
    #         this_code = self.code_[sample_indices[batch]]
    #         this_col_counter = self.col_counter_[sample_indices[batch]]
    #
    #         self.batch_fit(this_Dx_average, this_G_average,
    #                        this_X, this_code, this_col_counter)


    def batch_fit(self, this_Dx_average, this_G_average, this_X,
                    this_code, this_col_counter):
        subset = self.row_sampler_.yield_subset()
        w_sample = np.power(this_col_counter, -self.sample_learning_rate)
        w = pow(self.counter_, -self.learning_rate)
        self.compute_code(this_X, self.D_, self.G_,
                          this_G_average, this_Dx_average, this_code,
                          w_sample,
                          subset)
        if self.BC_agg == 'full':
            self.update_BC(this_X, this_code, self.C_, self.B_,
                           self.gradient_, w, subset)
            self.update_dict(self.D_, self.gradient_, self.C_, self.norm_,
                             self.G_,
                             subset)
        else:
            self.update_BC(this_X, this_code, self.C_, self.B_,
                           self.gradient_, w, subset)
            self.update_dict(self.D_, self.gradient_, self.C_, self.norm_,
                             self.G_,
                             subset)
            self.update_B_full(this_X, this_code, self.B_, w)

    def compute_code(self, this_X, D, G, G_average,
                     Dx_average, code,
                     w_sample, subset):
        n_rows, batch_size = this_X.shape
        reduction = subset.shape[0] / n_rows

        if self.Dx_agg != 'full' or self.G_agg != 'full':
            D_subset = D[subset, :]

        if self.Dx_agg == 'full':
            Dx = D.T.dot(this_X)
        else:
            X_subset = this_X[subset, :]
            Dx = X_subset.T.dot(D_subset).T * reduction
            if self.Dx_agg == 'average':
                Dx_average *= 1 - w_sample[np.newaxis, :]
                Dx_average += Dx * w_sample[np.newaxis, :]
                Dx = Dx_average

        if self.G_agg != 'full':
            D_subset = D[subset, :]
            G = D_subset.T.dot(D_subset) * reduction
            if self.Dx_agg == 'average':
                G_average *= 1 - w_sample[np.newaxis, :]
                G_average += Dx * w_sample[np.newaxis, :]

        for i in range(batch_size):
            if self.G_agg == 'average':
                self.linear_regression(G_average[:, :, i], Dx[:, i], this_X[:, i], code[:, i])
            else:
                self.linear_regression(G, Dx[:, i], this_X[:, i], code[:, i])

    def update_BC(self, this_X, this_code, C, B, gradient, w, subset):
        n_rows, batch_size = this_X.shape
        C *= (1 - w / batch_size)
        C += w * this_code.dot(this_code.T)

        if self.BC_agg == 'full':
            self.update_B_full(this_X, this_code, B, w)
            gradient[:, :] = B
        else:
            X_subset = this_X[subset, :]
            B_subset = B[subset, :]
            B_subset *= (1 - w / batch_size)
            B_subset += w * X_subset.dot(this_code.T)
            gradient[subset, :] = B_subset

    def update_B_full(self, this_X, this_code, B, w):
        n_rows, batch_size = this_X.shape
        B *= (1 - w / batch_size)
        B += w * this_X.dot(this_code.T)

    def update_dict(self, D, gradient, C, norm, G, subset):
        ger, = scipy.linalg.get_blas_funcs(('ger',), (D, C))
        len_subset = subset.shape[0]
        n_rows, n_components = D.shape
        D_subset = D[subset, :]
        atom_temp = np.zeros(len_subset)
        gradient_subset = gradient[subset, :]

        if self.G_agg == 'full' and len_subset < n_rows / 2.:
            G -= D_subset.T.dot(D_subset)

        gradient_subset -= D_subset.dot(C)

        order = self.random_state_.permutation(n_components)

        for k in order:
            subset_norm = enet_norm_fast(D_subset[k, :],
                                         self.D_l1_ratio)
            # r in the text
            norm_proj = norm[k] + subset_norm
            gradient_subset = ger(1.0, D_subset[:, k], C[k, :],
                                  a=gradient_subset, overwrite_a=True)
            if C[k, k] > 1e-20:
                D_subset[:, k] = gradient_subset[:, k] / self.C_[k, k]
                # Else do not update
            if self.D_pos:
                D_subset[D_subset < 0] = 0
            enet_projection_fast(D_subset[:, k],
                                 atom_temp,
                                 norm_proj, self.D_l1_ratio)
            D_subset[:, k] = atom_temp
            subset_norm = enet_norm_fast(D_subset[k, :len_subset],
                                         self.D_l1_ratio)
            self.norm_[k] = norm_proj - subset_norm
            gradient_subset = ger(-1.0, D_subset[:, k], C[k, :],
                                  a=gradient_subset, overwrite_a=True)

        D[subset, :] = D_subset

        if self.G_agg == 'full':
            if len_subset < n_rows / 2.:
                G += D_subset.T.dot(D_subset)
            else:
                G[:] = D.T.dot(D)

    def linear_regression(self, G, Dx, this_X, code):
        n_components = G.shape[0]
        if self.code_l1_ratio == 0:
            G.flat[::n_components + 1] += self.code_alpha
            code[:] = linalg.solve(G, Dx)
            G.flat[::n_components + 1] -= self.code_alpha
        else:
            # this_X is only used to scale tolerance
            cd_fast.enet_coordinate_descent_gram(
                code,
                self.code_alpha * self.code_l1_ratio,
                self.code_alpha * (1 - self.code_l1_ratio),
                G, Dx, this_X, 100, 1e-2, self.random_state_,
                False, self.code_pos)

    def transform(self, X):
        X = check_array(X, order='F', dtype='float')
        n_rows, n_cols = X.shape
        if self.G_agg != 'full':
            G = self.D_.T.dot(self.D_)
        Dx = self.D_.T.dot(X)
        code = np.array((n_rows, self.n_components))
        for i in range(n_cols):
            self.linear_regression(G, Dx[:, i], X[:, i], code[:, i])
        return code

    def score(self, X):
        code = self.transform(X)
        loss = np.sum((X - code.dot(self.D_)) ** 2) / 2
        norm1_code = np.sum(np.abs(code))
        norm2_code = np.sum(code ** 2)
        regul = self.code_alpha * (norm1_code * self.code_l1_ratio
                                   + (1 - self.code_l1_ratio) * norm2_code / 2)
        return (loss + regul) / X.shape[0]
