# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# noinspection PyUnresolvedReferences

cimport cython

cimport numpy as np
from cython cimport view

from libc.stdio cimport printf
# noinspection PyUnresolvedReferences
from libc.math cimport pow, ceil, floor, fmin, fmax, fabs

from libc.math cimport sqrt
from scipy.linalg.cython_blas cimport dgemm, dger, daxpy, ddot, dasum, dgemv
from scipy.linalg.cython_lapack cimport dposv

# noinspection PyUnresolvedReferences
from ._utils.randomkit.random_fast cimport Sampler, RandomState
# noinspection PyUnresolvedReferences
from ._utils.enet_proj_fast cimport enet_projection_fast, enet_norm_fast, \
      enet_scale_fast

from ._utils.enet_proj_fast import enet_scale_matrix_fast

from tempfile import NamedTemporaryFile
import numpy as np

# import
from cython.parallel import parallel, prange

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int ZERO = 0
cdef int ONE = 1
cdef double FZERO = 0
cdef double FONE = 1
cdef double FMONE = -1

cdef double abs_max(int n, double* A_) nogil:
    """np.max(np.abs(A_))"""
    cdef int i
    cdef double m = fabs(A_[0])
    cdef double D_
    for i in range(1, n):
        D_ = fabs(A_[i])
        if D_ > m:
            m = D_
    return m


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef double pmax(int n, double* A_) nogil:
    """np.max(A_)"""
    cdef int i
    cdef double m = A_[0]
    cdef double D_
    for i in range(1, n):
        D_ = A_[i]
        if D_ > m:
            m = D_
    return m

cpdef double get_simple_weights(long count, long batch_size,
           double learning_rate, double offset) nogil:
    cdef long i
    cdef double w = 1
    for i in range(count + 1 - batch_size, count + 1):
        w *= (1 - pow((1 + offset) / (offset + i), learning_rate))
    w = 1 - w
    return w

@cython.final
cdef class DictFactImpl(object):

    cdef readonly int batch_size
    cdef readonly double learning_rate
    cdef readonly double offset
    cdef readonly double sample_learning_rate
    cdef readonly double reduction
    cdef readonly double alpha
    cdef readonly double l1_ratio
    cdef readonly double pen_l1_ratio
    cdef readonly double lasso_tol

    cdef readonly int G_agg
    cdef readonly int Dx_agg
    cdef readonly int AB_agg

    cdef readonly bint non_negative_A
    cdef readonly bint non_negative_D

    cdef readonly double purge_tol

    cdef readonly int subset_sampling
    cdef readonly int dict_reduction

    cdef readonly int max_n_iter
    cdef readonly int n_samples
    cdef readonly int n_features
    cdef readonly int n_components
    cdef readonly int len_subset
    cdef readonly int n_threads
    cdef readonly int n_thread_batches
    cdef readonly unsigned long random_seed
    cdef public int[:] verbose_iter
    cdef public int verbose_iter_idx_

    cdef readonly double[::1, :] D_
    cdef readonly double[:, ::1] code_
    cdef readonly double[::1, :] A_
    cdef readonly double[::1, :] B_
    cdef readonly double[::1, :] G_

    cdef readonly double[::1, :, :] G_average_
    cdef readonly double[::1, :, :] G_average_temp
    cdef readonly double[::1, :] Dx_average_

    cdef readonly int[:] sample_counter_
    cdef readonly int[:] feature_counter_
    cdef readonly int total_counter_

    cdef readonly double[:] norm_

    cdef double[::1, :] this_X
    cdef double[:, ::1] X_batch
    cdef int[:] sample_indices_batch
    cdef double[::1, :] D_subset

    cdef double[::1, :] Dx
    cdef double[::1, :, :] G_temp
    cdef double[::1, :] R_
    cdef double[:] norm_temp

    cdef double[:] proj_temp
    cdef double[:, ::1] H
    cdef double[:, ::1] XtA

    cdef Sampler feature_sampler_1
    cdef Sampler feature_sampler_2
    cdef int[:] D_range

    cdef RandomState random_state_

    cdef object callback
    cdef readonly unicode G_average_filename
    cdef readonly unicode temp_dir


    def __init__(self,
                 double[::1, :] D,
                 int n_samples,
                 double alpha=1.0,
                 double l1_ratio=0.,
                 double pen_l1_ratio=0.,
                 double lasso_tol=1e-3,
                 double purge_tol=0,
                 # Hyper-parameters
                 double learning_rate=1.,
                 double sample_learning_rate=0.76,
                 int batch_size=1,
                 double offset=0,
                 # Reduction parameter
                 int reduction=1,
                 int G_agg=1,
                 int Dx_agg=1,
                 int AB_agg=1,
                 bint non_negative_A=0,
                 bint non_negative_D=0,
                 int subset_sampling=1,
                 int dict_reduction=0,
                 # Dict parameter
                 # Generic parameters
                 unsigned long random_seed=0,
                 int[:] verbose_iter=None,
                 int n_threads=1,
                 unicode temp_dir=None,
                 object callback=None):
        cdef int i
        cdef double* G_ptr
        cdef double* D_ptr
        cdef int G_average_temp_size

        self.temp_dir = temp_dir
        self.n_samples = n_samples
        self.n_components = D.shape[0]
        self.n_features = D.shape[1]

        self.reduction = reduction

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.sample_learning_rate = sample_learning_rate

        self.non_negative_A = non_negative_A
        self.non_negative_D = non_negative_D

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pen_l1_ratio = pen_l1_ratio

        self.G_agg = G_agg
        self.Dx_agg = Dx_agg

        self.subset_sampling = subset_sampling
        self.dict_reduction = dict_reduction
        self.AB_agg = AB_agg

        self.lasso_tol = lasso_tol
        self.purge_tol = purge_tol

        self.random_seed = random_seed

        self.n_threads = n_threads
        self.n_thread_batches = min(n_threads, self.batch_size)

        self.random_state_ = RandomState(seed=self.random_seed)

        self.D_ = view.array((self.n_components, self.n_features),
                                   sizeof(double),
                                   format='d',
                                   mode='fortran')

        self.D_[:] = D[:]
        enet_scale_matrix_fast(D, self.l1_ratio, radius=1)

        self.norm_ = view.array((self.n_components, ),
                                   sizeof(double),
                                   format='d')
        self.norm_[:] = 1

        self.code_ = view.array((self.n_samples, self.n_components),
                               sizeof(double), format='d',
                               mode='c')
        self.code_[:] = 1

        self.A_ = view.array((self.n_components, self.n_components),
                                              sizeof(double),
                                              format='d', mode='fortran')
        self.A_[:] = 0
        self.B_ = view.array((self.n_components, self.n_features),
                                              sizeof(double),
                                              format='d', mode='fortran')
        self.B_[:] = 0
        self.G_ = view.array((self.n_components, self.n_components),
                                          sizeof(double),
                                          format='d', mode='fortran')

        if self.G_agg == 2:
            D_ptr = &self.D_[0, 0]
            G_ptr = &self.G_[0, 0]
            dgemm(&NTRANS, &TRANS,
                  &self.n_components, &self.n_components, &self.n_features,
                  &FONE,
                  D_ptr, &self.n_components,
                  D_ptr, &self.n_components,
                  &FZERO,
                  G_ptr, &self.n_components
                  )
        elif self.G_agg == 3:
            if self.temp_dir is None:
                self.G_average_ = view.array((self.n_components, self.n_components,
                self.n_samples), sizeof(double),
                                format='d', mode='fortran')
            else:
                f = NamedTemporaryFile(dir=self.temp_dir,
                                       buffering=(8 * self.n_components ** 2))
                self.G_average_filename = f.name
                self.G_average_ = np.memmap(self.G_average_filename,
                                        dtype='double',
                                        mode='w+',
                                        order='F',
                                        shape=(self.n_components, self.n_components,
                                               self.n_samples))
        if self.Dx_agg == 3:
            self.Dx_average_ = view.array((self.n_components, self.n_samples),
                                              sizeof(double),
                                              format='d', mode='fortran')

        self.sample_counter_ = view.array((self.n_samples, ), sizeof(int),
                                         format='i')
        self.feature_counter_ = view.array((self.n_features, ), sizeof(int),
                                         format='i')
        self.feature_counter_[:] = 0
        self.sample_counter_[:] = 0
        self.total_counter_ = 0

        self.X_batch = view.array((self.batch_size, self.n_features),
                                  sizeof(double),
                                  format='d',
                                  mode='c')
        self.this_X = view.array((self.batch_size, self.n_features),
                                 sizeof(double),
                                 format='d',
                                 mode='fortran')
        self.sample_indices_batch = view.array((self.batch_size, ),
                                               sizeof(int),
                                               format='i')

        self.D_subset = view.array((self.n_components, self.n_features),
                                   sizeof(double),
                                   format='d',
                                   mode='fortran')
        self.Dx = view.array((self.n_components, self.batch_size),
                             sizeof(double),
                             format='d',
                             mode='fortran')

        if self.pen_l1_ratio == 0:
            self.G_temp = view.array((self.n_components, self.n_components,
                                      self.n_thread_batches),
                                              sizeof(double),
                                              format='d', mode='fortran')
        else:
            self.H = view.array((self.n_thread_batches, self.n_components),
                  sizeof(double),
                  format='d', mode='c')
            self.XtA = view.array((self.n_thread_batches, self.n_components),
                  sizeof(double),
                  format='d', mode='c')

        self.R_ = view.array((self.n_components, self.n_features),
                                   sizeof(double),
                                   format='d',
                                   mode='fortran')
        self.norm_temp = view.array((self.n_components, ), sizeof(double),
                                    format='d')
        self.proj_temp = view.array((self.n_features, ), sizeof(double),
                                    format='d')


        self.D_range = self.random_state_.permutation(self.n_components)

        for i in range(self.n_components):
            self.D_range[i] = i

        random_seed = self.random_state_.randint()
        self.feature_sampler_1 = Sampler(self.n_features, True, True,
                                    random_seed)

        if self.dict_reduction != 0:
            random_seed = self.random_state_.randint()
            self.feature_sampler_2 = Sampler(self.n_features,
                                             True, True,random_seed)

        self.callback = callback

        self.verbose_iter = verbose_iter
        self.verbose_iter_idx_ = 0

    cpdef void partial_fit(self, np.ndarray[double, ndim=2, mode='c'] X,
                           int[:] sample_indices) except *:
        cdef int this_n_samples = X.shape[0]
        cdef int n_batches = int(ceil(float(this_n_samples) / self.batch_size))
        cdef int start = 0
        cdef int stop = 0
        cdef int len_batch = 0

        cdef int i, ii, jj, bb, j, k, t, p, h

        cdef int[:] subset

        cdef int[:] random_order = self.random_state_.permutation(this_n_samples)

        cdef int reduction_int = int(self.reduction)

        cdef int next_verbose_iter = 0
        if self.verbose_iter is not None and \
                        self.verbose_iter_idx_ < self.verbose_iter.shape[0]:
            next_verbose_iter = self.verbose_iter[self.verbose_iter_idx_]
        else:
            next_verbose_iter = -1

        for k in range(n_batches):
            if self.verbose_iter is not None and\
                    self.total_counter_ >= next_verbose_iter >= 0:
                printf("Iteration %i\n", self.total_counter_)
                if self.callback is not None:
                    self.callback()
                self.verbose_iter_idx_ += 1
                if self.verbose_iter_idx_ < self.verbose_iter.shape[0]:
                    next_verbose_iter = self.verbose_iter[self.verbose_iter_idx_]
                else:
                    # Disable verbosity
                    next_verbose_iter = -1

            start = k * self.batch_size
            stop = start + self.batch_size
            if stop > this_n_samples:
                stop = this_n_samples
            len_batch = stop - start
            self.total_counter_ += len_batch

            subset = self.feature_sampler_1.yield_subset(self.reduction)
            for jj in range(subset.shape[0]):
                j = subset[jj]
                self.feature_counter_[j] += len_batch

            # X is a numpy array: we need gil
            for bb, ii in enumerate(range(start, stop)):
                self.sample_indices_batch[bb] = \
                    sample_indices[random_order[ii]]
                for j in range(self.n_features):
                    self.X_batch[bb, j] = X[random_order[ii], j]
                self.sample_counter_[self.sample_indices_batch[bb]] += 1
            with nogil:
                self.update_code(subset, self.X_batch[:len_batch],
                                 self.sample_indices_batch[:len_batch]
                                 )
                self.random_state_.shuffle(self.D_range)

                if self.dict_reduction != 0:
                    subset = self.feature_sampler_2.yield_subset(self.reduction)

                if self.purge_tol > 0 and not self.total_counter_ % reduction_int:
                    self.clean_dict()

                if self.AB_agg != 2:
                    self.update_dict(subset)
                else:
                    self.update_dict_and_B(subset, len_batch)


    cdef void update_dict_and_B(self, int[:] subset, int len_batch) nogil:
        """Parallel dictionary update and B update"""
        cdef double w_A, w_batch, one_m_w

        cdef int subset_step, start_corr, stop_corr, size_corr, t

        cdef int corr_threads = 2 if self.n_threads < 2 else self.n_threads

        if self.n_threads >= 3:
            subset_step = int(ceil(float(self.n_features) / (self.n_threads - 1)))
        else:
            subset_step = self.n_features

        w_A = get_simple_weights(self.total_counter_, len_batch,
                                 self.learning_rate,
                                 self.offset)
        w_batch = w_A / len_batch
        one_m_w = 1 - w_A
        with parallel(num_threads=self.n_threads):
            for t in prange(corr_threads):
                if t == 0:
                    self.update_dict(subset)
                else:
                    start_corr = (t - 1) * subset_step
                    stop_corr = t * subset_step
                    if stop_corr > self.n_features:
                        stop_corr = self.n_features
                    size_corr = stop_corr - start_corr
                    # Hack as X is C-ordered
                    dgemm(&NTRANS, &TRANS,
                          &self.n_components, &size_corr, &len_batch,
                          &w_batch,
                          &self.Dx[0, 0], &self.n_components,
                          &self.X_batch[0, 0] + start_corr, &self.n_features,
                          &one_m_w,
                          &self.B_[0, 0] + start_corr * self.n_components,
                          &self.n_components)

    cdef void clean_dict(self) nogil:
        cdef int k, p
        cdef double sum_A
        cdef double * D_ptr = &self.D_[0, 0]
        cdef double * G_ptr = &self.G_[0, 0]
        cdef int max_k = 0
        for k in range(self.n_components):
            # if self.A_[k, k] > self.A_[max_k, max_k]:
            #     max_k = k
            sum_A += self.A_[k, k]
        for k in range(self.n_components):
            if self.A_[k, k] * self.n_components / sum_A < self.purge_tol:
                printf('+')
                p = self.random_state_.randint(self.n_components - 1)
                p = (k + p) % self.n_components
                for jj in range(self.n_features):
                    self.D_[k, jj] = self.D_[p, jj]
                self.G_[p, k] = self.G_[p, p]
                self.G_[k, :] = self.G_[p, :]
                self.G_[:, k] = self.G_[k, :]
                for q in range(self.n_components):
                    self.A_[p, q] /= 2
                for q in range(self.n_components):
                    self.A_[q, p] /= 2
                self.A_[p, k] = self.A_[p, p]
                self.A_[k, :] = self.A_[p, :]
                self.A_[:, k] = self.A_[k, :]
                for jj in range(self.n_features):
                    self.B_[p, jj] /= 2
                self.B_[k, :] = self.B_[p, :]

    cdef int update_code(self, int[:] subset, double[:, ::1] X,
                         int[:] sample_indices) nogil except *:
        """
        Compute code_ for A_ mini-batch and update algorithm statistics accordingly

        Parameters
        ----------
        sample_indices
        X: masked data matrix

        """
        cdef int len_batch = sample_indices.shape[0]
        cdef int len_subset = subset.shape[0]
        cdef int n_components = self.n_components
        cdef int n_samples = self.n_samples
        cdef int n_features = self.n_features
        cdef double reduction = float(self.n_features) / len_subset
        cdef double* D_subset_ptr = &self.D_subset[0, 0]
        cdef double* D_ptr = &self.D_[0, 0]
        cdef double* A_ptr = &self.A_[0, 0]
        cdef double* B_ptr = &self.B_[0, 0]

        cdef double* G_ptr = &self.G_[0, 0]

        cdef double* Dx_ptr = &self.Dx[0, 0]
        cdef double* G_temp_ptr
        cdef double* this_X_ptr = &self.this_X[0, 0]
        cdef double* X_ptr = &X[0, 0]
        cdef double* G_average_ptr

        cdef int info = 0
        cdef int ii, jj, i, j, k, m, p, q, t, start, stop, size, ii_, i_, \
            p_, q_
        cdef int nnz
        cdef double v
        cdef int last = 0
        cdef double one_m_w, w_sample, w_sample_, w_batch, w_norm, w_A, w_B
        cdef int this_subset_thread_size
        cdef double this_X_norm

        cdef double[::1, :, :] this_G

        cdef int sample_step = int(ceil(float(len_batch)
                                        / self.n_thread_batches))
        cdef int subset_step = int(ceil(float(n_features) / self.n_threads))

        cdef double* R_ptr = &self.R_[0, 0]

        if self.G_agg == 3:
            G_average_ptr = &self.G_average_[0, 0, 0]
        if self.pen_l1_ratio == 0:
            G_temp_ptr = &self.G_temp[0, 0, 0]

        for ii in range(len_batch):
            for jj in range(len_subset):
                j = subset[jj]
                self.this_X[ii, jj] = X[ii, j]

        for k in range(self.n_components):
            for jj in range(len_subset):
                j = subset[jj]
                self.D_subset[k, jj] = self.D_[k, j]
        # Dx computation
        # Dx = np.dot(D_subset, this_X.T)
        if self.Dx_agg == 2:
            # X is C-ordered
            with parallel(num_threads=self.n_threads):
                for t in prange(self.n_threads, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    size = stop - start
                    dgemm(&NTRANS, &NTRANS,
                                  &n_components, &size, &n_features,
                                  &FONE,
                                  D_ptr, &n_components,
                                  X_ptr + start * n_features, &n_features,
                                  &FZERO,
                                  Dx_ptr + start * n_components,
                          &n_components
                                  )
        else:
            with parallel(num_threads=self.n_threads):
                for t in prange(self.n_threads, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    size = stop - start
                    dgemm(&NTRANS, &TRANS,
                          &n_components, &size, &len_subset,
                          &reduction,
                          D_subset_ptr, &n_components,
                          this_X_ptr + start, &len_batch,
                          &FZERO,
                          Dx_ptr + start * n_components, &n_components
                          )
            if self.Dx_agg == 3:
                for ii in range(len_batch):
                    i = sample_indices[ii]
                    w_sample = pow(self.sample_counter_[i],
                                   -self.sample_learning_rate)
                    for p in range(n_components):
                        self.Dx_average_[p, i] *= 1 - w_sample
                        self.Dx_average_[p, i] += self.Dx[p, ii] * w_sample
                        self.Dx[p, ii] = self.Dx_average_[p, i]

        # G_ computation
        if self.G_agg != 2:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &len_subset,
                  &reduction,
                  D_subset_ptr, &n_components,
                  D_subset_ptr, &n_components,
                  &FZERO,
                  G_ptr, &n_components
                  )

        if self.G_agg == 3:
            with parallel(num_threads=self.n_threads):
                for t in prange(self.n_threads, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    size = stop - start
                    for ii_ in range(start, stop):
                        i_ = sample_indices[ii_]
                        w_sample_ = pow(self.sample_counter_[i_],
                                       -self.sample_learning_rate)
                        for p_ in range(n_components):
                            for q_ in range(n_components):
                                self.G_average_[p_, q_, i_] *= (1 - w_sample_)
                                self.G_average_[p_, q_, i_] += self.G_[p_, q_] * w_sample_

        # code_ computation
        if self.pen_l1_ratio == 0:
            if self.G_agg == 3:
                with parallel(num_threads=self.n_threads):
                    for t in prange(self.n_thread_batches, schedule='static'):
                        start = t * sample_step
                        stop = min(len_batch, (t + 1) * sample_step)
                        for ii_ in range(start, stop):
                            i_ = sample_indices[ii_]
                            w_sample_ = pow(self.sample_counter_[i_],
                                            -self.sample_learning_rate)
                            for p_ in range(n_components):
                                for q_ in range(n_components):
                                    self.G_temp[p_, q_, t] = self.G_average_[p_, q_, i_]
                            for p_ in range(n_components):
                                self.G_temp[p_, p_, t] += self.alpha
                            dposv(&UP, &n_components, &ONE, G_temp_ptr
                                  + t * n_components * n_components,
                                  &n_components,
                                  Dx_ptr + ii_ * n_components, &n_components,
                                  &info)
                            for p_ in range(n_components):
                                self.G_temp[p_, p_, t] -= self.alpha
                            if info != 0:
                                with gil:
                                    raise ValueError
            else:
                with parallel(num_threads=self.n_threads):
                    for t in prange(self.n_thread_batches, schedule='static'):
                        for p_ in range(n_components):
                            for q_ in range(n_components):
                                self.G_temp[p_, q_, t] = self.G_[p_, q_]
                            self.G_temp[p_, p_, t] += self.alpha
                        start = t * sample_step
                        stop = min(len_batch, (t + 1) * sample_step)
                        size = stop - start
                        dposv(&UP, &n_components, &size,
                              G_temp_ptr + t * n_components * n_components,
                              &n_components,
                              Dx_ptr + n_components * start, &n_components,
                              &info)
                        if info != 0:
                            with gil:
                                raise ValueError
            for ii in range(len_batch):
                i = sample_indices[ii]
                for k in range(n_components):
                    self.code_[i, k] = self.Dx[k, ii]
        else:
            with parallel(num_threads=self.n_threads):
                for t in prange(self.n_thread_batches, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    for ii_ in range(start, stop):
                        i_ = sample_indices[ii_]
                        if self.Dx_agg == 2:
                            this_X_norm = ddot(&n_features,
                                           X_ptr + ii_ * n_features, &ONE,
                                           X_ptr + ii_ * n_features,
                                           &ONE)
                        else:
                            this_X_norm = ddot(&len_subset,
                                               this_X_ptr + ii_,
                                               &self.batch_size,
                                               this_X_ptr + ii_,
                                               &self.batch_size) * reduction ** 2

                        enet_coordinate_descent_gram(
                            self.code_[i_], self.alpha * self.pen_l1_ratio,
                                          self.alpha * (1 - self.pen_l1_ratio),
                            self.G_ if self.G_agg != 3 else
                            self.G_average_[:, :, i_],
                            self.Dx[:, ii_],
                            this_X_norm,
                            self.H[t],
                            self.XtA[t],
                            100,
                            self.lasso_tol, self.random_state_, 0, self.non_negative_A)
                        for p_ in range(n_components):
                            self.Dx[p_, ii_] = self.code_[i_, p_]

        # Aggregation
        w_A = get_simple_weights(self.total_counter_, len_batch,
                                 self.learning_rate,
                                 self.offset)
        # Dx = this_code
        w_batch = w_A / len_batch
        one_m_w = 1 - w_A
        # A_ *= 1 - w_A
        # A_ += this_code.dot(this_code.T) * w_A / batch_size
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &len_batch,
              &w_batch,
              Dx_ptr, &n_components,
              Dx_ptr, &n_components,
              &one_m_w,
              A_ptr, &n_components
              )

        if self.AB_agg == 1: # Full
            with parallel(num_threads=self.n_threads):
                for t in prange(self.n_threads, schedule='static'):
                    start = t * subset_step
                    stop = min(n_features, (t + 1) * subset_step)
                    size = stop - start
                    # Hack as X is C-ordered
                    dgemm(&NTRANS, &TRANS,
                          &n_components, &size, &len_batch,
                          &w_batch,
                          Dx_ptr, &n_components,
                          X_ptr + start, &n_features,
                          &one_m_w,
                          B_ptr + start * n_components, &n_components)
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    self.R_[k, jj] = self.B_[k, j]
        elif self.AB_agg == 2: # Async
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    self.R_[k, jj] = self.B_[k, j] * one_m_w
                for ii in range(len_batch):
                    self.this_X[ii, jj] *= w_batch
            dgemm(&NTRANS, &NTRANS,
                  &n_components, &len_subset, &len_batch,
                  &FONE,
                  Dx_ptr, &n_components,
                  this_X_ptr, &len_batch,
                  &FONE,
                  R_ptr, &n_components)
        elif self.AB_agg == 3: # Noisy
            for jj in range(len_subset):
                j = subset[jj]
                w_B = fmin(1.,
                           w_A * float(self.total_counter_) /
                           self.feature_counter_[j])
                one_m_w = 1. - w_B
                w_batch = w_B / len_batch
                for k in range(n_components):
                    self.R_[k, jj] = self.B_[k, j] * one_m_w
                for ii in range(len_batch):
                    self.this_X[ii, jj] *= w_batch
            dgemm(&NTRANS, &NTRANS,
                  &n_components, &len_subset, &len_batch,
                  &FONE,
                  Dx_ptr, &n_components,
                  this_X_ptr, &len_batch,
                  &FONE,
                  R_ptr, &n_components)
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    self.B_[k, j] = self.R_[k, jj]
        else:
            w_B = w_A
            one_m_w = 1. - w_B
            w_batch = w_B / len_batch
            for j in range(n_features):
                for k in range(n_components):
                    self.B_[k, j] *= one_m_w
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    self.R_[k, jj] = self.B_[k, j]
                for ii in range(len_batch):
                    self.this_X[ii, jj] *= w_batch * reduction
            dgemm(&NTRANS, &NTRANS,
                  &n_components, &len_subset, &len_batch,
                  &FONE,
                  Dx_ptr, &n_components,
                  this_X_ptr, &len_batch,
                  &FONE,
                  R_ptr, &n_components)
            for jj in range(len_subset):
                j = subset[jj]
                for k in range(n_components):
                    self.B_[k, j] = self.R_[k, jj]

        return 0

    cdef void update_dict(self,
                      int[:] subset) nogil except *:
        cdef int len_subset = subset.shape[0]
        cdef int n_components = self.D_.shape[0]
        cdef int n_features = self.D_.shape[1]
        cdef double* D_ptr = &self.D_[0, 0]
        cdef double* D_subset_ptr = &self.D_subset[0, 0]
        cdef double* A_ptr = &self.A_[0, 0]
        cdef double* R_ptr = &self.R_[0, 0]
        cdef double* G_ptr
        cdef double old_norm = 0
        cdef unsigned long k, kk, j, jj
        cdef double norm_temp = 0

        if self.G_agg == 2:
             G_ptr = &self.G_[0, 0]

        for k in range(n_components):
            for jj in range(len_subset):
                j = subset[jj]
                # self.R_[k, jj] = self.B_[k, j] done in code update
                self.D_subset[k, jj] = self.D_[k, j]

        for kk in range(self.n_components):
            k = self.D_range[kk]
            norm_temp = enet_norm_fast[double](self.D_subset[k, :len_subset],
                                       self.l1_ratio)
            self.norm_temp[k] = norm_temp + 1 - self.norm_[k]
            self.norm_[k] -= norm_temp
        if self.G_agg == 2 and len_subset < self.n_features / 2.:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &len_subset,
                  &FMONE,
                  D_subset_ptr, &n_components,
                  D_subset_ptr, &n_components,
                  &FONE,
                  G_ptr, &n_components
                  )
        # R = B_ - AQ
        dgemm(&NTRANS, &NTRANS,
              &n_components, &len_subset, &n_components,
              &FMONE,
              A_ptr, &n_components,
              D_subset_ptr, &n_components,
              &FONE,
              R_ptr, &n_components)

        for kk in range(self.n_components):
            k = self.D_range[kk]
            dger(&n_components, &len_subset, &FONE,
                 A_ptr + k * n_components,
                 &ONE, D_subset_ptr + k, &n_components, R_ptr, &n_components)

            for jj in range(len_subset):
                if self.A_[k, k] > 1e-20:
                    self.D_subset[k, jj] = self.R_[k, jj] / self.A_[k, k]
            if self.non_negative_D:
                for jj in range(len_subset):
                    if self.D_subset[k, jj] < 0:
                        self.D_subset[k, jj] = 0
            enet_projection_fast[double](self.D_subset[k, :len_subset],
                                    self.proj_temp[:len_subset],
                                    self.norm_temp[k], self.l1_ratio)
            for jj in range(len_subset):
                self.D_subset[k, jj] = self.proj_temp[jj]
            self.norm_[k] += enet_norm_fast[double](self.D_subset[k, :len_subset],
                                       self.l1_ratio)
            # R -= A_[:, k] Q[:, k].T
            dger(&n_components, &len_subset, &FMONE,
                 A_ptr + k * n_components,
                 &ONE, D_subset_ptr + k, &n_components, R_ptr, &n_components)

        for jj in range(len_subset):
            j = subset[jj]
            for k in range(n_components):
                self.D_[k, j] = self.D_subset[k, jj]

        if self.G_agg == 2:
            if len_subset < self.n_features / 2.:
                dgemm(&NTRANS, &TRANS,
                      &n_components, &n_components, &len_subset,
                      &FONE,
                      D_subset_ptr, &n_components,
                      D_subset_ptr, &n_components,
                      &FONE,
                      G_ptr, &n_components
                      )
            else:
                dgemm(&NTRANS, &TRANS,
                      &n_components, &n_components, &self.n_features,
                      &FONE,
                      D_ptr, &n_components,
                      D_ptr, &n_components,
                      &FZERO,
                      G_ptr, &n_components
                      )

    def transform(self, double[:, ::1] X, n_threads=None):
        cdef int n_samples = X.shape[0]
        cdef int i, t, start, stop, p, q, size, ii, jj, p_, q_
        cdef int n_features = self.n_features
        cdef int n_components = self.n_components
        cdef double X_norm

        cdef int this_n_threads
        if n_threads is None:
            this_n_threads = self.n_threads
        else:
            this_n_threads = int(n_threads)

        cdef int n_thread_batches = min(this_n_threads, n_samples)

        cdef double[::1, :] D_ = self.D_
        cdef double[:, ::1] code_ = view.array((n_samples, n_components),
                                              sizeof(double),
                                              format='d', mode='c')
        cdef double[::1, :] G_ = view.array((n_components, n_components),
                                              sizeof(double),
                                              format='d', mode='fortran')
        cdef double[::1, :] Dx = view.array((n_components, n_samples),
                                              sizeof(double),
                                              format='d', mode='fortran')
        Dx[:] = 0

        cdef double[:, ::1] H
        cdef double[:, ::1] XtA
        cdef double[::1, :, :] G_temp

        cdef double* X_ptr = &X[0, 0]
        cdef double* Dx_ptr = &Dx[0, 0]
        cdef double* G_ptr = &G_[0, 0]
        cdef double* D_ptr = &D_[0, 0]
        cdef double* G_temp_ptr

        cdef int sample_step = int(ceil(float(n_samples) / n_thread_batches))
        cdef int component_step = int(ceil(float(n_components) / n_thread_batches))

        cdef int info = 0

        if self.pen_l1_ratio != 0:
            H = view.array((n_thread_batches, n_components),
                  sizeof(double),
                  format='d', mode='c')
            XtA = view.array((n_thread_batches, n_components),
                  sizeof(double),
                  format='d', mode='c')
        else:
            G_temp = view.array((n_components, n_components,
                                 n_thread_batches),
                                  sizeof(double),
                                  format='d', mode='fortran')
            G_temp_ptr = &G_temp[0, 0, 0]
        with nogil, parallel(num_threads=this_n_threads):
           # G_ = D_.dot(D_.T)
            for t in prange(n_thread_batches, schedule='static'):
                start = t * component_step
                stop = min(n_components, (t + 1) * component_step)
                size = stop - start
                dgemm(&NTRANS, &TRANS,
                          &n_components, &size, &n_features,
                          &FONE,
                          D_ptr, &n_components,
                          D_ptr + start, &n_components,
                          &FZERO,
                          G_ptr + start * n_components, &n_components
                          )
        # Dx = D_.dot(X.T)
        # Hack as X is C-ordered
        with nogil, parallel(num_threads=this_n_threads):
            for t in prange(n_thread_batches, schedule='static'):
                start = t * sample_step
                stop = min(n_samples, (t + 1) * sample_step)
                size = stop - start
                dgemm(&NTRANS, &NTRANS,
                      &n_components, &size, &n_features,
                      &FONE,
                      D_ptr, &n_components,
                      X_ptr + start * n_features, &n_features,
                      &FZERO,
                      Dx_ptr + start * n_components, &n_components
                      )
        if self.pen_l1_ratio != 0:
            with nogil, parallel(num_threads=this_n_threads):
                for t in prange(n_thread_batches, schedule='static'):
                    start = t * sample_step
                    stop = min(n_samples, (t + 1) * sample_step)
                    for i in range(start, stop):
                        for p in range(n_components):
                            code_[i, p] = 1
                        X_norm = ddot(&n_features, X_ptr + i * n_features,
                                      &ONE, X_ptr + i * n_features, &ONE)
                        code_[i] = enet_coordinate_descent_gram(
                                    code_[i], self.alpha * self.pen_l1_ratio,
                                                self.alpha * (1 - self.pen_l1_ratio),
                                    G_, Dx[:, i],
                                    X_norm,
                                    H[t],
                                    XtA[t],
                                    100,
                                    self.lasso_tol, self.random_state_, 0, self.non_negative_A)
        else:
            with nogil, parallel(num_threads=this_n_threads):
                for t in prange(n_thread_batches, schedule='static'):
                    for p in range(n_components):
                        for q in range(n_components):
                            G_temp[p, q, t] = G_[p, q]
                        G_temp[p, p, t] += self.alpha
                    start = t * sample_step
                    stop = min(n_samples, (t + 1) * sample_step)
                    size = stop - start
                    dposv(&UP, &n_components, &size,
                          G_temp_ptr + t * n_components * n_components,
                          &n_components,
                          Dx_ptr + n_components * start, &n_components,
                          &info)
                    if info != 0:
                        with gil:
                            raise ValueError
                    for i in range(start, stop):
                        for p in range(n_components):
                            code_[i, p] = Dx[p, i]
        return code_

cdef double[:] enet_coordinate_descent_gram(double[:] w, double alpha,
                                            double beta,
                                 double[::1, :] Q,
                                 double[:] q,
                                 double y_norm2,
                                 double[:] H,
                                 double[:] XtA,
                                 int max_iter, double tol,
                                 RandomState random_state,
                                 bint random, bint positive) nogil:
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm_(w, 1) + (beta/2) * norm_(w, 2)^2

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef int n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    # cdef double[:] XtA = np.zeros(n_features)
    # cdef double[:] H = np.dot(Q, w)

    cdef double tmp
    cdef double w_ii
    cdef double mw_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef unsigned int ii
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter

    cdef double* w_ptr = &w[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* q_ptr = &q[0]
    cdef double* H_ptr = &H[0]
    cdef double* XtA_ptr = &XtA[0]
    cdef double w_norm2
    cdef double const
    cdef double q_dot_w

    tol = tol * y_norm2

    dgemv(&NTRANS,
          &n_features, &n_features,
          &FONE,
          Q_ptr, &n_features,
          w_ptr, &ONE,
          &FZERO,
          H_ptr, &ONE
          )

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            if random:
                ii = random_state.randint(n_features)
            else:
                ii = f_iter

            if Q[ii, ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                # H -= w_ii * Q[ii]
                mw_ii = -w_ii
                daxpy(&n_features, &mw_ii, Q_ptr + ii * n_features, &ONE,
                      H_ptr, &ONE)

            tmp = q[ii] - H[ii]

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (Q[ii, ii] + beta)

            if w[ii] != 0.0:
                # H +=  w[ii] * Q[ii] # Update H = X.T X w
                daxpy(&n_features, &w[ii], Q_ptr + ii * n_features, &ONE,
                      H_ptr, &ONE)

            # update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if fabs(w[ii]) > w_max:
                w_max = fabs(w[ii])

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            # q_dot_w = np.dot(w, q)
            q_dot_w = ddot(&n_features, w_ptr, &ONE, q_ptr, &ONE)

            for ii in range(n_features):
                XtA[ii] = q[ii] - H[ii] - beta * w[ii]
            if positive:
                dual_norm_XtA = pmax(n_features, XtA_ptr)
            else:
                dual_norm_XtA = abs_max(n_features, XtA_ptr)

            # temp = np.sum(w * H)
            tmp = 0.0
            for ii in range(n_features):
                tmp += w[ii] * H[ii]
            R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w

            # w_norm2 = np.dot(w, w)
            w_norm2 = ddot(&n_features, &w[0], &ONE, &w[0], &ONE)

            if dual_norm_XtA > alpha:
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            # The call to dasum is equivalent to the L1 norm_ of w
            gap += (alpha * dasum(&n_features, &w[0], &ONE) -
                    const * y_norm2 + const * q_dot_w +
                    0.5 * beta * (1 + const ** 2) * w_norm2)

            if gap < tol:
                # return if we reached desired tolerance
                break

    return w
