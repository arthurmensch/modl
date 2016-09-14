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
# from posix.time cimport gettimeofday, timeval, timezone, suseconds_t
from posix.time cimport clock_gettime, CLOCK_MONOTONIC_RAW, timespec, time_t

from scipy.linalg.cython_blas cimport dgemm, dger, daxpy, ddot, dasum, dgemv
from scipy.linalg.cython_lapack cimport dposv

# noinspection PyUnresolvedReferences
from ._utils.randomkit.random_fast cimport Sampler, RandomState
# noinspection PyUnresolvedReferences
from ._utils.enet_proj_fast cimport enet_projection_fast, enet_norm_fast, enet_scale_fast

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

cdef double abs_max(int n, double* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef double pmax(int n, double* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
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
    cdef readonly double tol

    cdef readonly int G_agg
    cdef readonly int Dx_agg
    cdef readonly int AB_agg

    cdef readonly int scale_up

    cdef readonly int subset_sampling
    cdef readonly int dict_reduction

    cdef readonly int max_n_iter
    cdef readonly int n_samples
    cdef readonly int n_features
    cdef readonly int n_components
    cdef readonly int len_subset
    cdef readonly int verbose
    cdef readonly int n_threads
    cdef readonly int n_thread_batches

    cdef readonly double[::1, :] D
    cdef readonly double[:, ::1] code
    cdef readonly double[::1, :] A
    cdef readonly double[::1, :] B
    cdef readonly double[::1, :] G

    cdef readonly double[:] norm

    cdef readonly unsigned long random_seed


    cdef readonly double[::1, :, :] G_average
    cdef readonly double[::1, :] Dx_average

    cdef readonly int[:] sample_counter
    cdef readonly int[:] feature_counter
    cdef readonly int total_counter

    cdef double[::1, :] this_X
    cdef double[:, ::1] X_batch
    cdef int[:] sample_indices_batch
    cdef double[::1, :] D_subset

    cdef double[::1, :] Dx
    cdef double[::1, :, :] G_temp
    cdef double[::1, :] R
    cdef double[:] norm_temp

    cdef double[:] proj_temp
    cdef double[:, ::1] H
    cdef double[:, ::1] XtA


    cdef Sampler feature_sampler_1
    cdef Sampler feature_sampler_2
    cdef int[:] D_range

    cdef RandomState random_state

    cdef object callback

    cdef readonly double[:] time

    def __init__(self,
                 double[::1, :] D,
                 int n_samples,
                 double alpha=1.0,
                 double l1_ratio=0.,
                 double pen_l1_ratio=0.,
                 double tol=1e-3,
                 # Hyper-parameters
                 double learning_rate=1.,
                 double sample_learning_rate=0.5,
                 int batch_size=1,
                 double offset=0,
                 # Reduction parameter
                 int reduction=1,
                 int G_agg=1,
                 int Dx_agg=1,
                 int AB_agg=1,
                 bint scale_up=1,
                 int subset_sampling=1,
                 int dict_reduction=0,
                 # Dict parameter
                 # Generic parameters
                 unsigned long random_seed=0,
                 int verbose=0,
                 int n_threads=1,
                 object callback=None):
        cdef int i
        cdef double* G_ptr
        cdef double* D_ptr

        self.n_samples = n_samples
        self.n_components = D.shape[0]
        self.n_features = D.shape[1]

        self.reduction = reduction

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.sample_learning_rate = sample_learning_rate

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pen_l1_ratio = pen_l1_ratio
        self.tol = tol

        self.G_agg = G_agg
        self.Dx_agg = Dx_agg

        self.subset_sampling = subset_sampling
        self.dict_reduction = dict_reduction
        self.AB_agg = AB_agg
        self.scale_up = scale_up

        self.random_seed = random_seed
        self.verbose = verbose

        self.n_threads = n_threads
        self.n_thread_batches = min(n_threads, self.batch_size)

        self.random_state = RandomState(seed=self.random_seed)

        self.D = enet_scale_fast(D, self.l1_ratio, radius=1)

        self.norm = view.array((self.n_components, ),
                                   sizeof(double),
                                   format='d')
        self.norm[:] = 1

        self.code = view.array((self.n_samples, self.n_components),
                               sizeof(double), format='d',
                               mode='c')
        self.code[:] = 1

        self.A = view.array((self.n_components, self.n_components),
                                              sizeof(double),
                                              format='d', mode='fortran')
        self.A[:] = 0
        self.B = view.array((self.n_components, self.n_features),
                                              sizeof(double),
                                              format='d', mode='fortran')
        self.B[:] = 0
        self.G = view.array((self.n_components, self.n_components),
                                          sizeof(double),
                                          format='d', mode='fortran')
        if self.G_agg == 2:
            D_ptr = &self.D[0, 0]
            G_ptr = &self.G[0, 0]
            dgemm(&NTRANS, &TRANS,
                  &self.n_components, &self.n_components, &self.n_features,
                  &FONE,
                  D_ptr, &self.n_components,
                  D_ptr, &self.n_components,
                  &FZERO,
                  G_ptr, &self.n_components
                  )
        elif self.G_agg == 3:
            self.G_average = view.array((self.n_components, self.n_components,
                        self.n_samples), sizeof(double),
                                        format='d', mode='fortran')
        if self.Dx_agg == 3:
            self.Dx_average = view.array((self.n_components, self.n_samples),
                                              sizeof(double),
                                              format='d', mode='fortran')

        self.sample_counter = view.array((self.n_samples, ), sizeof(int),
                                         format='i')
        self.feature_counter = view.array((self.n_features, ), sizeof(int),
                                         format='i')
        self.feature_counter[:] = 0
        self.sample_counter[:] = 0
        self.total_counter = 0

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

        self.R = view.array((self.n_components, self.n_features),
                                   sizeof(double),
                                   format='d',
                                   mode='fortran')
        self.norm_temp = view.array((self.n_components, ), sizeof(double),
                                    format='d')
        self.proj_temp = view.array((self.n_features, ), sizeof(double),
                                    format='d')


        self.D_range = self.random_state.permutation(self.n_components)

        for i in range(self.n_components):
            self.D_range[i] = i

        random_seed = self.random_state.randint()
        self.feature_sampler_1 = Sampler(self.n_features, self.reduction,
                                    self.subset_sampling, random_seed)

        # self.reduction_ = np.mean(np.maximum(np.random.geometric(self.reduction
        #                                                          / self.n_features, 1000),
        #                                      self.n_features))
        if self.dict_reduction != 0:
            random_seed = self.random_state.randint()
            self.feature_sampler_2 = Sampler(self.n_features,
                                             self.dict_reduction,
                                        self.subset_sampling, random_seed)

        self.callback = callback

        self.time = view.array((6, ), sizeof(double),
                               format='d')
        self.time[:] = 0

    cpdef void set_impl_params(self,
                               double alpha=1.0,
                               double l1_ratio=0.,
                               double pen_l1_ratio=0.,
                               double tol=1e-3,
                               # Hyper-parameters
                               double learning_rate=1.,
                               double sample_learning_rate=0.5,
                               int batch_size=1,
                               double offset=0,
                               # Reduction parameter
                               int reduction=1,
                               int G_agg=1,
                               int Dx_agg=1,
                               int AB_agg=1,
                               int subset_sampling=1,
                               int dict_reduction=0,
                               # Dict parameter
                               # Generic parameters
                               int verbose=0,
                               object callback=None):
        cdef int old_len_subset
        cdef int old_G_agg
        cdef double* G_ptr
        cdef double* D_ptr

        self.AB_agg = AB_agg
        self.subset_sampling = subset_sampling
        self.dict_reduction = dict_reduction
        old_G_agg = self.G_agg
        self.G_agg = G_agg
        self.Dx_agg = Dx_agg

        if self.G_agg == 2 and old_G_agg != 2:
            D_ptr = &self.D[0, 0]
            G_ptr = &self.G[0, 0]
            dgemm(&NTRANS, &TRANS,
                  &self.n_components, &self.n_components, &self.n_features,
                  &FONE,
                  D_ptr, &self.n_components,
                  D_ptr, &self.n_components,
                  &FZERO,
                  G_ptr, &self.n_components
                  )

        if self.G_agg != 3:
            self.G_average = None
        elif self.Dx_agg == 3 and self.G_average is None:
            self.G_average = view.array((self.n_components, self.n_components,
                        self.n_samples), sizeof(double),
                                        format='d', mode='fortran')

        if self.Dx_agg != 3:
            self.Dx_average = None
        elif self.Dx_agg == 3 and self.Dx_average is None:
            self.Dx_average = view.array((self.n_components, self.n_samples),
                                              sizeof(double),
                                              format='d', mode='fortran')
        if self.pen_l1_ratio != 0:
            self.G_temp = None
        elif self.pen_l1_ratio == 0 and self.G_temp is None:
            self.G_temp = view.array((self.n_components, self.n_components),
                                              sizeof(double),
                                              format='d', mode='fortran')
        if self.pen_l1_ratio == 0:
            self.H = None
            self.XtA = None
        elif self.pen_l1_ratio != 0 and self.H is None:
            self.H = view.array((self.n_thread_batches, self.n_components),
                  sizeof(double),
                  format='d', mode='c')
            self.XtA = view.array((self.n_thread_batches, self.n_components),
                  sizeof(double),
                  format='d', mode='c')

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pen_l1_ratio = pen_l1_ratio
        self.tol = tol

        self.reduction = reduction
        self.feature_sampler_1.reduction = self.reduction

        if self.dict_reduction != 0:
            if self.feature_sampler_2 is None:
                random_seed = self.random_state.randint()
                self.feature_sampler_2 = Sampler(self.n_features,
                                 self.dict_reduction,
                            self.subset_sampling, random_seed)
            else:
                self.feature_sampler_2.reduction = self.dict_reduction
        else:
            self.feature_sampler_2 = None
        self.learning_rate = learning_rate
        self.sample_learning_rate = sample_learning_rate
        self.offset = offset

        self.verbose = verbose

        self.callback = callback

    cpdef void partial_fit(self, double[:, ::1] X, int[:] sample_indices,
                           ) except *:
        cdef int this_n_samples = X.shape[0]
        cdef int n_batches = int(ceil(this_n_samples / self.batch_size))
        cdef int start = 0
        cdef int stop = 0
        cdef int len_batch = 0

        cdef int old_total_counter = self.total_counter
        cdef int new_verbose_iter = 0

        cdef int i, ii, jj, bb, j, k, t

        cdef int[:] subset

        cdef int[:] random_order = self.random_state.permutation(this_n_samples)

        cdef timespec tv0, tv1

        with nogil:
            for k in range(n_batches):
                if self.verbose and self.total_counter\
                        - old_total_counter >= new_verbose_iter:
                    printf("Iteration %i\n", self.total_counter)
                    new_verbose_iter += this_n_samples / self.verbose
                    if self.callback is not None:
                        with gil:
                            self.callback()
                clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
                start = k * self.batch_size
                stop = start + self.batch_size
                if stop > this_n_samples:
                    stop = this_n_samples
                len_batch = stop - start

                self.total_counter += len_batch

                subset = self.feature_sampler_1.yield_subset()
                for jj in range(subset.shape[0]):
                    j = subset[jj]
                    self.feature_counter[j] += len_batch

                for bb, ii in enumerate(range(start, stop)):
                    self.sample_indices_batch[bb] =\
                        sample_indices[random_order[ii]]
                    self.X_batch[bb] = X[random_order[ii]]
                    self.sample_counter[self.sample_indices_batch[bb]] += 1

                self.update_code(subset, self.X_batch[:len_batch],
                                 self.sample_indices_batch[:len_batch])
                self.random_state.shuffle(self.D_range)

                if self.dict_reduction != 0:
                    subset = self.feature_sampler_2.yield_subset()
                self.update_dict(subset)
                clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
                self.time[5] += tv1.tv_sec-tv0.tv_sec
                self.time[5] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

    cpdef double[::1, :] scaled_D(self):
        if self.scale_up:
            return self.D
        else:
            return enet_scale_fast(self.D, self.l1_ratio, radius=1)

    cdef int update_code(self, int[:] subset, double[:, ::1] X,
                         int[:] sample_indices) nogil except *:
        """
        Compute code for a mini-batch and update algorithm statistics accordingly

        Parameters
        ----------
        sample_indices
        X: masked data matrix
        this_subset: indices (loci) of masked data
        alpha: regularization parameter
        learning_rate: decrease rate in the learning sequence (in [.5, 1])
        offset: offset in the learning se   quence
        D_: Dictionary
        A_: algorithm variable
        B_: algorithm variable
        counter_: algorithm variable
        G: algorithm variable
        T: algorithm variable
        impute: Online update of Gram matrix
        D_subset : Temporary array. Holds the subdictionary
        Dx: Temporary array. Holds the codes for the mini batch
        G_temp: emporary array. Holds the Gram matrix.
        subset_mask: Holds the binary mask for visited features
        weights: Temporary array. Holds the update weights

        """
        cdef int len_batch = sample_indices.shape[0]
        cdef int len_subset = subset.shape[0]
        cdef int n_components = self.n_components
        cdef int n_samples = self.n_samples
        cdef int n_features = self.n_features
        cdef double reduction = float(self.n_features) / len_subset
        cdef double* D_subset_ptr = &self.D_subset[0, 0]
        cdef double* D_ptr = &self.D[0, 0]
        cdef double* A_ptr = &self.A[0, 0]
        cdef double* B_ptr = &self.B[0, 0]

        cdef double* G_ptr = &self.G[0, 0]

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

        cdef int sample_step = int(ceil(float(len_batch)
                                        / self.n_thread_batches))
        cdef int subset_step = int(ceil(float(n_features) / self.n_threads))

        cdef timespec tv0, tv1

        if self.G_agg == 3:
            G_average_ptr = &self.G_average[0, 0, 0]
        if self.pen_l1_ratio == 0:
            G_temp_ptr = &self.G_temp[0, 0, 0]

        with parallel(num_threads=self.n_threads):
            for ii in range(len_batch):
                for jj in range(len_subset):
                    j = subset[jj]
                    self.this_X[ii, jj] = X[ii, j]

            for jj in range(len_subset):
                j = subset[jj]
                for k in range(self.n_components):
                    self.D_subset[k, jj] = self.D[k, j]
            for ii in range(len_batch):
                for jj in range(len_subset):
                    self.this_X[ii, jj] *= reduction

            # Dx computation
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
            # Dx = np.dot(D_subset, this_X.T)
            if self.Dx_agg == 2:
                # X is C-ordered
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
                for t in prange(self.n_threads, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    size = stop - start
                    dgemm(&NTRANS, &TRANS,
                          &n_components, &size, &len_subset,
                          &FONE,
                          D_subset_ptr, &n_components,
                          this_X_ptr + start, &len_batch,
                          &FZERO,
                          Dx_ptr + start * n_components, &n_components
                          )
                if self.Dx_agg == 3:
                    for ii in range(len_batch):
                        i = sample_indices[ii]
                        w_sample = pow(self.sample_counter[i],
                                       -self.sample_learning_rate)
                        for p in range(n_components):
                            self.Dx_average[p, i] *= 1 - w_sample
                            self.Dx_average[p, i] += self.Dx[p, ii] * w_sample
                            self.Dx[p, ii] = self.Dx_average[p, i]
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
            self.time[0] += tv1.tv_sec-tv0.tv_sec
            self.time[0] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

            # G computation
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
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
                for t in prange(self.n_threads, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    size = stop - start
                    for ii_ in range(start, stop):
                        i_ = sample_indices[ii_]
                        w_sample_ = pow(self.sample_counter[i_],
                                       -self.sample_learning_rate)
                        for p_ in range(n_components):
                            for q_ in range(n_components):
                                self.G_average[p_, q_, i_] *= 1 - w_sample_
                                self.G_average[p_, q_, i_] += self.G[p_, q_]\
                                                           * w_sample_
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
            self.time[1] += tv1.tv_sec-tv0.tv_sec
            self.time[1] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

            # code computation
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
            if self.pen_l1_ratio == 0:
                if self.G_agg == 3:
                    for t in prange(self.n_thread_batches, schedule='static'):
                        start = t * sample_step
                        stop = min(len_batch, (t + 1) * sample_step)
                        for ii_ in range(start, stop):
                            i_ = sample_indices[ii_]
                            for p_ in range(n_components):
                                for q_ in range(n_components):
                                    self.G_temp[p_, q_, t] = self.G_average[p_, q_,
                                                                          i_]
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
                    for t in prange(self.n_thread_batches, schedule='static'):
                        for p_ in range(n_components):
                            for q_ in range(n_components):
                                self.G_temp[p_, q_, t] = self.G[p_, q_]
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
                        self.code[i, k] = self.Dx[k, ii]
            else:
                for t in prange(self.n_thread_batches, schedule='static'):
                    start = t * sample_step
                    stop = min(len_batch, (t + 1) * sample_step)
                    for ii_ in range(start, stop):
                        i_ = sample_indices[ii_]
                        this_X_norm = ddot(&len_subset,
                                           this_X_ptr + ii_ * len_subset, &ONE,
                                           this_X_ptr + ii_ * len_subset,
                                           &ONE) * reduction
                        enet_coordinate_descent_gram(
                            self.code[i_], self.alpha * self.pen_l1_ratio,
                                          self.alpha * (1 - self.pen_l1_ratio),
                            self.G_average[:, :, i_] if self.G_agg == 3
                            else self.G,
                            self.Dx[:, ii_],
                            this_X_norm,
                            self.H[t],
                            self.XtA[t],
                            1000,
                            self.tol, self.random_state, 0, 0)
                        for p_ in range(n_components):
                            self.Dx[p_, ii_] = self.code[i_, p_]
            for ii in range(len_batch):
                for jj in range(len_subset):
                    self.this_X[ii, jj] /= reduction
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
            self.time[2] += tv1.tv_sec-tv0.tv_sec
            self.time[2] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

            # Aggregation
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
            w_A = get_simple_weights(self.total_counter, len_batch,
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
            if self.AB_agg == 1: # Masked
                w_batch = reduction * w_A / len_batch
                for jj in range(n_features):
                    for k in range(n_components):
                        self.B[k, jj] *= 1 - w_A
                dgemm(&NTRANS, &NTRANS,
                      &n_components, &len_subset, &len_batch,
                      &w_batch,
                      Dx_ptr, &n_components,
                      this_X_ptr, &len_batch,
                      &FZERO,
                      D_subset_ptr, &n_components)
                for jj in range(len_subset):
                    j = subset[jj]
                    for k in range(n_components):
                        self.B[k, j] += self.D_subset[k, jj]
                # Reuse D_subset as B_subset
                # B += this_X.T.dot(P[row_batch]) * {w_B} / batch_size
            elif self.AB_agg == 2: # Full
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
            else: # Async
                for jj in range(len_subset):
                    j = subset[jj]
                    # w_B = get_simple_weights(self.feature_counter[j], len_batch,
                    #              self.learning_rate,
                    #              self.offset)
                    w_B = fmin(1., w_A * float(self.total_counter) /
                               self.feature_counter[j])
                    one_m_w = 1. - w_B
                    w_batch = w_B / len_batch
                    for k in range(n_components):
                        self.D_subset[k, jj] = self.B[k, j] * one_m_w
                    for ii in range(len_batch):
                        self.this_X[ii, jj] *= w_batch
                dgemm(&NTRANS, &NTRANS,
                      &n_components, &len_subset, &len_batch,
                      &FONE,
                      Dx_ptr, &n_components,
                      this_X_ptr, &len_batch,
                      &FONE,
                      D_subset_ptr, &n_components)
                for jj in range(len_subset):
                    j = subset[jj]
                    for k in range(n_components):
                        self.B[k, j] = self.D_subset[k, jj]
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
            self.time[3] += tv1.tv_sec-tv0.tv_sec
            self.time[3] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9
        return 0

    cdef void update_dict(self,
                      int[:] subset) nogil except *:
        cdef int len_subset = subset.shape[0]
        cdef int n_components = self.D.shape[0]
        cdef int n_cols = self.D.shape[1]
        cdef double* D_ptr = &self.D[0, 0]
        cdef double* D_subset_ptr = &self.D_subset[0, 0]
        cdef double* A_ptr = &self.A[0, 0]
        cdef double* R_ptr = &self.R[0, 0]
        cdef double* G_ptr
        cdef double old_norm = 0
        cdef unsigned long k, kk, j, jj
        cdef double norm_temp = 0

        cdef timespec tv0, tv1

        if self.G_agg == 2:
             G_ptr = &self.G[0, 0]

        for k in range(n_components):
            for jj in range(len_subset):
                j = subset[jj]
                self.D_subset[k, jj] = self.D[k, j]
                self.R[k, jj] = self.B[k, j]

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
        for kk in range(self.n_components):
            k = self.D_range[kk]
            norm_temp = enet_norm_fast(self.D_subset[k, :len_subset],
                                       self.l1_ratio)
            if self.scale_up:
                self.norm_temp[k] = norm_temp + 1 - self.norm[k]
                self.norm[k] -= norm_temp
            else:
                self.norm_temp[k] = norm_temp
        if self.G_agg == 2 and len_subset < self.n_features / 2.:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &len_subset,
                  &FMONE,
                  D_subset_ptr, &n_components,
                  D_subset_ptr, &n_components,
                  &FONE,
                  G_ptr, &n_components
                  )
        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
        self.time[1] += tv1.tv_sec-tv0.tv_sec
        self.time[1] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
        # R = B - AQ
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
                if self.A[k, k] > 1e-20:
                    self.D_subset[k, jj] = self.R[k, jj] / self.A[k, k]
                    # print(D_subset[k, jj])

            enet_projection_fast(self.D_subset[k, :len_subset],
                                    self.proj_temp[:len_subset],
                                    self.norm_temp[k], self.l1_ratio)
            for jj in range(len_subset):
                self.D_subset[k, jj] = self.proj_temp[jj]
            if self.scale_up:
                self.norm[k] += enet_norm_fast(self.D_subset[k, :len_subset],
                                           self.l1_ratio)
            # R -= A[:, k] Q[:, k].T
            dger(&n_components, &len_subset, &FMONE,
                 A_ptr + k * n_components,
                 &ONE, D_subset_ptr + k, &n_components, R_ptr, &n_components)

        for jj in range(len_subset):
            j = subset[jj]
            for kk in range(n_components):
                self.D[kk, j] = self.D_subset[kk, jj]

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
        self.time[4] += tv1.tv_sec-tv0.tv_sec
        self.time[4] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv0)
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
        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1)
        self.time[1] += tv1.tv_sec-tv0.tv_sec
        self.time[1] += (tv1.tv_nsec - tv0.tv_nsec) / 1e9

    def transform(self, double[:, ::1] X, n_threads=None):
        cdef int n_samples = X.shape[0]
        cdef int i, t, start, stop, p, q, size, ii
        cdef int n_features = self.n_features
        cdef int n_components = self.n_components
        cdef double X_norm

        cdef int this_n_threads
        if n_threads is None:
            this_n_threads = self.n_threads
        else:
            this_n_threads = int(n_threads)

        cdef int n_thread_batches = min(this_n_threads, n_samples)

        cdef double[::1, :] D = self.scaled_D()
        cdef double[:, ::1] code = view.array((n_samples, n_components),
                                              sizeof(double),
                                              format='d', mode='c')
        cdef double[::1, :] G = view.array((n_components, n_components),
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
        cdef double* G_ptr = &G[0, 0]
        cdef double* D_ptr = &D[0, 0]
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
           # G = D.dot(D.T)
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
        # Dx = D.dot(X.T)
        # Hack as X is C-ordered
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
                for t in prange(n_thread_batches, schedule='static'):
                    start = t * sample_step
                    stop = min(n_samples, (t + 1) * sample_step)
                    for i in range(start, stop):
                        for p in range(n_components):
                            code[i, p] = 1
                        X_norm = ddot(&n_features, X_ptr + i * n_features,
                                      &ONE, X_ptr + i * n_features, &ONE)
                        code[i] = enet_coordinate_descent_gram(
                                    code[i], self.alpha * self.pen_l1_ratio,
                                                self.alpha * (1 - self.pen_l1_ratio),
                                    G, Dx[:, i],
                                    X_norm,
                                    H[t],
                                    XtA[t],
                                    1000,
                                    self.tol, self.random_state, 0, 0)
            else:
                for t in prange(n_thread_batches, schedule='static'):
                    for p in range(n_components):
                        for q in range(n_components):
                            G_temp[p, q, t] = G[p, q]
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
                    for ii in range(start, stop):
                        for p in range(n_components):
                            code[ii, p] = Dx[p, ii]
        return code, D

cdef double[:] enet_coordinate_descent_gram(double[:] w, double alpha, double beta,
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

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

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

            # The call to dasum is equivalent to the L1 norm of w
            gap += (alpha * dasum(&n_features, &w[0], &ONE) -
                    const * y_norm2 + const * q_dot_w +
                    0.5 * beta * (1 + const ** 2) * w_norm2)

            if gap < tol:
                # return if we reached desired tolerance
                break

    return w
