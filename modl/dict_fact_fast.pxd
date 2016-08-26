# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# noinspection PyUnresolvedReferences
cimport numpy as np
ctypedef np.uint32_t UINT32_t

cdef void _update_subset_(bint replacement,
                   long _len_subset,
                   long[:] _subset_range,
                   long[:] _subset_lim,
                   long[:] _temp_subset,
                   UINT32_t* random_seed) nogil

cpdef double _get_simple_weights(long count, long batch_size,
           double learning_rate, double offset) nogil

cpdef void _get_weights(double[:] w, long[:] subset, long[:] counter,
                        long batch_size, double learning_rate,
                        double offset) nogil

cdef void enet_coordinate_descent_gram_(double[:] w, double alpha, double beta,
                                 double[::1, :] Q,
                                 double[:] q,
                                 double[:] y,
                                 double[:] H,
                                 double[:] XtA,
                                 int max_iter, double tol, UINT32_t* random_seed,
                                 bint random, bint positive) nogil

cdef int _update_code(double[::1, :] full_X,
                        long[:] subset,
                        long[:] this_sample_subset,
                        double alpha,
                        double pen_l1_ratio,
                        double tol,
                        double learning_rate,
                        double sample_learning_rate,
                        double offset,
                        long solver,
                        long weights,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[:, :] Dx_average_,
                        double[::1, :, :] G_average_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] D_subset,
                        double[::1, :] Dx,
                        double[::1, :] G_temp,
                        double[::1, :] this_X,
                        double[:, ::1] H,
                        double[:, ::1] XtA,
                        UINT32_t* random_seed,
                        int num_threads) nogil

cdef void _update_dict(double[::1, :] D_,
                  long[:] subset,
                  double l1_ratio,
                  long solver,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  long[:] D_range,
                  double[::1, :] R,
                  double[::1, :] D_subset,
                  double[:] norm_temp,
                  double[:] proj_temp) nogil