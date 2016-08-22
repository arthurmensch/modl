# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
ctypedef np.uint32_t UINT32_t

cpdef void _get_weights(double[:] w, int[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset)

cpdef void _update_code(double[::1, :] this_X,
                        int[:] subset,
                        long[:] sample_subset,
                        double alpha,
                        double pen_l1_ratio,
                        double learning_rate,
                        double offset,
                        bint present_boost,
                        long projection,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[::1, :] beta_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] D_subset,
                        double[::1, :] code_temp,
                        double[::1, :] G_temp,
                        double[:] w_temp,
                        np.ndarray[double, ndim=2, mode='c'] _beta_temp,
                        object rng) except *

cpdef void _update_dict(double[::1, :] D_,
                  int[:] dict_subset,
                  bint fit_intercept,
                  double l1_ratio,
                  long projection,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  long[:] _D_range,
                  double[::1, :] _R,
                  double[::1, :] _D_subset,
                  double[:] _norm_temp,
                  double[:] _proj_temp)

cpdef void _predict(double[:] X_data,
             int[:] X_indices,
             int[:] X_indptr,
             double[:, ::1] P,
             double[::1, :] Q)

cpdef void _update_subset(bint replacement,
                   long _len_subset,
                   int[:] _subset_range,
                   int[:] _subset_lim,
                   int[:] _temp_subset,
                   UINT32_t random_seed)