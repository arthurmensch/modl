# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
ctypedef np.uint32_t UINT32_t

cpdef void _get_weights(double[:] w, int[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset)

cpdef double _get_simple_weights(int[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset)

cpdef long _update_code_sparse_batch(double[:] X_data,
                                     int[:] X_indices,
                                     int[:] X_indptr,
                                     int n_rows,
                                     int n_cols,
                                     long[:] row_batch,
                                     long[:] sample_subset,
                                     double alpha,
                                     double learning_rate,
                                     double offset,
                                     long var_red,
                                     long projection,
                                     double[::1, :] D_,
                                     double[:, ::1] code_,
                                     double[::1, :] A_,
                                     double[::1, :] B_,
                                     double[::1, :] G_,
                                     double[::1, :] beta_,
                                     double[:] multiplier_,
                                     long[:] counter_,
                                     long[:] row_counter_,
                                     double[::1, :] D_subset,
                                     double[::1, :] code_temp,
                                     double[::1, :] G_temp,
                                     double[::1, :] _this_X,
                                     double[:] w_temp,
                                     char[:] subset_mask,
                                     int[:] dict_subset,
                                     int[:] dict_subset_lim,
                                     double[::1, :] dummy_2d_float,
                                     ) except *

cpdef void _update_code(double[::1, :] this_X,
                        int[:] subset,
                        long[:] sample_subset,
                        double alpha,
                        double learning_rate,
                        double offset,
                        long var_red,
                        long projection,
                        double[::1, :] D_,
                        double[:, ::1] code_,
                        double[::1, :] A_,
                        double[::1, :] B_,
                        double[::1, :] G_,
                        double[::1, :] beta_,
                        double[:] multiplier_,
                        long[:] counter_,
                        long[:] row_counter_,
                        double[::1, :] full_X,
                        double[::1, :] D_subset,
                        double[::1, :] code_temp,
                        double[::1, :] G_temp,
                        double[:] w_temp) except *

cpdef void _update_dict(double[::1, :] D_,
                  int[:] dict_subset,
                  bint fit_intercept,
                  double l1_ratio,
                  long projection,
                  long var_red,
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