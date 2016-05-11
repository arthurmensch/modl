# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cpdef void _get_weights(double[:] w, long[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset)

cpdef double _get_simple_weights(long[:] subset, long[:] counter, long batch_size,
           double learning_rate, double offset)

cpdef void _update_code(double[::1, :] X,
                        long[:] subset,
                        long[:] sample_subset,
                        double alpha,
                        double learning_rate,
                        double offset,
                        long var_red,
                        long projection,
                        double reduction,
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
                        double[::1, :] this_X,
                        double[::1, :] this_code,
                        double[::1, :] this_G,
                        double[:] w_arr) except *

cpdef void _update_dict(double[::1, :] D_,
                  long[:] dict_subset,
                  bint fit_intercept,
                  double l1_ratio,
                  long projection,
                  long var_red,
                  long[:] D_range,
                  double[::1, :] A_,
                  double[::1, :] B_,
                  double[::1, :] G_,
                  double[::1, :] R,
                  double[::1, :] D_subset,
                  double[:] norm,
                  double[:] buffer)

cpdef void _predict(double[:] X_data,
             int[:] X_indices,
             int[:] X_indptr,
             double[:, ::1] P,
             double[::1, :] Q)