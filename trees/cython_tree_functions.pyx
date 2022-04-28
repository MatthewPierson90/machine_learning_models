# cython:language_level=3
import cython as c
import numpy as np
cimport numpy as np

np.import_array()

cdef float new_avg(double old_avg,
                   double new_value,
                   long old_num,
                   long add_remove):
    return (old_num*old_avg+add_remove*new_value)/(old_num+add_remove)

def find_regression_feature_split(np.ndarray[c.double, ndim=1] x_values,
                                  np.ndarray[c.double, ndim=1] y_values,
                                  c.int min_points_in_split,
                                  c.int num_values,
                                  c.float min_error
                                  ) -> (c.double,c.double):
    cdef long index = 0
    cdef long num_avg_1 = 1
    cdef long num_avg_2 = num_values
    cdef double x_value = x_values[0]
    cdef double y_value = y_values[0]
    cdef double avg_1 = 0.
    cdef double avg_2 = 0.
    cdef double sum1 = 0.
    cdef double sum2 = 0.
    cdef double error = min_error
    cdef double min_split = x_value
    for index in range(num_values-1):
        if index == 0:
            x_value = x_values[index]
            y_value = y_values[index]
            min_split = x_value
            avg_1 = y_value
            avg_2 = y_values[index+1:].mean()
            num_avg_1 = 1
            num_avg_2 = num_values - 1
        else:
            x_value = x_values[index]
            y_value = y_values[index]
            avg_1 = new_avg(avg_1, y_value, num_avg_1, 1)
            avg_2 = new_avg(avg_2, y_value, num_avg_2, -1)
            num_avg_1 += 1
            num_avg_2 -= 1
        if index+1 < min_points_in_split or num_values-index-1 < min_points_in_split:
            continue
        elif index != num_values - 1:
            if x_values[index + 1] == x_value:
                continue
            else:
                sum1 = ((y_values[:index+1]-avg_1)**2).sum()
                sum2 = ((y_values[index+1:]-avg_2)**2).sum()
                error = sum1+sum2
                if error < min_error:
                    min_error = error
                    min_split = (x_value+x_values[index+1])/2
    return min_error, min_split
