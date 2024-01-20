import numpy as np
from numba import njit

@njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Only valid for 2D arrays.
    """
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def nb_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@njit
def nb_median(array, axis):
  return np_apply_along_axis(np.median, axis, array)

@njit
def avg_of_medians(X, label, label_categories):
    medians = np.empty((len(label_categories), X.shape[1]))
    for i, l in enumerate(label_categories):
        medians[i] = nb_median(X[label == l], axis=0)
    return nb_mean(medians, axis=0)
