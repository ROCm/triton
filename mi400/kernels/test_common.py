import torch
import numpy as np


# Testing
def allclose_numpy(input_array, other_array, rtol=1e-05, atol=1e-08, equal_nan=False):
    if input_array.shape != other_array.shape:
        return False

    if equal_nan:
        are_same = np.isclose(input_array, other_array, rtol=rtol, atol=atol, equal_nan=True)
    else:
        are_same = np.isclose(input_array, other_array, rtol=rtol, atol=atol, equal_nan=False)

    return np.all(are_same)
