import numpy as np
from .tensor import Tensor
from .types import float32, int64, convert_numpy_array_to_non_native_type


def manual_seed(seed):
    np.random.seed(seed)


def rand(*size, **kwargs):
    dtype = kwargs.get('dtype', float32)
    device = kwargs.get('device', 'cpu')
    data = np.random.rand(*size)
    return Tensor(data).to(dtype).to(device)


def randn(*size, **kwargs):
    dtype = kwargs.get('dtype', float32)
    device = kwargs.get('device', 'cpu')
    data = np.random.randn(*size)
    return Tensor(data).to(dtype).to(device)


def randint(*args, **kwargs):
    if len(args) == 2:
        low = 0
        high = args[0]
        size = args[1]
    else:
        assert len(args) == 3
        low = args[0]
        high = args[1]
        size = args[2]
    dtype = kwargs.get('dtype', int64)
    assert dtype.is_int_type()
    assert dtype.is_native_numpy_type()
    device = kwargs.get('device', 'cpu')
    data = np.random.randint(low, high, size, dtype=dtype.numpy_type)
    return Tensor(data).to(device)
