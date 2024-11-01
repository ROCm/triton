from hip import hip
import numpy as np
from .device import normalize_device
from .hip_common import hip_check
from .kernels import binary_op_kernel, unary_op_kernel, conversion_kernel, reduction_kernel
from .types import Dtype, float32, numpy_type_to_torch_type, \
    convert_numpy_array_to_non_native_type, convert_non_native_type_to_numpy_array


class Tensor:

    def __init__(self, data, *, dtype=None, shape=None, device=None):
        device = normalize_device(device)
        self.device = device
        data_d = None
        if dtype is not None:
            assert isinstance(dtype, Dtype)
        if isinstance(data, list) or isinstance(data, int) or isinstance(data, float) or \
                isinstance(data, np.ndarray) or isinstance(data, np.number):
            if isinstance(data, np.ndarray) or isinstance(data, np.number):
                if dtype is not None:
                    assert dtype.numpy_type == data.dtype
            else:
                assert dtype is None
                data = np.array(data, dtype=np.float32)
            if dtype is None:
                self.dtype = numpy_type_to_torch_type(data.dtype)
            else:
                self.dtype = dtype
            assert shape is None or data.shape == shape
            self.shape = data.shape
        else:
            assert hasattr(data, 'as_c_void_p')
            assert device.type == 'cuda'
            assert dtype is not None
            assert shape is not None
            self.dtype = dtype
            self.shape = shape
            data_d = data
        if device.type == 'cuda':
            if data_d is None:
                data_d = hip_check(hip.hipMalloc(data.size * data.itemsize))
                hip_check(hip.hipMemcpyHtoD(data_d, data, data_d.size))
            self.data_d = data_d
            self.data_h = None
        else:
            self.data_d = None
            self.data_h = data

    def __del__(self):
        if self.data_d is not None:
            hip_check(hip.hipFree(self.data_d))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.data_h is not None:
            data_h = self.data_h
        else:
            data_h = self.cpu().data_h
        if not self.dtype.is_native_numpy_type():
            data_h = convert_non_native_type_to_numpy_array(data_h, self.dtype)
        if self.device.type == "cpu":
            return f"tensor({data_h})"
        else:
            return f"tensor({data_h}, device='{self.device}')"

    def to(self, other_type):
        if not isinstance(other_type, Dtype):
            device = normalize_device(other_type)
            if device.type == "cpu":
                return self.cpu()
            return self.cuda()

        if other_type == self.dtype:
            return self
        if self.data_d is not None:
            return conversion_kernel(self, other_type)
        if self.dtype.is_native_numpy_type():
            if other_type.is_native_numpy_type():
                return Tensor(self.data_h.astype(other_type.numpy_type))
            data = convert_numpy_array_to_non_native_type(self.data_h, other_type)
            return Tensor(data, dtype=other_type)
        if other_type.is_native_numpy_type():
            data = convert_non_native_type_to_numpy_array(self.data_h, self.dtype)
            return Tensor(data.astype(other_type.numpy_type))
        raise NotImplementedError(f"Cannot convert between {self.dtype} and {other_type}")

    def cpu(self):
        if self.data_h is not None:
            return self
        data_h = np.empty(self.shape, dtype=self.dtype.numpy_type)
        hip_check(hip.hipMemcpyDtoH(data_h, self.data_d, self.data_d.size))
        return Tensor(data_h, dtype=self.dtype)

    def cuda(self):
        if self.data_d is not None:
            return self
        return Tensor(self.data_h, dtype=self.dtype, device='cuda')

    def numpy(self):
        assert self.data_h is not None, "can't convert device tensor to numpy"
        assert self.dtype.is_native_numpy_type(), "can't convert {self.dtype} to numpy"
        return self.data_h

    def stride(self, dim_idx=None):
        strides = []
        acc = 1
        if len(self.shape) > 0:
            strides.append(acc)
        # Iterate over all dims in reverse order, excluding the first
        for dim in self.shape[:0:-1]:
            acc *= dim
            strides.append(acc)
        strides.reverse()
        if dim_idx is None:
            return tuple(strides)
        return strides[dim_idx]

    def size(self):
        return self.shape

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def data_ptr(self):
        if self.data_h is not None:
            ptr, read_only = self.data_h.__array_interface__['data']
            assert not read_only
            return ptr
        return self.data_d.as_c_void_p().value

    def __add__(self, other):
        return binary_op_kernel(self, other, "add", "in1 + in2", lambda in1, in2: in1 + in2)

    def __sub__(self, other):
        return binary_op_kernel(self, other, "sub", "in1 - in2", lambda in1, in2: in1 - in2)

    def __mul__(self, other):
        return binary_op_kernel(self, other, "mul", "in1 * in2", lambda in1, in2: in1 * in2)

    def __getitem__(self, key):
        if self.data_h is not None:
            return Tensor(self.data_h[key], dtype=self.dtype)
        data_h = self.cpu().data_h
        return Tensor(data_h[key], dtype=self.dtype, device=self.device)


def zeros(*size, **kwargs):
    size = size if type(size[0]) is int else size[0]
    dtype = kwargs.get('dtype', float32)
    device = kwargs.get('device')
    data = np.zeros(size, dtype=dtype.numpy_type)
    return Tensor(data, dtype=dtype, device=device)


def ones(*size, **kwargs):
    size = size if type(size[0]) is int else size[0]
    dtype = kwargs.get('dtype', float32)
    device = kwargs.get('device')
    data = np.ones(size, dtype=dtype.numpy_type)
    return Tensor(data, dtype=dtype, device=device)


def empty(*size, **kwargs):
    size = size if type(size[0]) is int else size[0]
    dtype = kwargs.get('dtype', float32)
    device = kwargs.get('device')
    data = np.empty(size, dtype=dtype.numpy_type)
    return Tensor(data, dtype=dtype, device=device)


def empty_like(x, **kwargs):
    dtype = kwargs.get('dtype', x.dtype)
    device = kwargs.get('device', x.device)
    data = np.empty(x.shape, dtype=dtype.numpy_type)
    return Tensor(data, dtype=dtype, device=device)


def abs_(x):
    return unary_op_kernel(x, "abs", "abs(in1)", lambda in1: np.abs(in1))


def exp(x):
    return unary_op_kernel(x, "exp", "exp(in1)", lambda in1: np.exp(in1))


def max_(x):
    return reduction_kernel(x, lambda in1: np.max(in1))
