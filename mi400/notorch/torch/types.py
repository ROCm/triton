import ctypes
import numpy as np


class Dtype:

    def __init__(self, bit_width, name, hip_type_str, numpy_type):
        self.bit_width = bit_width
        self.name = name
        self.hip_type_str = hip_type_str
        self.numpy_type = numpy_type
        self._is_native_numpy_type = not (self.name.startswith("bfloat") or self.name.startswith("float8"))
        self._is_native_clang_type = not self.hip_type_str.startswith("__hip")
        self._is_int_type = self.name.startswith("int") or self.name.startswith("uint")
        self._is_fp8_type = self.name.startswith("float8")

    def is_native_numpy_type(self):
        return self._is_native_numpy_type

    def is_native_clang_type(self):
        return self._is_native_clang_type

    def is_int_type(self):
        return self._is_int_type

    def is_fp8_type(self):
        return self._is_fp8_type

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "torch." + self.name


float8_e5m2 = Dtype(8, "float8_e5m2", "__hip_fp8_e5m2", np.uint8)
float16 = Dtype(16, "float16", "_Float16", np.float16)
bfloat16 = Dtype(16, "bfloat16", "__bf16", np.uint16)
float32 = Dtype(32, "float32", "float", np.float32)
float64 = Dtype(64, "float64", "double", np.float64)
float_ = float32
int8 = Dtype(8, "int8", "int8_t", np.int8)
int16 = Dtype(16, "int16", "int16_t", np.int16)
int32 = Dtype(32, "int32", "int32_t", np.int32)
int64 = Dtype(64, "int64", "int64_t", np.int64)
int_ = int32
uint8 = Dtype(8, "uint8", "uint8_t", np.uint8)
uint16 = Dtype(16, "uint16", "uint16_t", np.uint16)
uint32 = Dtype(32, "uint32", "uint32_t", np.uint32)
uint64 = Dtype(64, "uint64", "uint64_t", np.uint64)


def numpy_type_to_torch_type(numpy_type):
    match numpy_type:
        case np.float16:
            return float16
        case np.float32:
            return float32
        case np.float64:
            return float64
        case np.int8:
            return int8
        case np.int16:
            return int16
        case np.int32:
            return int32
        case np.int64:
            return int64
        case np.uint8:
            return uint8
        case np.uint16:
            return uint16
        case np.uint32:
            return uint32
        case np.uint64:
            return uint64
        case _:
            raise RuntimeError(f"Unrecognized Numpy type {numpy_type}")


# Adapted from /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h
def float_2_bfloatraw(f):
    f = float(f)
    # Adapted from https://stackoverflow.com/a/16444786
    u32 = ctypes.c_uint32.from_buffer(ctypes.c_float(f)).value
    # We can't use `~` because Python treats all ints as signed
    not_u32 = 0xffffffff - u32
    if (not_u32 & 0x7f800000) != 0:
        # zero, normal or subnormal
        u32 += 0x7fff + ((u32 >> 16) & 1)
    elif (u32 & 0xffff) != 0:
        # Inf or NaN
        u32 |= 0x10000
    return u32 >> 16


# Adapted from /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h
def double_2_bfloatraw(d_in):
    d_in = float(d_in)
    f32 = ctypes.c_float(d_in).value
    u32 = ctypes.c_uint32.from_buffer(ctypes.c_float(f32)).value
    d = ctypes.c_double(f32).value

    if (d_in > 0.0 and d > d_in) or (d_in < 0.0 and d < d_in):
        u32 -= 1
        u32 |= 1

    f32 = ctypes.c_float.from_buffer(ctypes.c_uint32(u32)).value
    return float_2_bfloatraw(f32)


# Adapted from /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h
def bfloatraw_2_float(u16):
    u16 = int(u16)
    u32 = u16 << 16
    return ctypes.c_float.from_buffer(ctypes.c_uint32(u32)).value


__HIP_E4M3 = 0  # OCP E4M3
__HIP_E5M2 = 1  # OCP E5M2
__HIP_E4M3_FNUZ = 2  # Standard FP8
__HIP_E5M2_FNUZ = 3  # BF8

__HIP_NOSAT = 0  # No saturation
__HIP_SATFINITE = 1  # Saturate to finite


def clz(x):
    num_leading_zeros = 0
    for i in range(31, -1, -1):
        if (x & (1 << i)) != 0:
            return num_leading_zeros
        num_leading_zeros += 1
    return num_leading_zeros


# Adapted from /opt/rocm/include/hip/amd_detail/amd_hip_fp8.h
def cast_to_f8(dtype, is_fnuz, _x, wm, we, clip=False):
    assert dtype == float16 or dtype == float32 or dtype == float64
    if dtype == float16:
        # This is necessary because ctypes does not have support for float16
        dtype = float32
    if isinstance(_x, np.number):
        _x = _x.item()
    assert type(_x) is float
    if dtype == float64:
        mfmt = 52
        x = ctypes.c_ulonglong.from_buffer(ctypes.c_double(_x)).value
        head = x & 0xFFF0000000000000
        mantissa = x & 0xFFFFFFFFFFFFF
        exponent = (head >> 52) & 0x7FF
        sign = head >> 63
        bias = 1023
        fInf = 0x7FF0000000000000
        mask = 0x7FFFFFFFFFFFFFFF
    else:
        mfmt = 23
        x = ctypes.c_uint.from_buffer(ctypes.c_float(_x)).value
        head = x & 0xFF800000
        mantissa = x & 0x7FFFFF
        exponent = (head >> 23) & 0xFF
        sign = head >> 31
        bias = 127
        fInf = 0x7F800000
        mask = 0x7FFFFFFF

    if is_fnuz:
        signed_inf = ((sign << 7) + 0x7f) if clip else 0x80
        nan = 0x80
    else:
        if we == 4:  # e4m3
            signed_inf = (sign << 7) + (0x7e if clip else 0x7f)
        else:  # e5m2
            signed_inf = (sign << 7) + (0x7b if clip else 0x7c)
        nan = (sign << 7) + 0x7f

    # Max values
    if dtype == float64:
        if we == 5:  # 57344
            ifmax = 0x40EC000000000000
        else:
            if is_fnuz:  # 240
                ifmax = 0x406E000000000000
            else:  # 448
                ifmax = 0x407C000000000000
    else:
        if we == 5:
            ifmax = 0x47600000
        else:
            if is_fnuz:
                ifmax = 0x43700000
            else:
                ifmax = 0x43E00000

    # Deal with inf and NaNs
    if (x & fInf) == fInf:
        if is_fnuz or we == 4:
            return nan  # funz and OCP E4M3 has no INF
        if mantissa != 0:
            return nan  # NaN
        return 0x7C if sign == 0 else 0xFC  # E5M2 Inf

    if (x & mask) > ifmax:
        return signed_inf

    if x == 0:
        return 0

    # For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
    f8_bias = (1 << (we - 1)) - 1 + (1 if is_fnuz else 0)
    f8_denormal_act_exponent = 1 - f8_bias  # actual exponent of f8 denormal

    if exponent == 0:
        act_exponent = exponent - bias + 1
        exponent_diff = f8_denormal_act_exponent - act_exponent
    else:
        act_exponent = exponent - bias
        if act_exponent <= f8_denormal_act_exponent:
            exponent_diff = f8_denormal_act_exponent - act_exponent
        else:
            exponent_diff = 0
        mantissa += (1 << mfmt)

    midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1)) == \
        (1 << (mfmt - wm + exponent_diff - 1))

    if exponent_diff > 0:
        mantissa >>= exponent_diff
    elif exponent_diff == -1:
        mantissa <<= -exponent_diff
    implicit_one = bool(mantissa & (1 << mfmt))
    f8_exponent = (act_exponent + exponent_diff) + f8_bias - (0 if implicit_one else 1)

    drop_mask = (1 << (mfmt - wm)) - 1
    odd = bool(mantissa & (1 << (mfmt - wm)))
    mantissa += ((mantissa if odd else mantissa - 1) if midpoint else mantissa) & drop_mask

    if f8_exponent == 0:
        if (1 << mfmt) & mantissa:
            f8_exponent = 1  # denormal overflow to become normal, promote exponent
    else:
        if (1 << (mfmt + 1)) & mantissa:
            mantissa >>= 1
            f8_exponent += 1

    mantissa >>= (mfmt - wm)

    max_exp = (1 << we) - 1
    if f8_exponent > max_exp:
        if clip:
            mantissa = (1 << wm) - 1
            f8_exponent = max_exp
        else:
            return signed_inf

    if f8_exponent == 0 and mantissa == 0:
        return 0 if is_fnuz else (sign << 7)
    mantissa &= (1 << wm) - 1
    return (sign << 7) | (f8_exponent << wm) | mantissa


def cast_from_f8(dtype, is_fnuz, x, wm, we, clip=False):
    assert dtype == float16 or dtype == float32 or dtype == float64
    if dtype == float16:
        # This is necessary because ctypes does not have support for float16
        dtype = float32
    if isinstance(x, np.number) or isinstance(x, np.ndarray):
        x = x.item()
    assert type(x) is int

    weo = 8 if dtype == float32 else 11
    wmo = 23 if dtype == float32 else 52
    if dtype == float32:
        ifInf = 0x7F800000
        ifNegInf = 0xFF800000
        ifNaN = 0x7F800001
        ifNeg0 = 0x80000000
        # Max number in e5m2 57344
        ifmax = 0x47600000
        ifmin = 0xC7600000
        fInf = ctypes.c_float.from_buffer(ctypes.c_uint(ifInf)).value
        fNegInf = ctypes.c_float.from_buffer(ctypes.c_uint(ifNegInf)).value
        fNaN = ctypes.c_float.from_buffer(ctypes.c_uint(ifNaN)).value
        fNeg0 = ctypes.c_float.from_buffer(ctypes.c_uint(ifNeg0)).value
        fmax = ctypes.c_float.from_buffer(ctypes.c_uint(ifmax)).value
        fmin = ctypes.c_float.from_buffer(ctypes.c_uint(ifmin)).value
    else:
        ifInf = 0x7FF0000000000000
        ifNegInf = 0xFFF0000000000000
        ifNaN = 0x7FF0000000000001
        ifNeg0 = 0x8000000000000000
        # Max number in e5m2 57344
        ifmax = 0x40EC000000000000
        ifmin = 0xC0EC000000000000
        fInf = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifInf)).value
        fNegInf = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifNegInf)).value
        fNaN = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifNaN)).value
        fNeg0 = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifNeg0)).value
        fmax = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifmax)).value
        fmin = ctypes.c_double.from_buffer(ctypes.c_ulonglong(ifmin)).value

    if x == 0:
        return 0
    sign = x >> 7
    mantissa = x & ((1 << wm) - 1)
    exponent = (x & 0x7F) >> wm
    if is_fnuz:
        if x == 0x80:
            return fNaN
    else:
        if x == 0x80:
            return fNeg0
        if we == 4:  # e4m3
            if (x & 0x7F) == 0x7F:
                return fNaN
        elif (x & 0x7C) == 0x7C:  # e5m2 NaN/Inf
            if (x & 0x3) == 0:  # Inf
                if clip:
                    return fmin if sign else fmax
                return fNegInf if sign else fInf
            return fNaN

    exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (1 if is_fnuz else 0)

    # subnormal input
    if exponent == 0:
        sh = 1 + clz(mantissa) - (32 - wm)
        mantissa <<= sh
        exponent += 1 - sh
        mantissa &= ((1 << wm) - 1)
    exponent += exp_low_cutoff - 1
    mantissa <<= wmo - wm

    # subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if exponent <= 0:
        mantissa |= 1 << wmo
        mantissa >>= 1 - exponent
        exponent = 0

    if dtype == float32:
        retval = (sign << 31) | (exponent << 23) | mantissa
        return ctypes.c_float.from_buffer(ctypes.c_uint(retval)).value
    else:
        retval = (sign << 63) | (exponent << 52) | mantissa
        return ctypes.c_double.from_buffer(ctypes.c_ulonglong(retval)).value


def __hip_cvt_float_to_fp8(f, sat, interp):
    if interp == __HIP_E4M3_FNUZ or interp == __HIP_E5M2_FNUZ:
        we = 4 if interp == __HIP_E4M3_FNUZ else 5
        wm = 3 if interp == __HIP_E4M3_FNUZ else 2
        return cast_to_f8(float32, True, f, wm, we, sat == __HIP_SATFINITE)
    else:
        we = 4 if interp == __HIP_E4M3 else 5
        wm = 3 if interp == __HIP_E4M3 else 2
        return cast_to_f8(float32, False, f, wm, we, sat == __HIP_SATFINITE)


def __hip_cvt_double_to_fp8(f, sat, interp):
    if interp == __HIP_E4M3_FNUZ or interp == __HIP_E5M2_FNUZ:
        we = 4 if interp == __HIP_E4M3_FNUZ else 5
        wm = 3 if interp == __HIP_E4M3_FNUZ else 2
        return cast_to_f8(float64, True, f, wm, we, sat == __HIP_SATFINITE)
    else:
        we = 4 if interp == __HIP_E4M3 else 5
        wm = 3 if interp == __HIP_E4M3 else 2
        return cast_to_f8(float64, False, f, wm, we, sat == __HIP_SATFINITE)


def __hip_cvt_bfloat16raw_to_fp8(hr, sat, interp):
    fval = bfloatraw_2_float(hr)
    return __hip_cvt_float_to_fp8(fval, sat, interp)


def convert_to_fp8_from(dtype, interpret, val):
    saturation = __HIP_SATFINITE
    if dtype == float64:
        return __hip_cvt_double_to_fp8(float(val), saturation, interpret)
    elif dtype == bfloat16:
        assert type(val) == dtype.numpy_type
        return __hip_cvt_bfloat16raw_to_fp8(val, saturation, interpret)
    else:
        assert dtype.is_native_numpy_type()
        assert dtype.bit_width // 8 <= 4
        return __hip_cvt_float_to_fp8(float(val), saturation, interpret)


def convert_to_fp8_e4m3_from(dtype, val):
    return convert_to_fp8_from(dtype, __HIP_E4M3, val)


def convert_to_fp8_e5m2_from(dtype, val):
    return convert_to_fp8_from(dtype, __HIP_E5M2, val)


def convert_from_fp8_to(dtype, wm, we, val):
    if dtype == float64:
        return cast_from_f8(float64, False, val, wm, we)
    elif dtype == bfloat16:
        f = cast_from_f8(float32, False, val, wm, we)
        return float_2_bfloatraw(f)
    else:
        assert dtype.is_native_numpy_type()
        assert dtype.bit_width // 8 <= 4
        f = cast_from_f8(float32, False, val, wm, we)
        if dtype.is_int_type():
            return int(f)
        return f


def convert_from_fp8_e4m3_to(dtype, val):
    __we = 4
    __wm = 3
    return convert_from_fp8_to(dtype, __wm, __we, val)


def convert_from_fp8_e5m2_to(dtype, val):
    __we = 5
    __wm = 2
    return convert_from_fp8_to(dtype, __wm, __we, val)


def convert_numpy_array_to_non_native_type(data, dtype):
    assert not dtype.is_native_numpy_type()
    src_dtype = numpy_type_to_torch_type(data.dtype)
    conv = None
    if dtype == bfloat16:
        if src_dtype in [uint64, int64, uint32, int32, float64]:
            conv = double_2_bfloatraw
        elif src_dtype in [uint16, int16, float32]:
            conv = float_2_bfloatraw
    elif dtype == float8_e5m2:
        if src_dtype == float16 or src_dtype == float32 or src_dtype == float64:
            conv = lambda x: convert_to_fp8_e5m2_from(src_dtype, x)
        elif src_dtype.is_int_type():
            if src_dtype.bit_width // 8 <= 4:
                conv = lambda x: convert_to_fp8_e5m2_from(float32, float(x))
            else:
                conv = lambda x: convert_to_fp8_e5m2_from(float64, float(x))
    if conv is None:
        raise RuntimeError(f"Cannot convert from {data.dtype} to {dtype}")
    if data.ndim == 0:
        return np.array(conv(data), dtype=dtype.numpy_type)

    def func1d(vec):
        return np.array([conv(elem) for elem in vec], dtype=dtype.numpy_type)

    return np.apply_along_axis(func1d, data.ndim - 1, data)


def convert_non_native_type_to_numpy_array(data, dtype):
    assert not dtype.is_native_numpy_type()
    conv = None
    if dtype == bfloat16:
        conv = bfloatraw_2_float
    elif dtype == float8_e5m2:
        conv = lambda x: convert_from_fp8_e5m2_to(float32, x)
    if conv is None:
        raise RuntimeError(f"Cannot convert from {dtype} to float")
    if data.ndim == 0:
        return np.array(conv(data), dtype=np.float32)

    def func1d(vec):
        return np.array([conv(elem) for elem in vec], dtype=np.float32)

    return np.apply_along_axis(func1d, data.ndim - 1, data)
