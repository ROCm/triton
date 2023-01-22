# flake8: noqa: F821,F841
import itertools
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


def check_type_supported(dtype):
    '''
    skip test if dtype is not supported on the current device
    '''
    cc = torch.cuda.get_device_capability()
    if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
        pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda'):
    check_type_supported(dtype_x)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(z_ref), device=device, dst_type=dtype_x)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE, num_warps=4)
    # compare
    #np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    pytest.fail("fail always")


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', y_low=None, y_high=None):
    check_type_supported(dtype_x)  # early return if dtype_x is not supported
    check_type_supported(dtype_y)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
    if dtype_z is not None:
        z_ref = z_ref.astype(dtype_z)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
    kernel[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=expr, rtol=0.01)


def _mod_operation_ill_conditioned(dtype_x, dtype_y) -> bool:
    # The result of x % y is ill-conditioned if x % y is much smaller than x.
    # pytorch/CUDA has slightly different (probably better) rounding on
    # remainders than stock LLVM. We currently don't expect to match it
    # bit-for-bit.
    return (dtype_x, dtype_y) in [
        ('int32', 'bfloat16'),
        ('int32', 'float16'),
        ('int32', 'float32'),
        ('int64', 'bfloat16'),
        ('int64', 'float16'),
        ('int64', 'float32'),
        ('int64', 'float64'),
        ('uint16', 'bfloat16'),
        ('uint16', 'float16'),
        ('uint16', 'float32'),
        ('uint32', 'bfloat16'),
        ('uint32', 'float16'),
        ('uint32', 'float32'),
        ('uint64', 'bfloat16'),
        ('uint64', 'float16'),
        ('uint64', 'float32'),
        ('uint64', 'float64'),
    ]

# ----------------
# test math ops
# ----------------


@pytest.mark.parametrize("expr", [
    'exp', 'log', 'cos', 'sin'
])
def test_math_op(expr, device='cuda'):
    _test_unary('float32', f'tl.{expr}(x)', f'np.{expr}(x) ', device=device)


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = triton_gpu.convert_layout %3 : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [
    (f'x[{s}]', d)
    for s in ['None, :', ':, None',
              'None, :, :',
              ':, :, None']
    for d in ['int32', 'uint32', 'uint16']
])
def test_index1d(expr, dtype_str, device='cuda'):
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y + 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


# ---------------
# test tuples
# ---------------


@triton.jit
def fn(a, b):
    return a + b, \
        a - b, \
        a * b


def test_tuples():
    device = 'cuda'

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


# ---------------
# test atomics
# ---------------
@pytest.mark.parametrize("op, dtype_x_str, mode", itertools.chain.from_iterable([
    [
        ('add', 'float16', mode),
        ('add', 'uint32', mode), ('add', 'int32', mode), ('add', 'float32', mode),
        ('max', 'uint32', mode), ('max', 'int32', mode), ('max', 'float32', mode),
        ('min', 'uint32', mode), ('min', 'int32', mode), ('min', 'float32', mode),
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x_str, mode, device='cuda'):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        if dtype_x_str == 'float16':
            pytest.skip("Only test atomic float16 ops on devices with sm >= 70")
    if torch.version.hip is not None:
        if dtype_x_str == 'float16' or ((op == 'min' or op == 'max') and dtype_x_str == 'float32'):
            pytest.skip("Not yet supported on AMDGPU.")
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x)'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = numpy_random((n_programs, ), dtype_str=dtype_x_str, rs=rs)
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, device=device)

    z_tri = to_triton(np.array([neutral], dtype=getattr(np, dtype_x_str)), device=device)
    kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(getattr(np, dtype_x_str))
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == to_numpy(z_tri).item()
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


# ---------------
# test cast
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False)
    for dtype_x in dtypes
    for dtype_z in dtypes
] + [
    ('float32', 'bfloat16', False),
    ('bfloat16', 'float32', False),
    ('float32', 'int32', True),
    ('float32', 'int1', False),
] + [
    (f'uint{x}', f'int{x}', True) for x in [8, 16, 32, 64]
] + [
    (f'int{x}', f'uint{x}', True) for x in [8, 16, 32, 64]
])
def test_cast(dtype_x, dtype_z, bitcast, device='cuda'):
    # bfloat16 on cc < 80 will not be tested
    check_type_supported(dtype_x)
    check_type_supported(dtype_z)

    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    x0 = 43 if dtype_x in int_dtypes else 43.5
    if dtype_x in float_dtypes and dtype_z == 'int1':
        x0 = 0.5
    if dtype_x.startswith('bfloat'):
        x_tri = torch.tensor([x0], dtype=getattr(torch, dtype_x), device=device)
    else:
        x = np.array([x0], dtype=getattr(np, dtype_x))
        x_tri = to_triton(x)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr):
        x_ptr = X + tl.arange(0, 1)
        z_ptr = Z + tl.arange(0, 1)
        x = tl.load(x_ptr)
        z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        tl.store(z_ptr, z)

    dtype_z_np = dtype_z if dtype_z != 'int1' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((1,), dtype=getattr(torch, dtype_z), device=device)
    else:
        z_tri = to_triton(np.empty((1, ), dtype=getattr(np, dtype_z_np)), device=device)
    kernel[(1, )](x_tri, z_tri, BITCAST=bitcast)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        assert z_tri == z_ref
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        assert to_numpy(z_tri) == z_ref


@pytest.mark.parametrize("dtype_str", [dtype_str for dtype_str in torch_dtypes])
def test_store_constant(dtype_str):
    check_type_supported(dtype_str)

    """Tests that boolean True is stored as 1"""
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        output = GENERATE_TEST_HERE
        tl.store(output_ptr + offsets, output, mask=mask)

    triton_dtype_str = 'uint8' if dtype_str == 'bool' else dtype_str
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.zeros([BLOCK_SIZE], dtype=tl.{triton_dtype_str}) + 1'})
    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, dtype_str), device='cuda')
    output = torch.zeros([block_size], dtype=getattr(torch, dtype_str), device='cuda')
    kernel[(1,)](output, block_size, BLOCK_SIZE=block_size)

    assert torch.all(output == ref)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_f8_xf16_roundtrip(dtype):
    """Tests that converting an f8 to f16 and back to f8 doesn't change its value"""
    check_type_supported(dtype)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device='cuda')
    f8 = triton.reinterpret(f8_tensor, tl.float8)
    n_elements = f8_tensor.numel()
    xf16 = torch.empty_like(f8_tensor, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f8, xf16, n_elements, BLOCK_SIZE=1024)

    f8_output_tensor = torch.empty_like(xf16, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    copy_kernel[grid](xf16, f8_output, n_elements, BLOCK_SIZE=1024)

    assert torch.all(f8_tensor == f8_output_tensor)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_f8_xf16_roundtrip(dtype):
    """Tests that converting an f8 to f16 and back to f8 doesn't change its value"""
    check_type_supported(dtype)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device='cuda')
    f8 = triton.reinterpret(f8_tensor, tl.float8)
    n_elements = f8_tensor.numel()
    xf16 = torch.empty_like(f8_tensor, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f8, xf16, n_elements, BLOCK_SIZE=1024)

    f8_output_tensor = torch.empty_like(xf16, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    copy_kernel[grid](xf16, f8_output, n_elements, BLOCK_SIZE=1024)

    assert torch.all(f8_tensor == f8_output_tensor)

def test_f16_to_f8_rounding():
    """Takes all float16s, converts them to float8 and back to float16. Checks that the absolute
    error is the minimum over all float8.
    Or the same explanation a bit mathier:
    for all f16 |f16 - fromf8(tof8(f16))| == min over all f8 |f16 - fromf8(f8)|"""
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    # torch.view with a dtype isn't supported in triton's torch yet so use numpy's view
    f16_input_np = (
        np.array(
            range(-int(2 ** (16 - 1)), int(2 ** (16 - 1))), dtype=np.int16,
        )
        .view(np.float16)
    )
    f16_input = torch.tensor(f16_input_np, dtype=torch.float16, device='cuda')
    n_elements = f16_input.numel()
    f8_output_tensor = torch.empty_like(f16_input, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f16_input, f8_output, n_elements, BLOCK_SIZE=1024)

    f16_output = torch.empty_like(f16_input, dtype=torch.float16)
    copy_kernel[grid](f8_output, f16_output, n_elements, BLOCK_SIZE=1024)

    abs_error = torch.abs(f16_input - f16_output)

    all_f8_vals_tensor = torch.tensor(range(2 ** 8), dtype=torch.uint8, device='cuda')
    all_f8_vals = triton.reinterpret(all_f8_vals_tensor, tl.float8)
    all_f8_vals_in_f16 = torch.empty_like(all_f8_vals_tensor, dtype=torch.float16)
    copy_kernel[grid](all_f8_vals, all_f8_vals_in_f16, n_elements=256, BLOCK_SIZE=1024)

    all_finite_f8_vals_in_f16 = all_f8_vals_in_f16[
        torch.isfinite(all_f8_vals_in_f16)
    ]

    min_error = torch.min(
        torch.abs(
            f16_input.reshape((-1, 1))
            - all_finite_f8_vals_in_f16.reshape((1, -1))
        ),
        dim=1,
    )[0]
    # 1.9375 is float8 max
    mismatch = torch.logical_and(
        abs_error != min_error, torch.logical_and(torch.isfinite(f16_input), torch.abs(f16_input) < 1.9375)
    )
    assert torch.all(
        torch.logical_not(mismatch)
    ), f"f16_input[mismatch]={f16_input[mismatch]} f16_output[mismatch]={f16_output[mismatch]} abs_error[mismatch]={abs_error[mismatch]} min_error[mismatch]={min_error[mismatch]}"


# ---------------
# test reduce
# ---------------


def get_reduced_dtype(dtype_str, op):
    if op == 'argmin' or op == 'argmax':
        return 'int32'
    if dtype_str in ['int8', 'uint8', 'int16', 'uint16']:
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


@pytest.mark.parametrize("op, dtype_str, shape",
                         [(op, dtype, shape)
                          for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                          for dtype in dtypes_with_bfloat16
                          for shape in [32, 64, 128, 512]])
def test_reduce1d(op, dtype_str, shape, device='cuda'):
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(Z, GENERATE_TEST_HERE)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=0)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((shape,), dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'argmin': np.argmin, 'argmax': np.argmax}[op]
    # numpy result
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((1,), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK=shape)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op == 'argmin' or op == 'argmax':
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            np.testing.assert_equal(x[z_ref], x[z_tri])
        else:
            np.testing.assert_equal(z_ref, z_tri)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [
    (op, dtype, (1, 1024), axis) for dtype in dtypes_with_bfloat16
    for op in ['min', 'max', 'sum']
    for axis in [1]
]


# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
# , (32, 64), (64, 128)]
if 'V100' in torch.cuda.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]


reduce_configs2 = [
    (op, 'float32', shape, axis)
    for op in ['min', 'max', 'sum']
    for shape in reduce2d_shapes
    for axis in [0, 1]
]


@pytest.mark.parametrize("op, dtype_str, shape, axis", reduce_configs1 + reduce_configs2)
def test_reduce2d(op, dtype_str, shape, axis, device='cuda'):
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        z = GENERATE_TEST_HERE
        if AXIS == 1:
            tl.store(Z + range_m, z)
        else:
            tl.store(Z + range_n, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=AXIS)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'argmin': np.argmin, 'argmax': np.argmax}[op]
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    # numpy result
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_tri_dtype_str = 'bfloat16'
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
    else:
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((shape[1 - axis],), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op == 'argmin' or op == 'argmax':
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = np.expand_dims(z_ref, axis=axis)
            z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)


# ---------------
# test arange
# ---------------


@pytest.mark.parametrize("start", [0, 1, 7, 16])
def test_arange(start, device='cuda'):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr,
                START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)
    _kernel[(1,)](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    triton.testing.assert_almost_equal(z_tri, z_ref)


# ---------------
# test load
# ---------------


@pytest.mark.parametrize("dtype_str, size, size_diff", [(dtype_str, size, size_diff) for dtype_str in torch_dtypes for size in [128, 512] for size_diff in [0, 1, 2, 3, 4]])
def test_masked_load(dtype_str, size, size_diff, device='cuda'):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype)  # bfloat16 on cc < 80 will not be tested

    input_size = size - size_diff
    output_size = size
    if dtype_str == 'bool':
        input = torch.randint(0, 2, (input_size,), dtype=dtype, device=device)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        input = torch.randint(0, 127, (input_size,), dtype=dtype, device=device)
    else:
        input = torch.rand(input_size, dtype=dtype, device=device)
    output = torch.zeros((output_size,), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in_ptr, out_ptr, in_size: tl.constexpr, out_size: tl.constexpr):
        in_offsets = tl.arange(0, out_size)
        # Load inputs.
        x = GENERATE_TEST_HERE
        # Store output
        output_offsets = tl.arange(0, out_size)
        tl.store(out_ptr + output_offsets, x)

    mask_str = "mask=in_offsets < in_size, other=1" if size_diff > 0 else "None"
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.load(in_ptr + in_offsets, {mask_str})"})
    kernel[(1,)](input, output, input_size, output_size)

    reference_out = torch.cat((input, torch.ones((size_diff,), dtype=dtype, device=device)))
    triton.testing.allclose(output, reference_out)


# ---------------
# test store
# ---------------

# ---------------
# test if
# ---------------

# ---------------
# test for
# ---------------

# ---------------
# test while
# ---------------

# ---------------
# test default
# ---------------
# TODO: can't be local to test_default


@triton.jit
def _impl(value=10):
    return value


def test_default():
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device='cuda')
    ret1 = torch.zeros(1, dtype=torch.int32, device='cuda')

    @triton.jit
    def _kernel(ret0, ret1, value):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1,)](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

# ---------------
# test noop
# ----------------


def test_noop(device='cuda'):
    @triton.jit
    def kernel(x):
        pass
    x = to_triton(numpy_random((1,), dtype_str='int32'), device=device)
    kernel[(1, )](x)


@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_pointer_arguments(device):
    @triton.jit
    def kernel(x):
        pass
    x = torch.empty(1024, device=device)
    result = True
    try:
        kernel[(1,)](x)
    except ValueError:
        result = True if device == 'cpu' else False
    assert result


@pytest.mark.parametrize("value, value_type", [
    (-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
    (2**31, 'u32'), (2**32 - 1, 'u32'), (2**32, 'i64'), (2**63 - 1, 'i64'),
    (-2**63, 'i64'), (2**63, 'u64'), (2**64 - 1, 'u64')
])
def test_value_specialization(value: int, value_type: str, device='cuda') -> None:
    spec_type = None

    def cache_hook(*args, **kwargs):
        nonlocal spec_type
        spec_type = kwargs["compile"]["signature"][0]
    JITFunction.cache_hook = cache_hook

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device='cuda')
    pgm = kernel[(1, )](value, x)

    JITFunction.cache_hook = None
    assert spec_type == value_type

# --------------------
# value specialization
# --------------------


@pytest.mark.parametrize(
    "value, overflow",
    [(2**64 - 1, False), (2**64, True), (-2**63, False), (-2**63 - 1, True)]
)
def test_value_specialization_overflow(value: int, overflow: bool, device='cuda') -> None:

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device='cuda')

    if overflow:
        with pytest.raises(OverflowError):
            kernel[(1, )](value, x)
    else:
        kernel[(1, )](value, x)


# ----------------
# test constexpr
# ----------------

@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '<', '>'])
@pytest.mark.parametrize("is_lhs_constexpr", [False, True])
@pytest.mark.parametrize("is_rhs_constexpr", [True, False])
def test_bin_op_constexpr(op, is_lhs_constexpr, is_rhs_constexpr):

    @triton.jit
    def kernel(Z, X, Y):
        x = tl.load(X)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z, z)

    x_str = "3.14" if is_lhs_constexpr else "x"
    y_str = "4.13" if is_rhs_constexpr else "y"
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{x_str} {op} {y_str}"})
    x = numpy_random((1,), dtype_str="float32")
    y = numpy_random((1,), dtype_str="float32")
    z = np.array(eval(f"{x_str} {op} {y_str}"))
    x_tri = to_triton(x)
    y_tri = to_triton(y)
    z_tri = to_triton(np.empty((1,), dtype=z.dtype))
    kernel[(1,)](z_tri, x_tri, y_tri)
    np.testing.assert_allclose(z, to_numpy(z_tri))


def test_constexpr_shape():

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32))
    kernel[(1,)](x_tri)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256))


def test_constexpr_scalar_shape():

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32))
    kernel[(1,)](x_tri, 32)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256) % 8)

# -------------
# test call
# -------------


@triton.jit
def val_multiplier(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        vec = val_multiplier(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


def test_call():

    @triton.jit
    def kernel(ptr, n_elements, num1, num2):
        vecmul_kernel(ptr, n_elements, num1)
        vecmul_kernel(ptr, n_elements, num2)

    size = 1024
    rand_val = numpy_random((size,), dtype_str="float32")
    rand_val_tri = to_triton(rand_val, device='cuda')
    kernel[(size // 128,)](rand_val_tri, size, 3, 5)

    ans = rand_val * 1 * 2 * 1 * 2 * 3 * 4
    np.testing.assert_equal(to_numpy(rand_val_tri), ans)

# -------------
# test if
# -------------


def test_if():

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if pid % 2:
            tl.store(Ret, tl.load(XTrue))
        else:
            tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device='cuda')
    x_true = torch.tensor([3.14], dtype=torch.float32, device='cuda')
    x_false = torch.tensor([1.51], dtype=torch.float32, device='cuda')
    ret = torch.empty(1, dtype=torch.float32, device='cuda')
    kernel[(1,)](cond, x_true, x_false, ret)


def test_num_warps_pow2():
    dst = torch.empty(128, device='cuda')

    @triton.jit
    def _kernel(dst):
        pass

    with pytest.raises(AssertionError, match='must be a power of 2'):
        _kernel[(1,)](dst=dst, num_warps=3)
    _kernel[(1,)](dst=dst, num_warps=1)
    _kernel[(1,)](dst=dst, num_warps=2)
    _kernel[(1,)](dst=dst, num_warps=4)

# -------------
# test extern
# -------------

def system_libdevice_path() -> str:
    if torch.version.hip is not None:
        return ''
    else:
        _SYSTEM_LIBDEVICE_SEARCH_PATHS = [
            '/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc',
            '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc',
        ]
        SYSTEM_LIBDEVICE_PATH: Optional[str] = None
        for _p in _SYSTEM_LIBDEVICE_SEARCH_PATHS:
            if os.path.exists(_p):
                SYSTEM_LIBDEVICE_PATH = _p
        assert SYSTEM_LIBDEVICE_PATH is not None, \
            "Could not find libdevice.10.bc path"
        return SYSTEM_LIBDEVICE_PATH


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('int32', 'libdevice.ffs', ''),
                          ('float32', 'libdevice.pow', system_libdevice_path()),
                          ('float64', 'libdevice.norm4d', '')])
def test_libdevice_tensor(dtype_str, expr, lib_path):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)

    if expr == 'libdevice.ffs':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.ffs(x)'})
        y_ref = np.zeros(shape, dtype=x.dtype)
        for i in range(shape[0]):
            y_ref[i] = (int(x[i]) & int(-x[i])).bit_length()
    elif expr == 'libdevice.pow':
        # numpy does not allow negative factors in power, so we use abs()
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.pow(x, x)'})
        y_ref = np.power(x, x)
    elif expr == 'libdevice.norm4d':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.norm4d(x, x, x, x)'})
        y_ref = np.sqrt(4 * np.power(x, 2))

    x_tri = to_triton(x)
    # triton result
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    if expr == 'libdevice.ffs':
        np.testing.assert_equal(y_ref, to_numpy(y_tri))
    else:
        np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('float32', 'libdevice.pow', '')])
def test_libdevice_scalar(dtype_str, expr, lib_path):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = X
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((1,), dtype_str=dtype_str, rs=rs)
    y_ref = np.zeros(shape, dtype=x.dtype)

    # numpy does not allow negative factors in power, so we use abs()
    x = np.abs(x)
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.pow(x, x)'})
    y_ref[:] = np.power(x, x)

    # triton result
    x_tri = to_triton(x)[0].item()
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)

# -----------------------
# test layout conversions
# -----------------------
# TODO: backend hsould be tested separately


class MmaLayout:
    def __init__(self, version, warps_per_cta):
        self.version = version
        self.warps_per_cta = str(warps_per_cta)

    def __str__(self):
        return f"#triton_gpu.mma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA={self.warps_per_cta}}}>"


class BlockedLayout:
    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order):
        self.sz_per_thread = str(size_per_thread)
        self.threads_per_warp = str(threads_per_warp)
        self.warps_per_cta = str(warps_per_cta)
        self.order = str(order)

    def __str__(self):
        return f"#triton_gpu.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}}}>"


layouts = [
    # MmaLayout(version=1, warps_per_cta=[1, 4]),
    MmaLayout(version=(2, 0), warps_per_cta=[1, 4]),
    # MmaLayout(version=1, warps_per_cta=[4, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1]),
    BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0]),
    BlockedLayout([1, 4], [4, 8], [2, 2], [1, 0]),
    BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0]),
#    BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1]),
    BlockedLayout([4, 1], [8, 4], [2, 2], [0, 1]),
    BlockedLayout([1, 1], [32, 1], [2, 2], [0, 1]),
    BlockedLayout([4, 4], [1, 32], [4, 1], [1, 0])
]


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("dst_layout", layouts)
def test_convert2d(dtype, shape, src_layout, dst_layout, device='cuda'):
    if str(src_layout) == str(dst_layout):
        pytest.skip()
    if 'mma' in str(src_layout) and 'mma' in str(dst_layout):
        pytest.skip()

    ir = f"""
#src = {src_layout}
#dst = {dst_layout}
""" + """
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func public @kernel_0d1d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<128> : tensor<128x1xi32, #src>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>
    %2 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #src>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>) -> tensor<128x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<128x1xi32, #src>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>) -> tensor<1x128xi32, #src>
    %7 = tt.broadcast %6 : (tensor<1x128xi32, #src>) -> tensor<128x128xi32, #src>
    %8 = tt.broadcast %5 : (tensor<128x1xi32, #src>) -> tensor<128x128xi32, #src>
    %9 = arith.addi %8, %7 : tensor<128x128xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<128x128x!tt.ptr<f16>, #src>, tensor<128x128xi32, #src>
    %11 = tt.load %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #src>
    %3 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #dst>
    %12 = triton_gpu.convert_layout %9 : (tensor<128x128xi32, #src>) -> tensor<128x128xi32, #dst>
    %13 = triton_gpu.convert_layout %11 : (tensor<128x128xf16, #src>) -> tensor<128x128xf16, #dst>
    %14 = tt.addptr %3, %12 : tensor<128x128x!tt.ptr<f16>, #dst>, tensor<128x128xi32, #dst>
    tt.store %14, %13 : tensor<128x128xf16, #dst>
    return
  }
}
"""

    x = to_triton(numpy_random(shape, dtype_str=dtype))
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    assert torch.equal(z, x)
