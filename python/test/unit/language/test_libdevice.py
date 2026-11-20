import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef
from triton._internal_testing import is_hip, is_interpreter


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [
        ("j0", "bessel_j0"),
        ("j1", "bessel_j1"),
        ("y0", "bessel_y0"),
        ("y1", "bessel_y1"),
        ("cyl_bessel_i0", "i0"),
        ("cyl_bessel_i1", "i1"),
    ],
)
def test_bessel(dtype_str, libdevice_fn, torch_special_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    torch.manual_seed(42)
    x = torch.randn((SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    kernel[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)


def test_clz(device):

    @triton.jit
    def triton_clz(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.clz(tl.load(in_ptr + offsets)))

    x = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_clz[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([31, 30, 30, 29, 29, 28, 28, 27], dtype=torch.int32))


def test_popc(device):

    @triton.jit
    def triton_popc(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.popc(tl.load(in_ptr + offsets)))

    x = torch.tensor([0, 1, 2, 3, 4, 7, 8, 15], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_popc[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([0, 1, 1, 2, 1, 3, 1, 4], dtype=torch.int32))


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_isinf(device, dtype_str):

    @triton.jit
    def triton_isinf(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        in_tile = tl.load(in_ptr + offsets, mask=mask)
        if in_ptr.dtype.element_ty == tl.float32:
            out_tile = libdevice.finitef(in_tile)
        else:
            out_tile = libdevice.isfinited(in_tile)
        tl.store(out_ptr + offsets, out_tile, mask=mask)

    x = torch.tensor(
        [float(1), -float(1),
         float(0), -float(0),
         float("inf"), -float("inf"),
         float("nan"), -float("nan")], device=device, dtype=getattr(torch, dtype_str))
    res = torch.tensor([True, True, True, True, False, False, False, False])
    numel = x.numel()
    y = torch.empty_like(x, dtype=torch.bool)
    BLOCK_SIZE = 256
    triton_isinf[(triton.cdiv(numel, BLOCK_SIZE), )](x, y, numel, BLOCK_SIZE)
    assert torch.equal(y.cpu(), res)


@pytest.mark.skipif(not is_hip() or is_interpreter(), reason="requires AMD GPU code generation")
@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_hip_rint_edge_cases(device, dtype_str):
    from triton.language.extra.hip import libdevice as hip_libdevice

    @triton.jit
    def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < N, other=0)
        tl.store(Y + offsets, hip_libdevice.rint(x), offsets < N)

    dtype = getattr(torch, dtype_str)
    ties = torch.tensor([-127.5, -126.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 126.5, 127.5], dtype=dtype)
    large = 2**(24 if dtype == torch.float32 else 53)
    special = torch.tensor([-0.0, 0.0, -large, large, -float("inf"), float("inf"), float("nan")], dtype=dtype)
    x = torch.cat((ties, torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
                   torch.nextafter(ties, torch.full_like(ties, float("inf"))), special))
    expected = torch.round(x)
    actual = torch.empty_like(x, device=device)
    kernel[(1, )](x.to(device), actual, x.numel(), triton.next_power_of_2(x.numel()))
    actual = actual.cpu()
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite], expected[finite])
    assert torch.equal(torch.signbit(actual[expected == 0]), torch.signbit(expected[expected == 0]))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual[torch.isinf(expected)], expected[torch.isinf(expected)])


@pytest.mark.skipif(not is_hip() or is_interpreter(), reason="requires AMD GPU code generation")
def test_hip_rint_int8_quantization(device):
    from triton.language.extra.hip import libdevice as hip_libdevice

    @triton.jit
    def kernel(X, Q, S, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        x = tl.load(X + row * WIDTH + offsets, offsets < WIDTH, other=0)
        absmax = tl.max(tl.abs(x), 0)
        scale = tl.where(absmax > 0, absmax / 127.0, 1.0)
        rounded = hip_libdevice.rint(tl.div_rn(x, scale))
        q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        tl.store(Q + row * WIDTH + offsets, q, offsets < WIDTH)
        tl.store(S + row, scale)

    # Exact power-of-two scales keep halfway cases exact in both implementations.
    row = torch.tensor([-127.0, -126.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 126.5, 127.0],
                       dtype=torch.float32)
    x = torch.stack((torch.zeros_like(row), row, row * 0.25, row * 2.0))
    absmax = x.abs().amax(dim=1)
    expected_scale = torch.where(absmax > 0, absmax / 127.0, 1.0)
    expected = torch.round(x / expected_scale[:, None]).clamp(-127, 127).to(torch.int8)
    actual = torch.empty_like(x, dtype=torch.int8, device=device)
    scale = torch.empty(x.shape[0], dtype=torch.float32, device=device)
    kernel[(x.shape[0], )](x.to(device), actual, scale, x.shape[1], triton.next_power_of_2(x.shape[1]))
    actual, scale = actual.cpu(), scale.cpu()
    assert torch.equal(scale, expected_scale)
    assert torch.equal(actual, expected)
    assert torch.all((actual.float() * scale[:, None] - x).abs() <= scale[:, None] / 2)
