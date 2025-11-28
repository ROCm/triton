import torch

import triton
import triton.language as tl
from triton.runtime import driver

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@gluon.jit
def reduce_kernel(x_ptr, y_ptr, x_row_stride, x_col_stride,
                  M: gl.constexpr, N: gl.constexpr):

    layout_x: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 1])
    x_col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout_x))
    x_row_offsets = gl.arange(0, M, gl.SliceLayout(1, layout_x))
    x_offsets = x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride
    x_ptrs = x_ptr + x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride

    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=x_offsets)

    row_sum = gl.sum(x, 1)

    y_ptrs = y_ptr + x_row_offsets
    gl.store(y_ptrs, row_sum)

def test_reduce2D():

    num_warps = 1
    mfma_size = 32
    elemPerThread = 16
    mRepeats = 2
    M = num_warps * mfma_size * mRepeats
    N = elemPerThread // 16 * mfma_size
    x = torch.randn((M, N), dtype=torch.float32, device=DEVICE)
    y = torch.randn(M, dtype=torch.float32, device=DEVICE)

    grid = (1, 1)

    reduce_kernel[grid](x, y, x.stride(0), x.stride(1), M, N, num_warps=num_warps)

    ## torch result
    torch_ref = torch.sum(x, 1)
    torch.testing.assert_close(y, torch_ref)


if __name__ == "__main__":
    test_reduce2D()
