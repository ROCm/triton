import pytest
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel._unsafe_update_src(kernel.src.replace(key, value))
    return kernel

@triton.jit
def copy_kernel(in_ptr, out_ptr, in_stride_m_orig, in_stride_n, out_stride_m_orig, out_stride_n, M, N,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                isSmallTensor: tl.constexpr,
                needInt64Stride: tl.constexpr):
    if needInt64Stride:
        in_stride_m = tl.cast(in_stride_m_orig, tl.int64)
        out_stride_m = tl.cast(out_stride_m_orig, tl.int64)
    else:
        in_stride_m = in_stride_m_orig
        out_stride_m = out_stride_m_orig
    tl.assume(in_stride_n > 0)
    tl.assume(in_stride_m > 0)
    tl.assume(out_stride_n > 0)
    tl.assume(out_stride_m > 0)
    if isSmallTensor:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    tl.assume(pid >= 0)
    offs_mo = tl.arange(0, BLOCK_M) + pid * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)

    COMPUTE_OFFS_MI
    COMPUTE_IN_PTR
    out_ptrs = out_ptr + (offs_mo[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)

    data = tl.load(in_ptrs)
    tl.store(out_ptrs, data)

sizeMap = {
    'size0': 16 * 1024 * 1024,
    'size1': 32 * 1024 * 1024,
    'size2': 64 * 1024 * 1024,
    'size3': 256 * 1024 * 1024,
}

@pytest.mark.parametrize("size", ['size0', 'size1', 'size2', 'size3'])
@pytest.mark.parametrize("ptrCal", ["style0", "style1", "style2"])
def test_buffer_op(size, ptrCal):
    '''
    Allocate a tensor of size 32 x N x sizeof(float32)
    Copy the bottom left block of 16x64 elements, i.e input[16:32, 0:64], to an output tensor

    case 1: 2 GB: N = 2 x 1024^3 / 32 / 4 = 16 x 1024^2 (16777216)
    case 2: 4 Gb: N = 4 x 1024^3 / 32 / 4 = 32 x 1024^2 (3355443200)
    case 3: 8 GB: N = 8 x 1024^3 / 32 / 4 = 64 x 1024^2 (6710886400)
    case 4: 32 GB: N = 32 x 1024^3 / 32 / 4 = 256 x 1024^2 (26843545600)
    '''

    N = sizeMap[size]
    M = 32
    BLOCK_M = M // 2
    BLOCK_N = 64
    num_warps = 4
    input = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    output = torch.randn((BLOCK_M, BLOCK_N), device=input.device, dtype=torch.float32)
    assert input.device == DEVICE and output.device == DEVICE

    isSmallTensor = (N <= 32 * 1024 * 1024)
    needInt64Stride = (N > 4 * 1024 * 1024 * 1024)

    grid = (1, 1)
    stride_im, stride_in = input.stride(0), input.stride(1)
    stride_om, stride_on = output.stride(0), output.stride(1)

    if 'style0' == ptrCal:
        patch_offs_mi = 'offs_mi = tl.arange(0, BLOCK_M) + BLOCK_M + pid * BLOCK_M'
        patch_in_ptrs = 'in_ptrs = in_ptr + (offs_mi[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)'
    elif 'style1' == ptrCal:
        patch_offs_mi = 'offs_mi = tl.arange(0, BLOCK_M) + BLOCK_M + pid * BLOCK_M'
        patch_in_ptrs = 'in_ptrs = in_ptr + offs_mi[:, None] * in_stride_m + offs_n[None, :] * in_stride_n'
    elif 'style2' == ptrCal:
        patch_offs_mi = 'offs_mi = tl.arange(0, BLOCK_M) + pid * BLOCK_M'
        patch_in_ptrs = 'in_ptrs = (in_ptr + BLOCK_M * in_stride_m) + (offs_mi[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)'

    kernel = patch_kernel(copy_kernel, {'COMPUTE_OFFS_MI': patch_offs_mi, 'COMPUTE_IN_PTR': patch_in_ptrs})


    h = kernel.warmup(input, output, stride_im, stride_in, stride_om, stride_on,
                      M, N, BLOCK_M, BLOCK_N, isSmallTensor, needInt64Stride, num_warps=num_warps, grid=(1,1))

    foundBufferLoad = "buffer_load_dwordx4" in h.asm["amdgcn"]
    if not foundBufferLoad:
        pytest.fail("buffer_load not found in the isa")

    kernel[grid](input, output, stride_im, stride_in, stride_om, stride_on,
                              M, N, BLOCK_M, BLOCK_N, isSmallTensor, needInt64Stride, num_warps=num_warps)
    assert torch.all(output == input[BLOCK_M:M, 0:BLOCK_N])
