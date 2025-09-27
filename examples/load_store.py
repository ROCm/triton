import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def simple_copy_kernel0(in_ptr, out_ptr, in_stride_m, in_stride_n, out_stride_m, out_stride_n, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ## Each workgroup copies a BLOCK_M x N tile
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    in_ptrs = in_ptr + (offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)

    max_iter = tl.cdiv(N, BLOCK_N)
    for start_n in range(0, max_iter):
        data = tl.load(in_ptrs)
        tl.store(out_ptrs, data)
        ## Advance the pointers
        in_ptrs += BLOCK_N * in_stride_n
        out_ptrs += BLOCK_N * out_stride_n


@triton.jit
def simple_copy_kernel1(in_ptr, out_ptr, in_stride_m, in_stride_n, out_stride_m, out_stride_n, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ## Each workgroup copies a BLOCK_M x N tile
    pid = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    in_ptrs = in_ptr + pid * BLOCK_M * in_stride_m
    out_ptrs = out_ptr + pid * BLOCK_M * out_stride_m
    in_ptrs = in_ptrs + (offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)
    out_ptrs = out_ptrs + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)

    max_iter = tl.cdiv(N, BLOCK_N)
    for start_n in range(0, max_iter):
        data = tl.load(in_ptrs)
        tl.store(out_ptrs, data)
        ## Advance the pointers
        in_ptrs += BLOCK_N * in_stride_n
        out_ptrs += BLOCK_N * out_stride_n


@triton.jit
def simple_copy_kernel2(in_ptr, out_ptr, in_stride_m, in_stride_n, out_stride_m, out_stride_n, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ## Each workgroup copies a BLOCK_M x N tile
    pid = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    in_ptrs = in_ptr + (offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)

    in_ptrs += pid * BLOCK_M * in_stride_m
    out_ptrs += pid * BLOCK_M * out_stride_m

    max_iter = tl.cdiv(N, BLOCK_N)
    for start_n in range(0, max_iter):
        data = tl.load(in_ptrs)
        tl.store(out_ptrs, data)
        ## Advance the pointers
        in_ptrs += BLOCK_N * in_stride_n
        out_ptrs += BLOCK_N * out_stride_n


def simple_copy(x, BLOCK_M, BLOCK_N):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE

    M, N = x.shape[0], x.shape[1]

    grid = (M // BLOCK_M, 1)
    simple_copy_kernel0[grid](x, output, x.stride(0), x.stride(1), output.stride(0), output.stride(1), M, N, BLOCK_M,
                              BLOCK_N)
    return output


M, N = 256, 256
BLOCK_M, BLOCK_N = 128, 64
input = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
output = simple_copy(input, BLOCK_M, BLOCK_N)
if torch.allclose(output, input, atol=1e-2, rtol=0):
    print("✅ Correct")
else:
    print("❌ Incorrect")
