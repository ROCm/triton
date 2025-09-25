import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,  #

):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Enable buffer ops
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    #tl.assume(stride_bk > 0)
    #tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    c = a + b

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    tl.store(c_ptrs, c)


def add(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, N = a.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = (1,1)
    add_kernel[grid](
        a, b, c,  #
        M, N,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        num_warps = 1
    )
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
M, N = 16, 16
a = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
b = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
triton_output = add(a, a)
torch_output = a + a
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
    print(f"Max diff: {torch.max((torch_output - triton_output).abs())}")
