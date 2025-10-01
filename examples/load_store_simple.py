import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def simple_copy_kernel0(in_ptr, out_ptr, in_stride_m, in_stride_n, out_stride_m, out_stride_n, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ## Each workgroup copies a BLOCK_M x N tile
    tl.assume(in_stride_n > 0)
    tl.assume(in_stride_m > 0)
    tl.assume(out_stride_n > 0)
    tl.assume(out_stride_m > 0)
    N_new: tl.uint64 = N
    pid = tl.program_id(axis=0).to(tl.uint64)
    tl.assume(pid >= 0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    in_ptrs = in_ptr + (offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)

    data = tl.load(in_ptrs)
    tl.store(out_ptrs, data)


def simple_copy(x, BLOCK_M, BLOCK_N):
    output = torch.randn((BLOCK_M, BLOCK_N), device=x.device, dtype=torch.float32)
    assert x.device == DEVICE and output.device == DEVICE

    M, N = x.shape[0], x.shape[1]

    grid = (1, 1)
    simple_copy_kernel0[grid](x, output, x.stride(0), x.stride(1), output.stride(0), output.stride(1), M, N, BLOCK_M,
                              BLOCK_N, num_warps=1)
    return output


'''
Allocate a tensor of size
2 * 1024^3 * sizeof(float32) = 8 GB

The copy kernel launches one workgroup that copies the rightmost 2x32 sub-tensor
from the input to the output.
Though the subtile, i.e. output, tensor is very small, the input tensor has
a very large stride for the two rows.

row0: offsets 0 to 128 in bytes
row1: offsets 4 * 1024^3 to 4 * 1024^3 + 128 in bytes

The buffer is viewed as a number of records, each of size `stride`.
We choose stride = 256 bytes, then there are 8 * 1024^3 / 256 = 32 * 1024^2 records.

row0: index 0, offsets 0 - 128
row1: index 16 * 1024^2, offsets 0 - 128

The final offsets is presented as index * stride + offsets
'''
M, N = 2, 1024 * 1024 * 1024
BLOCK_M, BLOCK_N = 2, 32
input = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
for i in range(32):
    input[0, i] = i
    input[1, i] = i + 32

output = simple_copy(input, BLOCK_M, BLOCK_N)

torch.set_printoptions(precision=2, threshold=50000, linewidth=200, sci_mode=False)
if torch.allclose(output[0:2, 0:32], input[0:2, 0:32], atol=1e-2, rtol=0):
    print("✅ Correct")
else:
    print("❌ Incorrect")
    print("diff:")
    print(output - input[0:BLOCK_M, 0:BLOCK_N])
    print("output:")
    print(output)
