#!/usr/bin/env python3
import argparse
import sys

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_SIZE_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACTIVATION: tl.constexpr,
    output_datatype: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    if torch.version.hip is None:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(output_datatype)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# convert fp8 to fp16 for testing
@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input
    tl.store(output_ptr + offsets, output, mask=mask)


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def matmul(a, b, c, output_type, block_m, block_n, block_k, group_m, split_k, num_stages, num_warps, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    otype = tl.float32
    if output_type == torch.float16:
        otype = tl.float16
    elif output_type == torch.bfloat16:
        otype = tl.bfloat16

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    for _ in range(1):
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M = block_m, 
            BLOCK_SIZE_N = block_n, 
            BLOCK_SIZE_K = block_k,
            GROUP_SIZE_M = group_m,
            SPLIT_K = split_k,
            num_warps = num_warps,
            num_stages = num_stages,
            ACTIVATION=activation,
            output_datatype=otype,
        )


def gen_input(M, N, d_type, isFp8, seed, device='cuda'):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if isFp8: # convert fp8 to fp16 for ref input
        fp8_type = tl.float8e4
        f8_tensor = torch.randn((M, N), dtype=torch.float32, device='cuda') * 10
        f8_tensor = f8_tensor.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, fp8_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = f8_tensor.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)
    else: # other data type
        input = torch.randn((M, N), dtype=d_type, device=device)
        input_f16 = input
    return input, input_f16


def test_gemm(M, N, K, block_m, block_n, block_k, group_m, split_k, num_stages, num_warps, dtype, fp8a, fp8b):
    a, a_f16 = gen_input(M, K, d_type=dtype, isFp8=fp8a, seed=10, device='cuda')
    b, b_f16 = gen_input(K, N, d_type=dtype, isFp8=fp8b, seed=11, device='cuda')

    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    matmul(a, b, c, dtype, block_m, block_n, block_k, group_m, split_k, num_stages, num_warps)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="test gemm tuning",
        description="Tuning infra for triton gemm",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-block_m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-block_n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-block_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-group_m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-split_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-num_warps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-num_stages", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-dtype", type=str, default='fp16', help="Input/output data type")
    parsed_args = parser.parse_args(args)

    fp8a = False
    fp8b = False
    dtype = torch.float16
    if parsed_args.dtype == "fp8a":
        dtype = torch.float16
        fp8a = True
    elif parsed_args.dtype == 'fp8b':
        dtype = torch.float16
        fp8b = True
    elif parsed_args.dtype == 'fp16':
        dtype = torch.float16
    elif parsed_args.dtype == 'fp32':
        dtype = torch.float32
    elif parsed_args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        print(f"Unsupported datatype {args.dtype}")
        sys.exit(1)

    M = parsed_args.m
    N = parsed_args.n
    K = parsed_args.k
    block_m = parsed_args.block_m
    block_n = parsed_args.block_n
    block_k = parsed_args.block_k
    group_m = parsed_args.group_m
    split_k = parsed_args.split_k
    num_stages = parsed_args.num_stages
    num_warps = parsed_args.num_warps
    test_gemm(M, N, K, block_m, block_n, block_k, group_m, split_k, num_stages, num_warps, dtype, fp8a, fp8b)


if __name__ == '__main__':
    sys.exit(main())
