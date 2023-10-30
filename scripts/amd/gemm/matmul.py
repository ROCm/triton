#!/usr/bin/env python
"""
Matrix Multiplication Tuning Scripts, Changed from the tutorial example "python/tutorials/03-matrix-multiplication.py"
"""

import torch

import triton
import triton.language as tl
import argparse
import sys
import yaml
import os
import subprocess
import pdb



# global flag to indicate whether using the full tuing space
tuning_full_space = False

# pruned some unreasonable config
def prune_configs(configs, named_args):
    # call only for full tuning space
    if not tuning_full_space:
        return configs

    SIZE_M = named_args["a_ptr"].shape[0]
    SIZE_N = named_args["b_ptr"].shape[1]
    SIZE_K = named_args["a_ptr"].shape[1]

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K =\
            kw["BLOCK_SIZE_M"], kw["BLOCK_SIZE_N"], kw["BLOCK_SIZE_K"]
        if SIZE_M <=32 and BLOCK_SIZE_M != 32:
            continue
        if SIZE_N <=32 and BLOCK_SIZE_N != 32:
            continue
        if BLOCK_SIZE_M * BLOCK_SIZE_K / config.num_warps / 64 <= 2:
            continue
        if BLOCK_SIZE_N * BLOCK_SIZE_K / config.num_warps / 64 <= 2:
            continue

        pruned_configs.append(config)

    return pruned_configs


def get_full_tuning_space():
    configs = []
    if not tuning_full_space:
        return configs

    block_mn_range = [16, 32, 64, 128]
    block_k_range = [16, 32, 64]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for num_stages in num_stage_range:
                            configs.append(triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m, 'waves_per_eu': 2}, num_stages=num_stages, num_warps=num_warps))

    return configs


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs= get_full_tuning_space() if tuning_full_space else [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 3, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_stages=0, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': prune_configs,
        'perf_model': None,
        "top_k": None
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
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
    # output_datatype: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, EVEN_K: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)

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
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * (BLOCK_SIZE_K)
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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


def matmul(a, b, c, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        # output_datatype=otype,
    )


def get_best_config(M, N, K):
    best_config = matmul_kernel.get_best_config(M = M, N = N, K = K)
    return best_config

TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
name_to_torch_types = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'int8': torch.int8,
}
if TORCH_HAS_FP8E5B16:
    name_to_torch_types['fp8e4b8'] = torch.float8_e4m3fnuz
if TORCH_HAS_FP8E4B8:
    name_to_torch_types['fp8e5b16'] = torch.float8_e5m2fnuz

name_to_tl_types = {
    'fp32': tl.float32,
    'fp16': tl.float16,
    "int8": tl.int8,
    'fp8e4b8': tl.float8e4b8,
    'fp8e5b16': tl.float8e5b16,
}

def gen_input(M, N, d_type, seed, device='cuda'):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda') * 10
    tl_dtype = name_to_tl_types[d_type]
    if (tl_dtype == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
        (tl_dtype == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not tl_dtype.is_fp8():
        input = raw_data.to(name_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, tl_dtype)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


def test_correctness(M, N, K, a_type, b_type):
    a, a_f16 = gen_input(M, K, d_type=a_type, seed=10, device='cuda')
    b, b_f16 = gen_input(K, N, d_type=b_type, seed=20, device='cuda')

    if a_type == 'int8':
        assert a_type == b_type
        out_dtype = torch.int32
    elif a_type == 'fp32':
        assert a_type == b_type
        out_dtype = torch.float32
    else:
        out_dtype = torch.float16
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=out_dtype)

    matmul(a, b, c)
    torch_output = torch.matmul(a_f16, b_f16)
    print(f"triton_output={c}")
    print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    size_str = f'size, (M: {M}, N: {N}, K: {K})'
    if torch.allclose(c.to(torch_output.dtype), torch_output, atol=1e-2, rtol=rtol):
        print(f'✅ Triton and Torch match for {size_str}')
    else:
        print(f'❌ Triton and Torch differ for {size_str}')


def run_speed(M, N, K, a_type, b_type, provider):
    a, a_f16 = gen_input(M, K, a_type, seed=10, device='cuda')
    b, b_f16 = gen_input(K, N, b_type, seed=13, device='cuda')

    if a_type == 'int8':
        assert a_type == b_type
        out_dtype = torch.int32
    elif a_type == 'fp32':
        assert a_type == b_type
        out_dtype = torch.float32
    else:
        out_dtype = torch.float16

    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=out_dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a_f16, b_f16), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c), quantiles=quantiles)
    return min_ms

def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args(print_help_info = False):
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("-dta", type=str, default='fp16', help="Input data type (fp8e4b8, fp8e5b16), default is fp16")
    parser.add_argument("-dtb", type=str, default='fp16', help="Input data type (fp8e4b8, fp8e5b16), default is fp16")
    parser.add_argument("--use_size_file", action='store_true', default=False, help="Whether user specify input matrix size")
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("-gemm_size_file", type=str, help='yaml file to indicate matrix size')
    parser.add_argument("--rocprof", action='store_true', default=False, help='Use rocprof to measure kernel time, default uses do_bench()!')
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    if print_help_info:
        parser.print_help()

    args = parser.parse_args()

    return args


def print_usage():
    print("Usage: matrix size can be specified in two ways:")
    print("     1) With \"--use_size_file\" option set, use --gemm_size_file for file of matrix sizes")
    print("     2) Without \"--use_size_file\" option,  use -m, -n, and -k to specify matrix size (default mode)")
    parse_args(True)


def main():
    args = parse_args()

    if ((args.use_size_file) and args.gemm_size_file is None) or \
       (not args.use_size_file and (args.m is None or args.n is None or args.k is None)):
        print_usage()
        sys.exit(1)

    a_type = args.dta
    b_type = args.dtb
    use_rocprof = args.rocprof
    verbose = args.v

    mnks = []
    if args.use_size_file:
        matrix_size_file = args.gemm_size_file
        if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
            print(f"Matrix size file: {matrix_size_file} does not exist!")
            print_usage()
            sys.exit(1)

        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)

        for sizes in matrix_sizes:
            M = sizes['M']
            N = sizes['N']
            K = sizes['K']
            mnks.append((M, N, K))
    else:
        M = args.m
        N = args.n
        K = args.k
        if M == 0 or N == 0 or K == 0:
            print(f"Input matrix size: (M {M}, N {N}, K {K}) contains dim size 0!")
            print_usage()
        mnks = [(M, N, K)]

    for (m, n, k) in mnks:
        min_ms = run_speed(m, n, k, a_type, b_type, 'triton')

        # function to compute flops
        perf_flops = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

        if args.compare:
            test_correctness(m, n, k, a_type, b_type)
        best_config = get_best_config(m, n, k)

        if use_rocprof:
            dtype_str = 'fp16' if (not args.specify_type) else args.dtype 
            block_m = best_config.kwargs['BLOCK_SIZE_M']
            block_n = best_config.kwargs['BLOCK_SIZE_N']
            block_k = best_config.kwargs['BLOCK_SIZE_K']
            group_m = best_config.kwargs['GROUP_SIZE_M']
            num_stages = best_config.num_stages
            num_warps = best_config.num_warps
            driver = 'rocprof_gemm.py'
            TRITON_DIR = os.getenv('TRITON_DIR')
            if TRITON_DIR is not None:
                driver = os.path.join(TRITON_DIR, 'scripts/amd/gemm', driver)
            run_cmd = f'python {driver} -m {m} -n {n} -k {k} \
                        -block_m {block_m} -block_n {block_n} -block_k {block_k} \
                        -group_m {group_m} -num_stages {num_stages} \
                        -num_warps {num_warps} -dtype {dtype_str}'
            prof_cmd = f'rocprof --stats {run_cmd}'
            run_bash_command(prof_cmd)

            parse_result_cmd = f'sed -n \'/matmul_kernel/p\' results.csv | awk -F \',\' \'{{print $NF}}\' | tail -n1'
            parse_outputs = run_bash_command(parse_result_cmd)
            min_ms = int(parse_outputs[0]) / 1000000

        out_str = f'SIZE: {m},{n},{k} '
        # print best config
        if verbose:
            out_str += f'  best_config: ({best_config}),   '
        out_str += f'TFLOPS: {perf_flops(min_ms)} time(ns): {min_ms * 1000000}'
        print(out_str)


if __name__ == '__main__':
    sys.exit(main())
