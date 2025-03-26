import torch
import triton
import triton.language as tl
import re

DEBUG = True
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
num_xcds = 4


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == 'gfx950'


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e5': e5m2_type,
    'fp8e4': e4m3_type,
}


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
                'kpack': 1, 'matrix_instr_nonkdim': 16
            }, num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'kpack': 1, 'matrix_instr_nonkdim': 16
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'waves_per_eu': 0,
                'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'waves_per_eu': 0,
                'kpack': 1
            }, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit()
def persistent_gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_sms: int,
    num_xcds=4,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    grids = num_sms

    kk = persistent_gemm[(grids, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
    )
    if DEBUG:
        print(f"{kk.n_regs} registers used, {kk.n_spills} spills")

    return c


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1].split('/', 1)


def get_x_vals():
    x_vals = [
        # GEMMRS configurations
        (8192, 4096, 11008),  # LLaMA-7B
        (8192, 3584, 14336),  # Gemm2-9B
        (8192, 4608, 36864),  # Gemma2-27B
        (8192, 8192, 28672),  # LLaMA-3.1-70B
        (8192, 8192, 29568),  # Qwen2-72B

        # AGGEMM configurations
        (8192, 11008, 4096),  # LLaMA-7B
        (8192, 14336, 4096),  # LLaMA-3.1-8B/Mistral-7B
        (8192, 28672, 8192),  # LLaMA-3.1-70B
        (8192, 53248, 16384),  # LLaMA-3.1-405B
        (8192, 29568, 8192)  # Qwen2-72B
    ]

    return x_vals


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K', 'num_sms', 'num_xcds'],
        x_vals=[tuple(list(size) + [num_sms, num_xcds]) for size in get_x_vals()],
        line_arg='provider',
        line_vals=['hipblaslt(fp16/fp16)', 'hipblaslt(bf16/bf16)', 'triton(fp16/fp16)', 'triton(bf16/bf16)'],
        line_names=["rocBLAS.Fp16", "rocBLAS.Bf16", "Triton.Fp16", "Triton.Bf16"],
        ylabel="TFLOPS",
        plot_name="persistent-gemm-performance",
        args={},
    ))
def benchmark(M, N, K, num_sms, num_xcds, provider, model=None, args=None):
    in_dtype_a, in_dtype_b = [name_to_torch_types[x] for x in get_type(provider)]
    out_dtype = in_dtype_a

    quantiles = [0.5, 0.2, 0.8]
    A = torch.randn(M, K, device="cuda", dtype=in_dtype_a)
    B = torch.randn(N, K, device="cuda", dtype=in_dtype_b).T

    if 'hipblaslt' in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
    else:  # triton, different data types
        assert "triton" in provider
        # Allocates output.
        C = torch.zeros((M, N), device="cuda", dtype=out_dtype)

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: persistent_matmul(A, B, C, num_sms, num_xcds),
                                                     quantiles=quantiles)
        if DEBUG:
            print(f'Best tuning config for M={M}, N={N}, K={K}, '
                  f'dtype={in_dtype_a} / {in_dtype_b} / {out_dtype}: \n({persistent_gemm.best_config})\n')
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
