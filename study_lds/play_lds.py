import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.amd.cdna3 import sched_barrier, extract_slice

@gluon.jit
def get_pids(M, N, BM: gl.constexpr, BN: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
             GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K: gl.constexpr, stride_am, stride_ak,  #
           stride_bk, stride_bn,  #
           stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
           GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr  #
           ):

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    if BLOCK_N == 256: ## 128 x 256
        gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
            lane_bases=[[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]], warp_bases=[],
            block_bases=[], shape=[BLOCK_K, BLOCK_N])
        sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[1024, 16], [2048, 32]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
             [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8],
             [0, 128]], [],
            [BLOCK_K, BLOCK_N])
        num_warps: gl.constexpr = 1
    elif BLOCK_N == 512: ## 128 x 512
        gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 128], [0, 256]],
            lane_bases=[[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[], shape=[BLOCK_K, BLOCK_N])
        sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[1024, 4]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
             [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8],
             [0, 128], [0, 256]], [],
            [BLOCK_K, BLOCK_N])
        num_warps: gl.constexpr = 4

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                                    tiles_per_warp=[1, 1], warps_per_cta=[1, num_warps])

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=32)

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, dotOpLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, dotOpLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=32)

    nBuffers: gl.constexpr = gl.cdiv(160 * 1024, BLOCK_K * BLOCK_N) - 1

    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax: gl.constexpr = gl.cdiv(K, BLOCK_K)

    a = gl.amd.cdna4.buffer_load(a_base, a_offsets)


    for k in range(0, iterMax):

        cdna4_async_copy.buffer_load_to_shared(smemB.index(0), b_base, b_offsets)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(0)

        b = cdna4_async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)

        acc = gl.amd.cdna4.mfma_scaled(a, None, 'e5m2', b, None, 'e5m2', acc)

        b_base += BLOCK_K * stride_bk


    c = acc.to(tl.float16)
    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    gl.amd.cdna3.buffer_store(stored_value=c, ptr=c_base, offsets=c_offsets)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_type = {"fp16": torch.float16, "bf16": torch.bfloat16}

## Change this to 1 or 4 to run different experiments
num_warps = 4

def matmul(a, b, num_warps):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    BLOCK_M, BLOCK_K = 16, 128
    if num_warps == 1:
        BLOCK_N = 256
    elif num_warps == 4:
        BLOCK_N = 512
    assert K == BLOCK_K
    if a.dtype == torch.float8_e5m2:
        BLOCK_K = 128
    else:
        assert False
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GRID_MN=GRID_MN, NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M, num_warps=num_warps)
    return c


def get_x_vals():
    return [
        (4096, 4096, 128),
    ]


def test_correctness(dtype):

    if dtype == 'f8':
        torch_dtype = torch.float16
    else:
        torch_dtype = name_to_torch_type[dtype]

    for M, N, K in get_x_vals():
        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - .5
        b = torch.rand((N, K), device=DEVICE, dtype=torch_dtype).T - .5
        if dtype == 'f8':
            a = a.to(torch.float8_e5m2)
            b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b, num_warps)
        if dtype == 'f8':
            torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        else:
            torch_output = torch.matmul(a, b)
        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"{M=} {N=} {K=}: ✅ Triton and Torch match")
        else:
            print(f"{M=} {N=} {K=}: ❌ Triton and Torch differ")


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=get_x_vals(),
        line_arg="dtype",
        line_vals=["f8"],
        line_names=["f8"],
        styles=[("green", "-"), ("yellow", "--")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, dtype):
    if dtype == 'f8':
        torch_dtype = torch.float16
    else:
        torch_dtype = name_to_torch_type[dtype]
    a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
    b = torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T
    if dtype == 'f8':
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, num_warps), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


test_correctness("f8")
benchmark.run(show_plots=False, print_data=True)
