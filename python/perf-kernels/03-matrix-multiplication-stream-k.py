#!/usr/bin/env python
## matmul stream-k implementation
## Credit goes to @pommedeterresautee
## See https://github.com/openai/triton/issues/1393

# (echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"') | sudo tee -a /etc/modprobe.d/RestrictedProfiling.conf >/dev/null
# sudo update-initramfs -u -k all
# cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# sudo apt-get install zlib1g-dev
# for reproductible experiments
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005
from typing import Optional

import torch
import triton
import triton.language as tl
import random

#from triton.runtime.driver import CudaUtils
import json

torch.manual_seed(123)
random.seed(123)

#device = torch.cuda.current_device()
#cuda_utils = CudaUtils()
#total_sm = cuda_utils.get_device_properties(device)["multiprocessor_count"]
#total_sm = 110 # for MI250
total_sm = 304 # for MI300X
print(f"total SMs: {total_sm}")

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit()
def swizzle_tile(tile_id,
                 M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr
                 ):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit()
def linear_tile(tile_id,
                M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr
                ):
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n

@triton.jit()
def streamk_gemm(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile,
        total_tiles_streamk, total_programs_streamk, ACC_TYPE: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        # where are we in the grid
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M  == 1:
            pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)
        else:
            pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for current_iter in range(start_iter, end_iter):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        if remainder ==0 and end_iter % iters_per_tile ==0:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, acc, mask=mask)
        else:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.atomic_add(C_, acc, mask=mask)

        start_iter = end_iter
# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class matmul(torch.autograd.Function):

    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, total_programs_streamk: int, BLK_M: int, BLK_N: int, BLK_K: int, gsize_m: int, two_tiles: bool, num_stages: int, num_warps: int, waves_per_eu: int,  mfmaInstrSize: int, kpack: int):
        device = a.device

#        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # compute grid (work to do per SM on the first wave)
        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)

        total_tiles = total_blocks_M * total_blocks_N

        total_tiles_streamk = total_tiles 
        total_blocking_tiles = total_tiles - total_tiles_streamk
        total_iters_streamk = total_tiles * iters_per_tile
        # iterations related to full waves
        total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
        # iterations related to last (partial) wave
        total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
        grids = total_programs_streamk


        if matmul._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{total_full_tiles_streamk=}")
            print(f"{total_partial_tiles_streamk=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")

        # allocates locks to sync work accross SMs
  #      grids = total_programs_streamk + total_blocking_tiles
  #      print(f"{grids=}")
        kk = streamk_gemm[(grids,)](
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
            total_full_tiles_streamk=total_full_tiles_streamk,
            total_partial_tiles_streamk=total_partial_tiles_streamk,
            iters_per_tile=iters_per_tile,
            total_tiles_streamk=total_tiles_streamk,
            total_programs_streamk= total_programs_streamk,
            ACC_TYPE=ACC_TYPE,
            GROUP_SIZE_M=gsize_m,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu = waves_per_eu,
            matrix_instr_nonkdim = mfmaInstrSize,
            kpack = kpack,
        )
        if matmul._debug:
            print(f"{kk.n_regs} registers used, {kk.n_spills} spills")
       #     print(kk.asm['ttgir'])
       #     print(kk.asm['amdgcn'])

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, grid: int, BLK_M = 128, BLK_N = 128, BLK_K = 32, gsize_m = 1, two_tiles = True, num_stages = 3, num_warps = 4,  waves_per_eu = 2, mfmaInstrSize = 16, kpack = 1):
        matmul._call(a = a, b = b, c = c, total_programs_streamk = grid, BLK_M = BLK_M, BLK_N = BLK_N, BLK_K = BLK_K, gsize_m = gsize_m, two_tiles = two_tiles, num_warps = num_warps, num_stages = num_stages,  waves_per_eu = waves_per_eu, mfmaInstrSize = mfmaInstrSize, kpack = kpack)
        return c

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

m, n, k = 4864, 4096, 8256  # some problem size to test
#m, n, k = 4096, 4096, 8192  # some problem size to test
#m, n, k = 8192, 8192, 8192  # some problem size to test
#m, n, k = 6912, 768, 256  # some problem size to test
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
# allocates output
C = torch.zeros((m, n), device="cuda", dtype=A.dtype)
BLK_M = 128
BLK_N = 256
BLK_K = 64
gsize_m = 4
two_tiles = 'True'
num_stages = 0
num_warps = 4
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2

matmul.set_debug(True)
C = matmul.apply(A, B, C, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack)
matmul.set_debug(False)
expected = A @ B

assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
print("pass validation test")

triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
print(f"PyTorch: {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, C, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack))
print(f"hybrid stream-k (grid={total_sm}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, C, total_sm * 2, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack))
print(f"hybrid stream-k (grid={total_sm * 2}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

exit(0)
# ---------------------------------------------------------------------------
# Log-sampled benchmark
# ---------------------------------------------------------------------------

# tried to reproduce the tests described in the paper
num_samples = 1000  # 32768
step = 256
values = ((torch.logspace(torch.tensor(step).log2(), torch.tensor(8192).log2(), num_samples, base=2) / step).round() * step).unique().tolist()
shapes = [(int(m), int(n), int(k)) for m in values for n in values for k in values]
shapes = random.sample(shapes, num_samples)
assert len(shapes) == num_samples

results = []
for idx, (m, n, k) in enumerate(shapes):
    # print progress bar
    if idx % 10 == 0 and idx > 0:
        speedups = [r["speedup"] for r in results]
        print(f"{idx}/{num_samples} - average speedup: {sum(speedups) / len(speedups):.3f}")

    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    output: Optional[torch.Tensor] = torch.zeros((m, n), device="cuda", dtype=A.dtype)


    def wrapper_matmul(*args, **kwargs):
        global output
        output = matmul.apply(*args, **kwargs)
        return output


    expected = A @ B
    pytorch_ms = triton.testing.do_bench(lambda: A @ B)
    measures = list()
    for two_tiles in [True, False]:
        nb_sm = [total_sm, total_sm * 2]
        total_tile = (m // BLK_M) * (n // BLK_N)
        if total_tile < total_sm * 2:
            nb_sm.append(total_tile)
        nb_sm += random.sample(range(2, total_sm * 2, 2), 10)
        for sm in nb_sm:
            triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, output, sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack))
            C = torch.zeros((m, n), device="cuda", dtype=A.dtype)
            C = matmul.apply(A, B, C, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack)
            max_disc = (C - expected).abs().max().item()
            # large tolerance to accomodate for large K (rounding due to half precision), we just want to catch bugs.
            assert max_disc <= 5., f"pb size: {m}x{n}x{k} - max discrepancy: {max_disc} - sm: {sm}, 2 tiles: {two_tiles}\n{output}\n{expected}"
            info = {
                "2 tiles": two_tiles,
                "sm": sm,
                "disc": max_disc,
                "triton_ms": triton_ms,
            }
            measures.append(info)
    best_triton_ms = min([m["triton_ms"] for m in measures])
    d = {
        "m": m,
        "n": n,
        "k": k,
        "triton": measures,
        "pytorch_ms": pytorch_ms,
      #  "speedup": pytorch_ms / best_triton_ms,
        "speedup": best_triton_ms / pytorch_ms,
    }
    results.append(d)
    measures = list()

results.sort(key=lambda x: x["speedup"], reverse=False)

# ---------------------------------------------------------------------------
# Benchmark export
# ---------------------------------------------------------------------------

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# 32760/32768 - average speedup: 0.962 (A100)
# 990/1000 - average speedup: 1.063 (3090 RTX with while loop and 2 tiles disabled / enabled)
