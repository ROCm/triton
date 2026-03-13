"""
Triton ROCm Bug: Incorrect results when two tl.dot() operations share an
intermediate tensor inside a loop with matrix_instr_nonkdim constraining the
MFMA instruction size.

Bug: When a loop body contains two tl.dot operations where:
  - dot1: [M, D] x [D, N] -> [M, N]  (produces intermediate)
  - dot2: [M, N] x [N, D] -> [M, D]  (consumes intermediate)
and N < 2 * MFMA_K_per_inst, the results are wildly incorrect (100s off).

The MFMA K-per-instruction depends on matrix_instr_nonkdim:
  - matrix_instr_nonkdim=16 -> MFMA_16x16x16_bf16 -> K_per_inst=16, N must be >= 32
  - matrix_instr_nonkdim=32 -> MFMA_32x32x8_bf16  -> K_per_inst=8,  N must be >= 16

Each tl.dot in isolation works correctly for all dimensions.
The bug only manifests when both dots execute in the SAME LOOP BODY.

Likely cause: register layout mismatch between the MFMA output of dot1 and
the expected input layout for dot2 when the shared dimension N is too small
for the configured MFMA tile.

Environment:
  - Triton: 3.5.1+rocm7.2.0.gita272dfa8
  - PyTorch: 2.9.1+rocm7.2.0.git7e1940d4
  - ROCm: 7.2.26015-fc0010cf6a
  - GPU: AMD Instinct MI350X (gfx942)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def two_dots_in_loop(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_q_m, stride_k_d, stride_k_n, stride_v_n,
    seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Two tl.dot() in a loop sharing an intermediate [M, N] tensor."""
    HEAD_DIM: tl.constexpr = 128
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.full([BLOCK_M, HEAD_DIM], 0.0, tl.float32)
    q = tl.load(Q_ptr + offs_m[:, None] * stride_q_m + offs_d[None, :])

    for offset_n in tl.range(0, seq_len, BLOCK_N):
        offs_n = offset_n + tl.arange(0, BLOCK_N)

        k = tl.load(K_ptr + offs_d[:, None] * stride_k_d + offs_n[None, :] * stride_k_n)
        # dot1: [M, 128] x [128, N] -> [M, N]
        qk = tl.dot(
            tl.cast(q, tl.bfloat16), tl.cast(k, tl.bfloat16),
            input_precision="ieee", out_dtype=tl.float32,
        )

        v = tl.load(V_ptr + offs_n[:, None] * stride_v_n + offs_d[None, :])
        # dot2: [M, N] x [N, 128] -> [M, 128]  (uses dot1 output as input)
        acc += tl.dot(
            tl.cast(qk, tl.bfloat16), tl.cast(v, tl.bfloat16),
            input_precision="ieee", out_dtype=tl.float32,
        )

    tl.store(
        Out_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        tl.cast(acc, tl.bfloat16),
    )


def reference(q, k, v, N):
    """Equivalent tiled computation in PyTorch."""
    seq_len = k.shape[1]
    M = q.shape[0]
    acc = torch.zeros(M, 128, dtype=torch.float32, device="cuda")
    for i in range(0, seq_len, N):
        qk = torch.mm(q.float(), k[:, i : i + N].float()).to(torch.bfloat16)
        acc += torch.mm(qk.float(), v[i : i + N, :].float())
    return acc.to(torch.bfloat16)


def test(M, N, seq_len, minkd):
    torch.manual_seed(42)
    q = torch.randn(M, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(128, seq_len, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(seq_len, 128, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, 128, dtype=torch.bfloat16, device="cuda")
    ref = reference(q, k, v, N)

    two_dots_in_loop[(1,)](
        q, k, v, out,
        q.stride(0), k.stride(0), k.stride(1), v.stride(0),
        seq_len,
        BLOCK_M=M, BLOCK_N=N,
        num_warps=1,
        matrix_instr_nonkdim=minkd,
    )

    diff = (out - ref).abs().max().item()
    status = "PASS" if diff < 1.0 else "FAIL"
    print(f"{status} minkd={minkd:2d} M={M:3d} N={N:3d} iters={seq_len // N:3d}: max_diff={diff:.2f}")
    return status == "PASS"


if __name__ == "__main__":
    print("Reproducer: Two tl.dot() in loop with shared intermediate tensor")
    print("dot1: [M, 128] x [128, N] -> [M, N]")
    print("dot2: [M, N]   x [N, 128] -> [M, 128]")
    print()

    all_pass = True
    # for minkd in [16, 32]:
    for minkd in [16]:
        print(f"--- matrix_instr_nonkdim={minkd} ---")
        # for M in [16, 32, 64]:
        for M in [16]:
            # for N in [8, 16, 32, 64]:
            for N in [8]:
                seq_len = max(N * 4, 64)
                passed = test(M, N, seq_len, minkd)
                all_pass = all_pass and passed
            print()

    if not all_pass:
        print("BUG REPRODUCED: Some configurations produce incorrect results.")
        print()
        print("Failure rule:")
        print("  minkd=16 -> fails when BLOCK_N < 32")
        print("  minkd=32 -> fails when BLOCK_N < 16")
        print()
        print("Each tl.dot works correctly in isolation.")
        print("Bug only manifests when BOTH dots are in the SAME loop body")
        print("and share the intermediate [M, N] tensor.")
