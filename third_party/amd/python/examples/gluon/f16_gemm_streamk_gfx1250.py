# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import pytest
import torch
import math
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
    from .f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        TileScheduler,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        TileScheduler,
    )


@gluon.jit
def streamk_gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr, p_ptr, locks_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                      stride_bn, stride_cm, stride_cn, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                      BLOCK_K: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr,
                                      NUM_WARPS: ttgl.constexpr, WARP_BASES: ttgl.constexpr,
                                      STREAMK_TILES: ttgl.constexpr, GROUP_SIZE_M: ttgl.constexpr = 8):
    """
    StreamK GEMM kernel with TDM and software pipelining.
    When STREAMK_TILES=0: Behaves exactly like persistent_gemm_tdm_pipelined_kernel
    When STREAMK_TILES>0: Adds StreamK processing after full tiles

    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler with STREAMK_TILES
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # ============================================================================
    # Phase 1: Process full tiles (persistent scheduling - same as f16_gemm_gfx1250.py)
    # ============================================================================
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_full_tiles = scheduler.get_num_full_tiles()

    # Enable chiplet transformation (8 XCDs) to improve l2 reuse
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=8, chunk_size=2)

    # Persistent loop: each CU processes its assigned tiles with stride NUM_SMS
    for tile_idx in range(pid, num_full_tiles, num_sms):
        # get_swizzled_tile_coords: local tile index -> global (pid_m, pid_n) with swizzling
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        # Prefill pipeline
        for _ in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        # Steady state: overlap load and compute
        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

        # Drain pipeline
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

        # Store result
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    #============================================================================
    # Phase 2: Process StreamK tiles (remainder tiles)
    # ============================================================================
    if STREAMK_TILES == 0:
        return

    # Initialize P buffer and locks
    rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    p_offset = pid * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
    ttgl.store(p_ptr + p_offset, ttgl.zeros((BLOCK_M, BLOCK_N), dtype=p_ptr.type.element_ty, layout=WMMA_LAYOUT))
    ttgl.store(locks_ptr + pid, 0)

    iters_per_tile = scheduler.get_iters_per_tile()
    total_streamk_iters, streamk_iters_pcu, streamk_remainder_iters = scheduler.get_streamk_params()
    start_iter, last_iter = scheduler.get_streamk_iteration_range()

    current_start_iter = start_iter
    while current_start_iter < last_iter:
        remainder = current_start_iter % iters_per_tile
        end_iter = ttgl.minimum(current_start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = current_start_iter // iters_per_tile
        tile_iter = tile_id * iters_per_tile

        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_id, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)
        num_k_iters = end_iter - current_start_iter

        for k_idx in range(num_k_iters):
            k_offset = (remainder + k_idx) * BLOCK_K

            # Load FULL blocks
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_offset], a_buffer.index(0))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn], b_buffer.index(0))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_offset], b_buffer.index(0))

            ttgl.amd.gfx1250.tdm.async_wait(0)

            a_operand = a_buffer.index(0).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(0).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = b_buffer.index(0).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)

            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)

        # ====================================================================
        # Contributor or Owner logic (quadrant accumulation for aggregation)
        # ====================================================================
        if current_start_iter != tile_iter:
            # ============================================================
            # Contributor: Store FULL accumulator to P buffer
            # ============================================================
            rm1 = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
            rn1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
            p_offset = pid * BLOCK_M * BLOCK_N + rm1[:, None] * BLOCK_N + rn1[None, :]

            ttgl.store(p_ptr + p_offset, accumulator)
            ttgl.barrier()
            ttgl.atomic_xchg(locks_ptr + pid, 1)

        else:
            # ============================================================
            # Owner: Aggregate contributors and store result
            # - Use quadrant aggregation for 256x256 to reduce spills
            # - Use full accumulator for smaller tiles to avoid parsing issues
            # ============================================================
            next_pid = pid + 1
            end = end_iter

            if BLOCK_M == 256 and BLOCK_N == 256:
                # Split accumulator into quadrants (top-left, top-right, bottom-left, bottom-right)
                acc_4d = accumulator.reshape([2, BLOCK_M // 2, 2, BLOCK_N // 2])
                acc_4d = acc_4d.permute(1, 3, 0, 2)
                acc_n0, acc_n1 = acc_4d.split()
                acc_00, acc_10 = acc_n0.split()
                acc_01, acc_11 = acc_n1.split()

                # Use default layouts for quadrant indexing to match sliced accumulator layout
                rm_q = ttgl.arange(0, BLOCK_M // 2)
                rn_q = ttgl.arange(0, BLOCK_N // 2)

                # Aggregate from contributors using quadrant loads to reduce pressure
                while (end < tile_iter + iters_per_tile and next_pid < num_sms):
                    # Wait for contributor
                    while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                        pass

                    P_base = p_ptr + next_pid * BLOCK_M * BLOCK_N

                    # Quadrant 00 (top-left)
                    p00 = P_base + rm_q[:, None] * BLOCK_N + rn_q[None, :]
                    acc_00 += ttgl.load(p00)

                    # Quadrant 01 (top-right)
                    p01 = P_base + rm_q[:, None] * BLOCK_N + (rn_q[None, :] + BLOCK_N // 2)
                    acc_01 += ttgl.load(p01)

                    # Quadrant 10 (bottom-left)
                    p10 = P_base + (rm_q[:, None] + BLOCK_M // 2) * BLOCK_N + rn_q[None, :]
                    acc_10 += ttgl.load(p10)

                    # Quadrant 11 (bottom-right)
                    p11 = P_base + (rm_q[:, None] + BLOCK_M // 2) * BLOCK_N + (rn_q[None, :] + BLOCK_N // 2)
                    acc_11 += ttgl.load(p11)

                    end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                    next_pid += 1

                # Store aggregated quadrants to output
                rm_top = pid_m * BLOCK_M + rm_q
                rm_bottom = pid_m * BLOCK_M + (rm_q + BLOCK_M // 2)
                rn_left = pid_n * BLOCK_N + rn_q
                rn_right = pid_n * BLOCK_N + (rn_q + BLOCK_N // 2)

                mask00 = (rm_top[:, None] < M) & (rn_left[None, :] < N)
                offs00 = stride_cm * rm_top[:, None] + stride_cn * rn_left[None, :]
                ttgl.store(c_ptr + offs00, acc_00, mask=mask00)

                mask01 = (rm_top[:, None] < M) & (rn_right[None, :] < N)
                offs01 = stride_cm * rm_top[:, None] + stride_cn * rn_right[None, :]
                ttgl.store(c_ptr + offs01, acc_01, mask=mask01)

                mask10 = (rm_bottom[:, None] < M) & (rn_left[None, :] < N)
                offs10 = stride_cm * rm_bottom[:, None] + stride_cn * rn_left[None, :]
                ttgl.store(c_ptr + offs10, acc_10, mask=mask10)

                mask11 = (rm_bottom[:, None] < M) & (rn_right[None, :] < N)
                offs11 = stride_cm * rm_bottom[:, None] + stride_cn * rn_right[None, :]
                ttgl.store(c_ptr + offs11, acc_11, mask=mask11)
            else:
                # Aggregate full accumulator for smaller tiles
                rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
                rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
                offs_m = pid_m * BLOCK_M + rm
                offs_n = pid_n * BLOCK_N + rn

                while (end < tile_iter + iters_per_tile and next_pid < num_sms):
                    while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                        pass

                    p_offset_load = next_pid * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
                    contrib_acc = ttgl.load(p_ptr + p_offset_load)
                    accumulator += contrib_acc

                    end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                    next_pid += 1

                # Store aggregated result to output (reuse rm1/rn1 derived offsets)
                mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                ttgl.store(c_ptr + offs_c, accumulator, mask=mask)

        current_start_iter = end_iter


def run_streamk_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, STREAMK_TILES, M, N, K,
                                   num_warps):
    """Helper function for main block - takes explicit STREAMK_TILES parameter."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        print(f"Skipping: K/BLOCK_K ({triton.cdiv(K, BLOCK_K)}) < NUM_BUFFERS ({NUM_BUFFERS})")
        return

    # Validate STREAMK_TILES
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if STREAMK_TILES > total_tiles:
        raise ValueError(f"STREAMK_TILES ({STREAMK_TILES}) cannot exceed total_tiles ({total_tiles})")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Use persistent grid (StreamK uses persistent kernel infrastructure)
    num_sms = 8
    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)

    # Allocate StreamK buffers
    p = torch.empty(num_sms * BLOCK_M * BLOCK_N, dtype=torch.float32)
    locks = torch.empty(num_sms, dtype=torch.int32)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    p_device = p.cuda()
    locks_device = locks.cuda()

    print(f"\nTesting StreamK kernel with STREAMK_TILES={STREAMK_TILES}")
    print(f"Grid: {grid}, Total tiles: {total_tiles}")

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    kernel = streamk_gemm_tdm_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        p_device, locks_device, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=num_warps, WARP_BASES=warp_bases,  #
        STREAMK_TILES=STREAMK_TILES, num_warps=num_warps, waves_per_eu=num_warps // 4)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    print("✓ StreamK kernel test passed!")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (258, 258, 510)])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_streamk_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, num_warps):
    """Test StreamK GEMM kernel with K-dimension splitting."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    # Calculate STREAMK_TILES: Use remainder tiles for load balancing
    num_sms = 8
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    STREAMK_TILES = total_tiles % num_sms

    # Call the helper function
    run_streamk_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, STREAMK_TILES, M, N, K,
                                   num_warps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='StreamK GEMM kernel test - automatically calculates StreamK tiles for load balancing',
        epilog='Example: python3 f16_sk_gemm_gfx1250.py -M 258 -N 258 -K 510')
    parser.add_argument("-M", type=int, default=1028, help='problem M size (default: 258)')
    parser.add_argument("-N", type=int, default=1028, help='problem N size (default: 258)')
    parser.add_argument("-K", type=int, default=1024, help='problem K size (default: 510)')
    parser.add_argument("--block-m", type=int, default=256, help='BLOCK_M tile size (default: 32)')
    parser.add_argument("--block-n", type=int, default=256, help='BLOCK_N tile size (default: 32)')
    parser.add_argument("--block-k", type=int, default=128, help='BLOCK_K tile size (default: 128)')
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4, help='num warps (default: 4)')
    parser.add_argument("--num-buffers", type=int, choices=[2, 4], default=2,
                        help='num shared memory buffers (default: 2)')
    parser.add_argument("--num-sms", type=int, default=8, help='number of SMs to use (default: 8)')
    parser.add_argument("--streamk-tiles", type=int, default=None, metavar='N',
                        help='Override StreamK tiles count (default: auto = total_tiles %% num_sms)')
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = args.num_warps
    TRANSPOSE_B = True
    NUM_SMS = args.num_sms

    # Calculate tile dimensions
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)

    # Calculate STREAMK_TILES (remainder tiles for load balancing)
    if args.streamk_tiles is not None:
        STREAMK_TILES = args.streamk_tiles
    else:
        STREAMK_TILES = total_tiles % NUM_SMS

    print(f"StreamK mode: STREAMK_TILES={STREAMK_TILES} (out of {total_tiles} total tiles)")
    print(f"Mode: StreamK with {STREAMK_TILES} StreamK tiles")
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, PERSISTENT=True, STREAMK=True, PREFETCH=False"
    )

    # Run StreamK kernel test
    run_streamk_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K,  #
                                   NUM_BUFFERS, TRANSPOSE_B, STREAMK_TILES,  #
                                   M, N, K, NUM_WARPS)
