import hip

hip.hip.hipInit(0)

import pytest
import torch

import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                              M, N, K,  #
                              stride_am, stride_ak,  #
                              stride_bk, stride_bn,  #
                              stride_cm, stride_cn,  #
                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                              NUM_BUFFERS: ttgl.constexpr,  #
                              NUM_WARPS: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [NUM_WARPS // 2, 2], [16, 16, 32])
    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0])
    SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_K, BLOCK_N],
                                                                                [1, 0])
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=a_ptr + pid_m * BLOCK_M * stride_am,  #
        shape=(M, K),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(BLOCK_M, BLOCK_K),  #
        layout=SHARED_LAYOUT_A)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=b_ptr + pid_n * BLOCK_N * stride_bn,  #
        shape=(K, N),  #
        strides=(stride_bk, stride_bn),  #
        block_shape=(BLOCK_K, BLOCK_N),  #
        layout=SHARED_LAYOUT_B)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    load_idx = 0
    wmma_idx = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K],  #
                                        a_buffer.index(load_idx % NUM_BUFFERS))
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K, 0],  #
                                        b_buffer.index(load_idx % NUM_BUFFERS))
        load_idx += 1

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K],  #
                                        a_buffer.index(load_idx % NUM_BUFFERS))
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K, 0],  #
                                        b_buffer.index(load_idx % NUM_BUFFERS))
        load_idx += 1

        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
        wmma_idx += 1

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
        wmma_idx += 1

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (250, 250, 510)])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, M, N, K, num_warps):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    gemm_tdm_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, NUM_WARPS=num_warps,  #
        num_warps=num_warps, waves_per_eu=num_warps // 4)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    M, N, K = 256, 256, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    NUM_BUFFERS = 2
    NUM_WARPS = 4
    print(f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {NUM_BUFFERS=}, {NUM_WARPS=}")
    test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, M, N, K, NUM_WARPS)
    NUM_WARPS = 8
    print(f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {NUM_BUFFERS=}, {NUM_WARPS=}")
    test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, M, N, K, NUM_WARPS)
