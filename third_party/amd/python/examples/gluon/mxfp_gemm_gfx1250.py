# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import torch
import pytest
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import numpy as np


def generate_configs():
    configs = []
    # Add many small shapes.
    dtypes = [['float8_e5m2', 'float4'], ['float4', 'float8_e4m3'], ['float8_e4m3', 'float8_e5m2'],
              ['float4', 'float4']]
    for dtypeA, dtypeB in dtypes:
        for (M, N, K, BM, BN, BK) in [(1024, 1024, 128, 64, 64, 64), (1024, 1024, 128, 64, 64, 128)]:
            for b_trans in (True, False):
                for num_buffers in (2, 4):
                    # For correctness, we need masking when not using exact tiles.
                    # python3: /home/dtanner/repos/gfx_triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp:233: mlir::Value mlir::triton::AMD::{anonymous}::generateScaledWMMAIntrinsic(mlir::ConversionPatternRewriter&, mlir::Location, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Type, mlir::Type, mlir::Type, int): Assertion `scaleKWidth == 2 ||     scaleKWidth == 4 || scaleKWidth == 8' failed.
                    if dtypeA == 'float4' and BK < K:
                        continue
                    # Temporarily skip for small block K
                    if BK < 128:
                        continue
                    configs.append({
                        "M": M, "N": N, "K": K, "BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "NUM_WARPS": 4, "NUM_CTAS":
                        1, "SCALE_BLOCK": 32, "DTYPE_A": dtypeA, "DTYPE_B": dtypeB, "TRANSPOSE_B": b_trans,
                        "NUM_BUFFERS": num_buffers
                    })
    return configs


@gluon.jit
def mxgemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk,
                                stride_bn, stride_cm, stride_cn, stride_scale, DTYPE_A: gl.constexpr,
                                DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                TRANSPOSE_B: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                     instr_shape=[16, 16, 128])
    WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                            instr_shape=[16, 16, 64])
    A_SCALE_LINEAR_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[0, 0], [16, 0]],
        block_bases=[], shape=[32, 4])
    B_SCALE_LINEAR_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[16, 0], [0, 0]],
        block_bases=[], shape=[32, 4])

    DOT_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    DOT_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(operand_index=1,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    if TRANSPOSE_B:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                                [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]],
                                                                                [BLOCK_K_PACKED_B, BLOCK_N], [1, 0])

    SHARED_LAYOUT_A_SCALE: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    SHARED_LAYOUT_B_SCALE: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am,
                                                       shape=(M, K // DIV_FACTOR_A), strides=(stride_am, stride_ak),
                                                       block_shape=(BLOCK_M, BLOCK_K_PACKED_A), layout=SHARED_LAYOUT_A)
    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)

    if TRANSPOSE_B:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn,
                                                           shape=(N, K // DIV_FACTOR_B), strides=(stride_bn, stride_bk),
                                                           block_shape=(BLOCK_N, BLOCK_K_PACKED_B),
                                                           layout=SHARED_LAYOUT_B)
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn,
                                                           shape=(K // DIV_FACTOR_B, N), strides=(stride_bk, stride_bn),
                                                           block_shape=(BLOCK_K_PACKED_B, BLOCK_N),
                                                           layout=SHARED_LAYOUT_B)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    a_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_scale + pid_m * BLOCK_M * stride_scale,
                                                             shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                                             block_shape=(BLOCK_M, BLOCK_K_SCALE),
                                                             layout=SHARED_LAYOUT_A_SCALE)
    a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                               layout=a_scale_desc.layout)

    b_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_scale + pid_n * BLOCK_N * stride_scale,
                                                             shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                                             block_shape=(BLOCK_N, BLOCK_K_SCALE),
                                                             layout=SHARED_LAYOUT_B_SCALE)
    b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                               layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    # prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K_PACKED_A], a_buffer.index(load_idx))
        if TRANSPOSE_B:
            gl.amd.gfx1250.tdm.async_load(b_desc, [0, load_idx * BLOCK_K_PACKED_B], b_buffer.index(load_idx))
        else:
            gl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K_PACKED_B, 0], b_buffer.index(load_idx))
        gl.amd.gfx1250.tdm.async_load(a_scale_desc, [0, load_idx * BLOCK_K_SCALE], a_scale_buffer.index(load_idx))
        gl.amd.gfx1250.tdm.async_load(b_scale_desc, [0, load_idx * BLOCK_K_SCALE], b_scale_buffer.index(load_idx))
        load_idx += 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for _ in range(0, gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K_PACKED_A], a_buffer.index(load_idx % NUM_BUFFERS))
        if TRANSPOSE_B:
            gl.amd.gfx1250.tdm.async_load(b_desc, [0, load_idx * BLOCK_K_PACKED_B],
                                          b_buffer.index(load_idx % NUM_BUFFERS))
        else:
            gl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K_PACKED_B, 0],
                                          b_buffer.index(load_idx % NUM_BUFFERS))
        gl.amd.gfx1250.tdm.async_load(a_scale_desc, [0, load_idx * BLOCK_K_SCALE],
                                      a_scale_buffer.index(load_idx % NUM_BUFFERS))
        gl.amd.gfx1250.tdm.async_load(b_scale_desc, [0, load_idx * BLOCK_K_SCALE],
                                      b_scale_buffer.index(load_idx % NUM_BUFFERS))

        load_idx += 1

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 4)

        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_A)
        if TRANSPOSE_B:
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=DOT_LAYOUT_B)
        else:
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_B)
        scale_a = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=A_SCALE_LINEAR_LAYOUT)
        scale_b = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=B_SCALE_LINEAR_LAYOUT)

        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        wmma_idx += 1

    # epilogue
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 4)
        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_A)
        if TRANSPOSE_B:
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=DOT_LAYOUT_B)
        else:
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_B)
        scale_a = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=A_SCALE_LINEAR_LAYOUT)
        scale_b = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=B_SCALE_LINEAR_LAYOUT)
        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        wmma_idx += 1

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.store(c_ptrs, accumulator, mask=c_mask)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    return torch.matmul(a_f32 * a_scale_f32, b_f32 * b_scale_f32).to(torch.float32)


def init_data(dtype, d0: int, d1: int):
    if dtype == 'float4':
        return MXFP4Tensor(size=(d0, d1)).random()
    elif dtype == "float8_e5m2":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e5m2)
    elif dtype == "float8_e4m3":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        raise NotImplementedError(f"NYI: unsupported dtype: {dtype}")


def run(config):
    print(config)
    M = config["M"]
    N = config["N"]
    K = config["K"]
    blockSizeM = config["BLOCK_M"]
    blockSizeN = config["BLOCK_N"]
    blockSizeK = config["BLOCK_K"]
    numCtas = config['NUM_CTAS']
    numWarps = config['NUM_WARPS']
    dtype_a = config['DTYPE_A']
    dtype_b = config['DTYPE_B']
    scale_block = config['SCALE_BLOCK']
    TRANSPOSE_B = config['TRANSPOSE_B']
    NUM_BUFFERS = config['NUM_BUFFERS']

    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K)
    b = init_data(dtype_b, K, N)
    a_scale_size = (M, (K + scale_block - 1) // scale_block)
    b_scale_size = (N, (K + scale_block - 1) // scale_block)
    a_scale = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    b_scale = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    a_scale = a_scale.data
    b_scale = b_scale.data

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype_a in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if dtype_b in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    if TRANSPOSE_B:
        b_d = b.data.T.contiguous().cuda()
    else:
        b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk = b_d.stride(1) if TRANSPOSE_B else b_d.stride(0)
    stride_bn = b_d.stride(0) if TRANSPOSE_B else b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = a_scale_d.stride(0)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                      stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[dtype_a],
                                      dtype_converter[dtype_b], scale_block, blockSizeM, blockSizeN, blockSizeK,
                                      group_size_m, TRANSPOSE_B, NUM_BUFFERS, num_warps=numWarps, num_ctas=numCtas)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
    print('✅Pass')


@pytest.mark.parametrize("config", generate_configs())
def test_runtime_mxgemm_tdm_pipelined(config):
    print(config)
    M = config["M"]
    N = config["N"]
    K = config["K"]
    blockSizeM = config["BLOCK_M"]
    blockSizeN = config["BLOCK_N"]
    blockSizeK = config["BLOCK_K"]
    numCtas = config['NUM_CTAS']
    numWarps = config['NUM_WARPS']
    dtype_a = config['DTYPE_A']
    dtype_b = config['DTYPE_B']
    scale_block = config['SCALE_BLOCK']
    TRANSPOSE_B = config['TRANSPOSE_B']
    NUM_BUFFERS = config['NUM_BUFFERS']

    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K)
    b = init_data(dtype_b, K, N)
    a_scale_size = (M, (K + scale_block - 1) // scale_block)
    b_scale_size = (N, (K + scale_block - 1) // scale_block)
    a_scale_mxfp4 = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    b_scale_mxfp4 = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale_mxfp4, b_scale_mxfp4, scale_block, M, N, K)

    a_scale = a_scale_mxfp4.data
    b_scale = b_scale_mxfp4.data

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype_a in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if dtype_b in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    if TRANSPOSE_B:
        b_d = b.data.T.contiguous().cuda()
    else:
        b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    if TRANSPOSE_B:
        stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    else:
        stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = a_scale_d.stride(0)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    k = mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                          stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[dtype_a],
                                          dtype_converter[dtype_b], scale_block, blockSizeM, blockSizeN, blockSizeK,
                                          group_size_m, TRANSPOSE_B, NUM_BUFFERS, num_warps=numWarps, num_ctas=numCtas)

    if TRANSPOSE_B:
        assert 'ds_load_u8' not in k.asm['amdgcn']

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    for dtypeA, dtypeB in (("float8_e5m2", "float4"), ("float8_e4m3", "float8_e5m2"), ("float4", "float4")):
        config = {
            "M": 256, "N": 256, "K": 1024, "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "NUM_WARPS": 4, "NUM_CTAS":
            1, "SCALE_BLOCK": 32, "DTYPE_A": dtypeA, "DTYPE_B": dtypeB, "TRANSPOSE_B": True, "NUM_BUFFERS": 2
        }
        run(config)
