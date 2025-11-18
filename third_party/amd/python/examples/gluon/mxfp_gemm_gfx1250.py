# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import torch
import pytest
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import numpy as np


def generate_configs():
    configs = []
    # Add many small shapes.
    dtypes = [['float8_e5m2', 'float4'], ['float4', 'float8_e4m3'], ['float8_e4m3', 'float8_e5m2'],
              ['float4', 'float4']]
    for dtypeA, dtypeB in dtypes:
        for (M, N, K, BM, BN, BK) in [(1024, 1024, 128, 64, 64, 64), (1024, 1024, 128, 64, 64, 128),
                                      (1024, 1024, 128, 128, 128, 128)]:
            for b_trans in (True, False):
                for num_buffers in (2, 4):
                    for scale_preshuffle in (True, False):
                        # For correctness, we need masking when not using exact tiles.
                        # python3: /home/dtanner/repos/gfx_triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp:233: mlir::Value mlir::triton::AMD::{anonymous}::generateScaledWMMAIntrinsic(mlir::ConversionPatternRewriter&, mlir::Location, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Type, mlir::Type, mlir::Type, int): Assertion `scaleKWidth == 2 ||     scaleKWidth == 4 || scaleKWidth == 8' failed.
                        if dtypeA == 'float4' and BK < K:
                            continue

                        # skip block sizes too small for preshuffling
                        if scale_preshuffle and (BM < 128 or BN < 128 or BK < 128):
                            continue

                        configs.append({
                            "M": M, "N": N, "K": K, "BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "NUM_WARPS": 4,
                            "NUM_CTAS": 1, "SCALE_BLOCK": 32, "DTYPE_A": dtypeA, "DTYPE_B": dtypeB, "TRANSPOSE_B":
                            b_trans, "NUM_BUFFERS": num_buffers, "SCALE_PRESHUFFLE": scale_preshuffle, 'WITH_A_SCALE':
                            True
                        })
    return configs


@aggregate
class MXFPGEMMConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    DIV_FACTOR_A: gl.constexpr
    DIV_FACTOR_B: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    TRANSPOSE_B: gl.constexpr
    WITH_A_SCALE: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr

    # Layouts
    shared_layout_a: gl.constexpr
    dot_layout_a: gl.constexpr

    shared_layout_b: gl.constexpr
    dot_layout_b: gl.constexpr

    shared_layout_a_scale: gl.constexpr
    layout_a_scale: gl.constexpr

    shared_layout_b_scale: gl.constexpr
    layout_b_scale: gl.constexpr

    acc_layout: gl.constexpr

    # Scales
    SCALE_PRESHUFFLE: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    SCALE_KWIDTH: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    tiles_per_warp: gl.constexpr
    SCALE_BLOCK: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, NUM_BUFFERS, TRANSPOSE_B, WITH_A_SCALE,
                 SCALE_PRESHUFFLE):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.TRANSPOSE_B = gl.constexpr(TRANSPOSE_B)
        self.WITH_A_SCALE = gl.constexpr(WITH_A_SCALE)
        self.SCALE_PRESHUFFLE = gl.constexpr(SCALE_PRESHUFFLE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_A = gl.constexpr(2 if DTYPE_A == "e2m1" else 1)
        self.DIV_FACTOR_B = gl.constexpr(2 if DTYPE_B == "e2m1" else 1)
        self.NUM_LOADS_IN_BATCH = gl.constexpr(4 if WITH_A_SCALE else 3)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.SCALE_KWIDTH = gl.constexpr(4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE)
        self.PRESHUFFLE_FACTOR = gl.constexpr(128 if SCALE_PRESHUFFLE else 1)
        self.tiles_per_warp = gl.constexpr([2, 2] if SCALE_PRESHUFFLE else [1, 1])

        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // self.PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // self.PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(BLOCK_K_SCALE * self.PRESHUFFLE_FACTOR)

        WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                         instr_shape=[16, 16, 128], tiles_per_warp=self.tiles_per_warp)
        WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                                instr_shape=[16, 16,
                                                                             64], tiles_per_warp=self.tiles_per_warp)

        self.dot_layout_a = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                k_width=16))
        self.dot_layout_b = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                k_width=16))
        self.layout_a_scale = gl.constexpr(
            gl.amd.gfx1250.get_wmma_scale_layout(self.dot_layout_a, [BLOCK_M, BLOCK_K_SCALE]))
        self.layout_b_scale = gl.constexpr(
            gl.amd.gfx1250.get_wmma_scale_layout(self.dot_layout_b, [BLOCK_N, BLOCK_K_SCALE]))
        self.acc_layout = gl.constexpr(WMMA_LAYOUT)

        BLOCK_K_PACKED_A = BLOCK_K // self.DIV_FACTOR_A
        BLOCK_K_PACKED_B = BLOCK_K // self.DIV_FACTOR_B
        PAD_INTERVAL_A = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
        PAD_INTERVAL_B = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
        self.shared_layout_a = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]], [BLOCK_M, BLOCK_K_PACKED_A], [1, 0]))
        if TRANSPOSE_B:
            self.shared_layout_b = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]], [BLOCK_N, BLOCK_K_PACKED_B], [1, 0]))
        else:
            self.shared_layout_b = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K_PACKED_B, BLOCK_N], [1, 0]))

        if WITH_A_SCALE:
            self.shared_layout_a_scale = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for([[256, 16]],
                                                        [self.BLOCK_M_PRESHUFFLED, self.BLOCK_K_SCALE_PRESHUFFLED],
                                                        [1, 0]))
        else:
            self.shared_layout_a_scale = None
        self.shared_layout_b_scale = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[256, 16]],
                                                    [self.BLOCK_N_PRESHUFFLED, self.BLOCK_K_SCALE_PRESHUFFLED], [1, 0]))


@gluon.jit
def create_tensor_descriptor(cfg: MXFPGEMMConfig, a_ptr, a_offs, b_ptr, b_offs, a_scale_ptr, a_scale_offs, b_scale_ptr,
                             b_scale_offs, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_scale):
    SCALE_BLOCK: gl.constexpr = cfg.SCALE_BLOCK
    PRESHUFFLE_FACTOR: gl.constexpr = cfg.PRESHUFFLE_FACTOR
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + a_offs,  #
                                                       shape=(M, K // cfg.DIV_FACTOR_A),  #
                                                       strides=(stride_am, stride_ak),  #
                                                       block_shape=(cfg.BLOCK_M, cfg.BLOCK_K // cfg.DIV_FACTOR_A),  #
                                                       layout=cfg.shared_layout_a)

    if cfg.TRANSPOSE_B:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + b_offs,  #
                                                           shape=(N, K // cfg.DIV_FACTOR_B),  #
                                                           strides=(stride_bn, stride_bk),  #
                                                           block_shape=(cfg.BLOCK_N,
                                                                        cfg.BLOCK_K // cfg.DIV_FACTOR_B),  #
                                                           layout=cfg.shared_layout_b)
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + b_offs,  #
                                                           shape=(K // cfg.DIV_FACTOR_B, N),  #
                                                           strides=(stride_bk, stride_bn),  #
                                                           block_shape=(cfg.BLOCK_K // cfg.DIV_FACTOR_B,
                                                                        cfg.BLOCK_N),  #
                                                           layout=cfg.shared_layout_b)

    if cfg.WITH_A_SCALE:
        a_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_scale_ptr + a_scale_offs,  #
            shape=(M // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),  #
            strides=(stride_scale, 1),  #
            block_shape=(cfg.BLOCK_M // PRESHUFFLE_FACTOR, cfg.BLOCK_K_SCALE_PRESHUFFLED),  #
            layout=cfg.shared_layout_a_scale)
    else:
        a_scale_desc = None

    b_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr + b_scale_offs,  #
        shape=(N // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),  #
        strides=(stride_scale, 1),  #
        block_shape=(cfg.BLOCK_N // PRESHUFFLE_FACTOR, cfg.BLOCK_K_SCALE_PRESHUFFLED),  #
        layout=cfg.shared_layout_b_scale)

    return a_desc, b_desc, a_scale_desc, b_scale_desc


@gluon.jit
def issue_loads(cfg: MXFPGEMMConfig, load_idx, a_desc, b_desc, a_scale_desc, b_scale_desc, a_buffer, b_buffer,
                a_scale_buffer, b_scale_buffer):
    BLOCK_K_PACKED_A: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_B

    gl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K_PACKED_A], a_buffer.index(load_idx % cfg.NUM_BUFFERS))
    if cfg.TRANSPOSE_B:
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, load_idx * BLOCK_K_PACKED_B],
                                      b_buffer.index(load_idx % cfg.NUM_BUFFERS))
    else:
        gl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K_PACKED_B, 0],
                                      b_buffer.index(load_idx % cfg.NUM_BUFFERS))
    if cfg.WITH_A_SCALE:
        gl.amd.gfx1250.tdm.async_load(a_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                                      a_scale_buffer.index(load_idx % cfg.NUM_BUFFERS))
    gl.amd.gfx1250.tdm.async_load(b_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                                  b_scale_buffer.index(load_idx % cfg.NUM_BUFFERS))


@gluon.jit
def issue_local_loads(cfg: MXFPGEMMConfig, wmma_idx, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer):
    BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
    a = a_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_a)
    if cfg.TRANSPOSE_B:
        b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).permute([1, 0]).load(layout=cfg.dot_layout_b)
    else:
        b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_b)
    if cfg.WITH_A_SCALE:
        a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
    b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
    if cfg.SCALE_PRESHUFFLE:
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer_slice.reshape((
                cfg.BLOCK_M_PRESHUFFLED,  #
                BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                cfg.PRESHUFFLE_FACTOR // 4,  #
                4,  #
                cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
        b_scale_buffer_slice = b_scale_buffer_slice.reshape((
            cfg.BLOCK_N_PRESHUFFLED,  #
            BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
            cfg.PRESHUFFLE_FACTOR // 4,  #
            4,  #
            cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
    if cfg.WITH_A_SCALE:
        scale_a = a_scale_buffer_slice.load(layout=cfg.layout_a_scale)
    else:
        scale_a = None
    scale_b = b_scale_buffer_slice.load(layout=cfg.layout_b_scale)

    return a, b, scale_a, scale_b


@gluon.jit
def mxgemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk,
                                stride_bn, stride_cm, stride_cn, stride_scale, DTYPE_A: gl.constexpr,
                                DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                TRANSPOSE_B: gl.constexpr, NUM_BUFFERS: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
                                WITH_A_SCALE: gl.constexpr):
    cfg = MXFPGEMMConfig(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, NUM_BUFFERS, TRANSPOSE_B,
                         WITH_A_SCALE, SCALE_PRESHUFFLE)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_scale_offs = pid_m * cfg.BLOCK_M_PRESHUFFLED * stride_scale
    b_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_scale
    a_desc, b_desc, a_scale_desc, b_scale_desc = create_tensor_descriptor(cfg, a_ptr, a_offs, b_ptr, b_offs, a_scale,
                                                                          a_scale_offs, b_scale, b_scale_offs, M, N, K,
                                                                          stride_am, stride_ak, stride_bk, stride_bn,
                                                                          stride_scale)
    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    if cfg.WITH_A_SCALE:
        a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                   layout=a_scale_desc.layout)
    else:
        a_scale_buffer = None

    b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                               layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    # prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        issue_loads(cfg, load_idx, a_desc, b_desc, a_scale_desc, b_scale_desc, a_buffer, b_buffer, a_scale_buffer,
                    b_scale_buffer)
        load_idx += 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)
    for _ in range(0, gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        issue_loads(cfg, load_idx, a_desc, b_desc, a_scale_desc, b_scale_desc, a_buffer, b_buffer, a_scale_buffer,
                    b_scale_buffer)

        load_idx += 1

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * cfg.NUM_LOADS_IN_BATCH)

        a, b, scale_a, scale_b = issue_local_loads(cfg, wmma_idx, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer)
        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        wmma_idx += 1

    # epilogue
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * cfg.NUM_LOADS_IN_BATCH)
        a, b, scale_a, scale_b = issue_local_loads(cfg, wmma_idx, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer)
        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        wmma_idx += 1

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.acc_layout))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.store(c_ptrs, accumulator, mask=c_mask)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    if a_scale is None:
        a_scale_f32 = torch.full((M, K), 1.0, dtype=torch.float32)
    else:
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
    SCALE_PRESHUFFLE = config['SCALE_PRESHUFFLE']
    WITH_A_SCALE = config['WITH_A_SCALE']

    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K)
    b = init_data(dtype_b, K, N)
    a_scale_size = (M, (K + scale_block - 1) // scale_block)
    b_scale_size = (N, (K + scale_block - 1) // scale_block)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    if WITH_A_SCALE:
        a_scale = a_scale.data
    b_scale = b_scale.data

    if SCALE_PRESHUFFLE:
        a_scale = pack_scale(a_scale)
        b_scale = pack_scale(b_scale)

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
    if WITH_A_SCALE:
        a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk = b_d.stride(1) if TRANSPOSE_B else b_d.stride(0)
    stride_bn = b_d.stride(0) if TRANSPOSE_B else b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                      stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[dtype_a],
                                      dtype_converter[dtype_b], scale_block, blockSizeM, blockSizeN, blockSizeK,
                                      group_size_m, TRANSPOSE_B, NUM_BUFFERS, SCALE_PRESHUFFLE, WITH_A_SCALE,
                                      num_warps=numWarps, num_ctas=numCtas, waves_per_eu=numWarps // 4)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
    print('✅Pass')


def pack_scale(x):
    if x is None:
        return x
    NON_K, K_SCALE = x.shape
    preshuffle_factor = 128
    num_chunk_m = NON_K // preshuffle_factor
    SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(NON_K // preshuffle_factor, K_SCALE * preshuffle_factor)


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
    SCALE_PRESHUFFLE = config['SCALE_PRESHUFFLE']
    WITH_A_SCALE = config['WITH_A_SCALE']

    if not WITH_A_SCALE and dtype_a == "float4":
        pytest.skip("Skip fp4 x mxfp gemm to reduce test cases.")

    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K)
    b = init_data(dtype_b, K, N)
    a_scale_size = (M, (K + scale_block - 1) // scale_block)
    b_scale_size = (N, (K + scale_block - 1) // scale_block)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    if WITH_A_SCALE:
        a_scale = a_scale.data
    b_scale = b_scale.data

    if SCALE_PRESHUFFLE:
        a_scale = pack_scale(a_scale)
        b_scale = pack_scale(b_scale)

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
    if WITH_A_SCALE:
        a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    if TRANSPOSE_B:
        stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    else:
        stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    k = mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                          stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[dtype_a],
                                          dtype_converter[dtype_b], scale_block, blockSizeM, blockSizeN, blockSizeK,
                                          group_size_m, TRANSPOSE_B, NUM_BUFFERS, SCALE_PRESHUFFLE, WITH_A_SCALE,
                                          num_warps=numWarps, num_ctas=numCtas, waves_per_eu=numWarps // 4)

    if TRANSPOSE_B:
        assert 'ds_load_u8' not in k.asm['amdgcn']

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    import argparse

    supported_dtypes = ['float8_e4m3', 'float8_e5m2', 'float4']

    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=8192, help='problem M size')
    parser.add_argument('-N', type=int, default=8192, help='problem N size')
    parser.add_argument('-K', type=int, default=1024, help='problem K size')
    parser.add_argument('-BM', type=int, default=256, help='BLOCK_M')
    parser.add_argument('-BN', type=int, default=256, help='BLOCK_N')
    parser.add_argument('-BK', type=int, default=128, help='BLOCK_K')
    parser.add_argument('--num_warps', type=int, default=4, choices=[4, 8])
    parser.add_argument('--num_buffers', type=int, default=2, choices=[2, 4])
    parser.add_argument('--scale_preshuffled', action='store_true')
    parser.add_argument('--with_a_scale', action='store_true')
    parser.add_argument('--dtype_a', type=str, default='float8_e4m3', choices=supported_dtypes)
    parser.add_argument('--dtype_b', type=str, default='float8_e4m3', choices=supported_dtypes)

    args = parser.parse_args()

    config = {
        "M": args.M, "N": args.N, "K": args.K,  #
        "BLOCK_M": args.BM, "BLOCK_N": args.BN, "BLOCK_K": args.BK,  #
        "NUM_CTAS": 1, "SCALE_BLOCK": 32,  #
        "DTYPE_A": args.dtype_a, "DTYPE_B": args.dtype_b,  #
        "TRANSPOSE_B": True,  #
        "NUM_BUFFERS": args.num_buffers,  #
        "SCALE_PRESHUFFLE": args.scale_preshuffled,  #
        "WITH_A_SCALE": args.with_a_scale,  #
        "NUM_WARPS": args.num_warps
    }
    run(config)
