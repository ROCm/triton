"""
Multi-head attention kernel in Gluon
"""
# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import argparse
import re
import pytest
import torch
import math

from triton import cdiv
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

from triton.experimental.gluon.language.amd.gfx1250 import wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp

torch.random.manual_seed(0)

# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


def composition(cls):
    """ A decorator lets aggregate type to directly access attributes from its aggregate member. """

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and not hasattr(member, name):
                continue
            return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


@gluon.constexpr_function
def get_padded_shared_layout(shape, transposed=False):
    """ Get a padded shared layout without back conflict for a given tensor shape. """
    _, inner_dim = shape
    padding_interval = inner_dim
    padding_amount = 16 if transposed else 8
    return ttgl.PaddedSharedLayout.with_identity_for([[padding_interval, padding_amount]], shape, [1, 0])


@gluon.constexpr_function
def get_load_layout(shape, num_warps):
    """ Get a layout with better vectorized access for a given tensor shape. """
    _, inner_dim = shape
    assert inner_dim in [64, 128, 256, 512]
    if inner_dim == 512:
        return ttgl.BlockedLayout([1, 16], [1, 32], [num_warps, 1], [1, 0])
    if inner_dim == 256:
        return ttgl.BlockedLayout([1, 8], [1, 32], [num_warps, 1], [1, 0])
    elif inner_dim == 128:
        return ttgl.BlockedLayout([1, 4], [1, 32], [num_warps, 1], [1, 0])
    else:
        return ttgl.BlockedLayout([1, 4], [2, 16], [num_warps, 1], [1, 0])


@aggregate
class AttentionConfigBase:
    Q_TYPE: ttgl.constexpr  # the data type for Q, either 'e5m2' or 'e4m3'
    P_TYPE: ttgl.constexpr  # the data type for P; we always assume P_TYPE == Q_TYPE
    KV_TYPE: ttgl.constexpr  # the data type for K and V, either 'e5m2', 'e4m3' or 'e2m1'
    SEQLEN_Q: ttgl.constexpr
    SEQLEN_K: ttgl.constexpr
    NUM_Q_HEADS: ttgl.constexpr
    NUM_K_HEADS: ttgl.constexpr
    HEAD_SZ: ttgl.constexpr
    BLOCK_M: ttgl.constexpr
    BLOCK_N: ttgl.constexpr
    NUM_WARPS: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 NUM_WARPS, NUM_BUFFERS):
        self.Q_TYPE = ttgl.constexpr(Q_TYPE)
        self.P_TYPE = ttgl.constexpr(Q_TYPE)
        self.KV_TYPE = ttgl.constexpr(KV_TYPE)
        self.SEQLEN_Q = ttgl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = ttgl.constexpr(SEQLEN_K)
        self.NUM_Q_HEADS = ttgl.constexpr(NUM_Q_HEADS)
        self.NUM_K_HEADS = ttgl.constexpr(NUM_K_HEADS)
        self.HEAD_SZ = ttgl.constexpr(HEAD_SZ)
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.BLOCK_N = ttgl.constexpr(BLOCK_N)
        self.NUM_WARPS = ttgl.constexpr(NUM_WARPS)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)


@composition
@aggregate
class GlobalScaledAttentionConfig:
    base: AttentionConfigBase

    q_layout: ttgl.constexpr
    k_smem_layout: ttgl.constexpr
    k_layout: ttgl.constexpr
    p_layout: ttgl.constexpr
    v_smem_layout: ttgl.constexpr
    v_layout: ttgl.constexpr
    acc_layout: ttgl.constexpr

    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3']
        assert NUM_WARPS == 4 or NUM_WARPS == 8
        assert P_K_WIDTH == 16 or P_K_WIDTH == 8

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_WARPS, NUM_BUFFERS)

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[NUM_WARPS, 1], instr_shape=[16, 16, 128])
        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, 16))
        self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, 16))
        self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, P_K_WIDTH))
        self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, P_K_WIDTH))
        # Use k_width=8 for p can make it has the same layout of qk
        self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)
        self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N, HEAD_SZ]))
        self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout([HEAD_SZ, BLOCK_N]))
        self.acc_layout = ttgl.constexpr(wmma_layout)


@composition
@aggregate
class BlockScaledAttentionConfig:
    base: AttentionConfigBase

    q_layout: ttgl.constexpr
    q_scale_layout: ttgl.constexpr

    k_smem_layout: ttgl.constexpr
    k_layout: ttgl.constexpr
    k_scale_load_layout: ttgl.constexpr
    k_scale_smem_layout: ttgl.constexpr
    k_scale_layout: ttgl.constexpr

    p_layout: ttgl.constexpr
    p_scale_layout: ttgl.constexpr

    v_smem_layout: ttgl.constexpr
    v_layout: ttgl.constexpr
    v_scale_load_layout: ttgl.constexpr
    v_scale_smem_layout: ttgl.constexpr
    v_scale_layout: ttgl.constexpr

    acc_layout: ttgl.constexpr

    # Whether scales for K and V are preshuffled for better memory access.
    SCALE_PRESHUFFLED: ttgl.constexpr
    # Whether to use per-block scaling for P; if False, use an uniform scale of 1.0.
    P_SCALING: ttgl.constexpr
    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout. But note we can use k_width=8 for
    # V when it is a mxfp4, so this only applies when KV_TYPE is not 'e2m1'.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 P_SCALING, SCALE_PRESHUFFLED, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']
        assert NUM_WARPS == 4 or NUM_WARPS == 8
        assert P_K_WIDTH == 16 or (KV_TYPE != 'e2m1' and P_K_WIDTH == 8)
        KV_PACK_DIV: ttgl.constexpr = 2 if KV_TYPE == 'e2m1' else 1

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_WARPS, NUM_BUFFERS)

        self.P_SCALING = ttgl.constexpr(P_SCALING)
        self.SCALE_PRESHUFFLED = ttgl.constexpr(SCALE_PRESHUFFLED)

        tiles_per_warp: ttgl.constexpr = [2, 2] if SCALE_PRESHUFFLED else [1, 1]
        num_warps: ttgl.constexpr = NUM_WARPS

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 128],
            tiles_per_warp=tiles_per_warp)
        wmma_layout_packed: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 64],
            tiles_per_warp=tiles_per_warp)

        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=16))
        if KV_TYPE == 'e2m1':
            self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout_packed, k_width=16))
            self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=16))
            self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout_packed, k_width=16))
            self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(False)
        else:
            self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, k_width=16))
            self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=P_K_WIDTH))
            self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, k_width=P_K_WIDTH))
            self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)

        self.q_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.q_layout, [BLOCK_M, HEAD_SZ // 32]))

        self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N, HEAD_SZ // KV_PACK_DIV]))

        self.k_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.k_layout, [BLOCK_N, HEAD_SZ // 32]))
        self.k_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        self.k_scale_load_layout = ttgl.constexpr(get_load_layout([HEAD_SZ // 32, BLOCK_N], num_warps))

        self.p_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.p_layout, [BLOCK_M, BLOCK_N // 32]))

        self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout([HEAD_SZ, BLOCK_N // KV_PACK_DIV]))
        self.v_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.v_layout, [HEAD_SZ, BLOCK_N // 32]))
        self.v_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        self.v_scale_load_layout = ttgl.constexpr(get_load_layout([BLOCK_N // 32, HEAD_SZ], num_warps))

        self.acc_layout = ttgl.constexpr(wmma_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Primitives
# ===-----------------------------------------------------------------------===#


@aggregate
class MemoryUnit:
    """
    MemoryUnit abstracts the logic of transferring data to/from global memory for 2D tensor.
    It supports following methods:

    - `issue_tdm_load`: issue an async load via TDM from global memory to shared memory.
    - `issue_async_copy`: issue an async copy from global memory to shared memory.
    - `buffer_load` / `buffer_store`: transfer data between global memory and registers.

    To help use a MemoryUnit in a loop, it supports load with an `idx` argument, meaning loading the `idx`-th block
    along the `axis` dimension. This requires the one dimension of the tensor shape equals to the block size, and we
    will slide the block along the other dimension.
    """
    smem: ttgl.shared_memory_descriptor
    desc: tdm.tensor_descriptor
    ptr: ttgl.tensor
    offs: ttgl.tensor

    dtype: ttgl.constexpr
    shape: ttgl.constexpr
    block_shape: ttgl.constexpr
    strides: ttgl.constexpr
    axis: ttgl.constexpr

    layout: ttgl.constexpr
    smem_layout: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, smem, desc, ptr, offs,  #
                 dtype, shape, strides, block_shape, axis,  #
                 layout, smem_layout):
        self.smem = smem
        self.desc = desc
        self.ptr = ptr
        self.offs = offs
        self.dtype = ttgl.constexpr(dtype)
        self.shape = ttgl.constexpr(shape)
        self.block_shape = ttgl.constexpr(block_shape)
        self.strides = ttgl.constexpr(strides)
        self.axis = ttgl.constexpr(axis)
        self.layout = ttgl.constexpr(layout)
        self.smem_layout = ttgl.constexpr(smem_layout)

    @gluon.jit
    def issue_tdm_load(self, idx, buf, pred):
        axis: ttgl.constexpr = self.axis
        step: ttgl.constexpr = self.block_shape[axis]
        smem = self.smem.index(buf)
        if axis == 0:
            tdm.async_load(self.desc, [idx * step, 0], smem, pred)
        else:
            tdm.async_load(self.desc, [0, idx * step], smem, pred)

    @gluon.jit
    def issue_async_copy(self, idx, buf):
        axis: ttgl.constexpr = self.axis
        step: ttgl.constexpr = self.block_shape[axis] * self.strides[axis]
        smem = self.smem.index(buf)
        ptrs = self.ptr + self.offs + idx * step
        cp.global_to_shared(smem, ptrs)
        cp.commit_group()

    @gluon.jit
    def buffer_load(self, other=0.0, idx=0):
        axis: ttgl.constexpr = self.axis
        step: ttgl.constexpr = self.block_shape[axis] * self.strides[axis]
        offs = self.offs + idx * step
        off_axis = ttgl.arange(0, self.block_shape[axis], ttgl.SliceLayout(1 - axis, self.layout))
        mask = ttgl.expand_dims(off_axis, axis=1 - axis) < self.shape[axis]
        return buffer_load(self.ptr, offs, mask, other=other)

    @gluon.jit
    def buffer_store(self, data, idx=0):
        axis: ttgl.constexpr = self.axis
        step: ttgl.constexpr = self.block_shape[axis] * self.strides[axis]
        offs = self.offs + idx * step
        off_axis = ttgl.arange(0, self.block_shape[axis], ttgl.SliceLayout(1 - axis, self.layout))
        mask = ttgl.expand_dims(off_axis, axis=1 - axis) < self.shape[axis]
        buffer_store(data, self.ptr, offs, mask)

    @gluon.jit
    def initialize(base, shape, block_shape, layout, smem_layout=None, num_buffers=1):
        ttgl.static_assert(len(block_shape) == 2 and len(shape) == 2)

        dtype: ttgl.constexpr = base.dtype.element_ty

        shared_layout: ttgl.constexpr = (
            smem_layout  #
            if smem_layout is not None  #
            else ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        desc = tdm.make_tensor_descriptor(  #
            base=base,  #
            shape=shape,  #
            strides=[shape[1], 1],  #
            block_shape=block_shape,  #
            layout=shared_layout)
        smem = ttgl.allocate_shared_memory(  #
            dtype,  #
            [num_buffers] + block_shape,  #
            shared_layout)

        if shape[0] != block_shape[0]:
            ttgl.static_assert(shape[1] == block_shape[1])
            axis: ttgl.constexpr = 0
        else:
            axis: ttgl.constexpr = 1

        offs_m = ttgl.arange(0, block_shape[0], ttgl.SliceLayout(1, layout))
        offs_n = ttgl.arange(0, block_shape[1], ttgl.SliceLayout(0, layout))
        offs = offs_m[:, None] * shape[1] + offs_n[None, :]

        return MemoryUnit(smem, desc, base, offs,  #
                          dtype, shape, [shape[1], 1], block_shape, axis,  #
                          layout, smem_layout)


@aggregate
class GlobalScaledAttentionProgram:
    cfg: GlobalScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor
    k_mem: MemoryUnit
    k_scale: ttgl.tensor
    v_mem: MemoryUnit
    v_scale: ttgl.tensor
    o_mem: MemoryUnit
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_mem, k_scale,  #
                 v_mem, v_scale,  #
                 o_mem,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_mem = k_mem
        self.k_scale = k_scale
        self.v_mem = v_mem
        self.v_scale = v_scale
        self.o_mem = o_mem
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg, q_ptr, q_scale, k_ptr, k_scale, v_ptr, v_scale, o_ptr, sm_scale):
        ttgl.static_assert(isinstance(cfg, GlobalScaledAttentionConfig))
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        off_h = ttgl.program_id(0)  # NUM_Q_HEADS
        off_m = ttgl.program_id(1)  # NUM_BLOCKS
        off_z = ttgl.program_id(2)  # BATCH

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        group_sz: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // group_sz

        q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) +\
                BLOCK_M * off_m * HEAD_SZ
        q_mem = MemoryUnit.initialize(  #
            base=q_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.q_layout)

        k_off = SEQLEN_K * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        k_mem = MemoryUnit.initialize(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.k_layout,  #
            smem_layout=cfg.k_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        v_mem = MemoryUnit.initialize(  #
            base=v_ptr + k_off,  #
            shape=[HEAD_SZ, SEQLEN_K],  #
            block_shape=[HEAD_SZ, BLOCK_N],  #
            layout=cfg.v_layout,  #
            smem_layout=cfg.v_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        o_mem = MemoryUnit.initialize(  #
            base=o_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.acc_layout)

        q = q_mem.buffer_load()

        return GlobalScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_mem, k_scale,  #
            v_mem, v_scale,  #
            o_mem,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, i, buf, pred=True):
        self.k_mem.issue_tdm_load(i, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, i, buf, pred=True):
        self.v_mem.issue_tdm_load(i, buf, pred)

    @gluon.jit
    def shared_load_k(self, buf, wait_count):
        cfg = self.cfg

        tdm.async_wait(wait_count)
        k_buffer = self.k_mem.smem.index(buf).permute((1, 0))
        k = k_buffer.load(cfg.k_layout)
        k_scale = self.k_scale
        return k, k_scale

    @gluon.jit
    def shared_load_v(self, buf, wait_count):
        cfg = self.cfg

        tdm.async_wait(wait_count)
        v_buffer = self.v_mem.smem.index(buf).permute((1, 0))
        v = v_buffer.load(cfg.v_layout)
        v_scale = self.v_scale
        return v, v_scale

    @gluon.jit
    def compute_qk(self, k, k_scale):
        cfg = self.cfg
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, zero)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def softmax0(self, qk, m_i):
        sm_scale = self.sm_scale

        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))

        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)

        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)

        return p, alpha, m_ij

    @gluon.jit
    def softmax1(self, p, alpha, acc, l_i):
        cfg = self.cfg

        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij

        p = p.to(ttgl.float8e4nv if cfg.P_TYPE == 'e4m3' else ttgl.float8e5)
        p = ttgl.convert_layout(p, cfg.p_layout, cfg.CONVERT_LAYOUT_TRIVIAL)
        p_scale = 0x7F

        return p, p_scale, acc, l_i

    @gluon.jit
    def store_output(self, acc):
        o = acc.to(self.o_mem.dtype)
        self.o_mem.buffer_store(o)


@aggregate
class BlockScaledAttentionProgram:
    cfg: BlockScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor
    k_mem: MemoryUnit
    k_scale_mem: MemoryUnit
    v_mem: MemoryUnit
    v_scale_mem: MemoryUnit
    o_mem: MemoryUnit
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_mem, k_scale_mem,  #
                 v_mem, v_scale_mem,  #
                 o_mem,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_mem = k_mem
        self.k_scale_mem = k_scale_mem
        self.v_mem = v_mem
        self.v_scale_mem = v_scale_mem
        self.o_mem = o_mem
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg,  #
                   q_ptr, q_scale_ptr,  #
                   k_ptr, k_scale_ptr,  #
                   v_ptr, v_scale_ptr,  #
                   o_ptr,  #
                   sm_scale):
        ttgl.static_assert(isinstance(cfg, BlockScaledAttentionConfig))
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        KV_PACK_DIV: ttgl.constexpr = 2 if cfg.KV_TYPE == 'e2m1' else 1
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        off_h = ttgl.program_id(0)  # NUM_Q_HEADS
        off_m = ttgl.program_id(1)  # NUM_BLOCKS
        off_z = ttgl.program_id(2)  # BATCH

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        group_sz: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // group_sz

        q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                BLOCK_M * off_m * HEAD_SZ
        q_mem = MemoryUnit.initialize(  #
            base=q_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.q_layout)

        q_scale_off = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_Q_HEADS * off_z + off_h) + \
                      BLOCK_M * off_m * (HEAD_SZ // 32)
        q_scale_mem = MemoryUnit.initialize(  #
            base=q_scale_ptr + q_scale_off,  #
            shape=[SEQLEN_Q, HEAD_SZ // 32],  #
            block_shape=[BLOCK_M, HEAD_SZ // 32],  #
            layout=cfg.q_scale_layout)

        k_off = SEQLEN_K * (HEAD_SZ // KV_PACK_DIV) * (NUM_K_HEADS * off_z + off_hk)
        k_mem = MemoryUnit.initialize(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ // KV_PACK_DIV],  #
            block_shape=[BLOCK_N, HEAD_SZ // KV_PACK_DIV],  #
            layout=cfg.k_layout,  #
            smem_layout=cfg.k_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        if cfg.SCALE_PRESHUFFLED:
            K_SCALE_DIV: ttgl.constexpr = 128
            k_scale_off = (SEQLEN_K // K_SCALE_DIV) * (HEAD_SZ // 32 * K_SCALE_DIV) * (NUM_K_HEADS * off_z + off_hk)
            k_scale_mem = MemoryUnit.initialize(  #
                base=k_scale_ptr + k_scale_off,  #
                shape=[SEQLEN_K // K_SCALE_DIV, HEAD_SZ // 32 * K_SCALE_DIV],  #
                block_shape=[BLOCK_N // K_SCALE_DIV, HEAD_SZ // 32 * K_SCALE_DIV],  #
                layout=cfg.k_scale_layout,  #
                smem_layout=cfg.k_scale_smem_layout,  #
                num_buffers=NUM_BUFFERS)
        else:
            k_scale_off = SEQLEN_K * (HEAD_SZ // 32) * (NUM_K_HEADS * off_z + off_hk)
            k_scale_mem = MemoryUnit.initialize(  #
                base=k_scale_ptr + k_scale_off,  #
                shape=[HEAD_SZ // 32, SEQLEN_K],  #
                block_shape=[HEAD_SZ // 32, BLOCK_N],  #
                layout=cfg.k_scale_load_layout,  #
                smem_layout=cfg.k_scale_smem_layout,  #
                num_buffers=NUM_BUFFERS)

        v_off = (SEQLEN_K // KV_PACK_DIV) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        v_mem = MemoryUnit.initialize(  #
            base=v_ptr + v_off,  #
            shape=[HEAD_SZ, SEQLEN_K // KV_PACK_DIV],  #
            block_shape=[HEAD_SZ, BLOCK_N // KV_PACK_DIV],  #
            layout=cfg.v_layout,  #
            smem_layout=cfg.v_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        if cfg.SCALE_PRESHUFFLED:
            V_SCALE_DIV: ttgl.constexpr = 128 if HEAD_SZ == 128 else 64
            v_scale_off = (SEQLEN_K // 32 * V_SCALE_DIV) * (HEAD_SZ // V_SCALE_DIV) * (NUM_K_HEADS * off_z + off_hk)
            v_scale_mem = MemoryUnit.initialize(  #
                base=v_scale_ptr + v_scale_off,  #
                shape=[HEAD_SZ // V_SCALE_DIV, SEQLEN_K // 32 * V_SCALE_DIV],  #
                block_shape=[HEAD_SZ // V_SCALE_DIV, BLOCK_N // 32 * V_SCALE_DIV],  #
                layout=cfg.v_scale_layout,  #
                smem_layout=cfg.v_scale_smem_layout,  #
                num_buffers=NUM_BUFFERS)
        else:
            v_scale_off = (SEQLEN_K // 32) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
            v_scale_mem = MemoryUnit.initialize(  #
                base=v_scale_ptr + v_scale_off,  #
                shape=[SEQLEN_K // 32, HEAD_SZ],  #
                block_shape=[BLOCK_N // 32, HEAD_SZ],  #
                layout=cfg.v_scale_load_layout,  #
                smem_layout=cfg.v_scale_smem_layout,  #
                num_buffers=NUM_BUFFERS)

        o_unit = MemoryUnit.initialize(  #
            base=o_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.acc_layout,  #
            smem_layout=None)

        q = q_mem.buffer_load()
        q_scale = q_scale_mem.buffer_load(other=0x7F)

        return BlockScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_mem, k_scale_mem,  #
            v_mem, v_scale_mem,  #
            o_unit,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, i, buf, pred=True):
        cfg = self.cfg

        self.k_mem.issue_tdm_load(i, buf, pred)
        if cfg.SCALE_PRESHUFFLED:
            self.k_scale_mem.issue_tdm_load(i, buf, pred)
        else:
            # TODO: We use TDM to avoid register spills for preshuffling, but TDM increases register usage for
            # non-preshuffling case, so that we fall back to async copy here. Because async copy does not have a pred
            # field, we have to branch here, and also commit a group to keep the wait counts consistent. Switch to use
            # the TDM once the issue is resolved.
            if pred:
                self.k_scale_mem.issue_async_copy(i, buf)
            else:
                cp.commit_group()

    @gluon.jit
    def issue_global_load_v(self, i, buf, pred=True):
        cfg = self.cfg

        self.v_mem.issue_tdm_load(i, buf, pred)
        if cfg.SCALE_PRESHUFFLED:
            self.v_scale_mem.issue_tdm_load(i, buf, pred)
        else:
            # TODO: We use TDM to avoid register spills for preshuffling, but TDM increases register usage for
            # non-preshuffling case, so that we fall back to async copy here. Because async copy does not have a pred
            # field, we have to branch here, and also commit a group to keep the wait counts consistent. Switch to use
            # the TDM once the issue is resolved.
            if pred:
                self.v_scale_mem.issue_async_copy(i, buf)
            else:
                cp.commit_group()

    @gluon.jit
    def shared_load_k(self, buf, wait_count):
        cfg = self.cfg

        self._async_wait(wait_count)

        k_buffer = self.k_mem.smem.index(buf).permute((1, 0))
        k = k_buffer.load(cfg.k_layout)

        k_scale_buffer = self.k_scale_mem.smem.index(buf)
        if cfg.SCALE_PRESHUFFLED:
            K_SCALE_DIV: ttgl.constexpr = 128
            k_scale_buffer = self._unshuffle_scale(k_scale_buffer, cfg.BLOCK_N, cfg.HEAD_SZ // 32, K_SCALE_DIV)
        else:
            k_scale_buffer = k_scale_buffer.permute((1, 0))
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)

        return k, k_scale

    @gluon.jit
    def shared_load_v(self, buf, wait_count):
        cfg = self.cfg

        self._async_wait(wait_count)

        v_buffer = self.v_mem.smem.index(buf).permute((1, 0))
        v = v_buffer.load(cfg.v_layout)

        v_scale_buffer = self.v_scale_mem.smem.index(buf)
        if cfg.SCALE_PRESHUFFLED:
            V_SCALE_DIV: ttgl.constexpr = 128 if cfg.HEAD_SZ == 128 else 64
            v_scale_buffer = self._unshuffle_scale(v_scale_buffer, cfg.HEAD_SZ, cfg.BLOCK_N // 32, V_SCALE_DIV)
        else:
            v_scale_buffer = v_scale_buffer.permute((1, 0))
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)

        return v, v_scale

    @gluon.jit
    def compute_qk(self, k, k_scale):
        cfg = self.cfg
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, zero)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def softmax0(self, qk, m_i):
        sm_scale = self.sm_scale

        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))

        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)

        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)

        return p, alpha, m_ij

    @gluon.jit
    def softmax1(self, p, alpha, acc, l_i):
        cfg = self.cfg

        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij

        if cfg.P_SCALING:
            p, p_scale = self._downcast_fp32_to_mxfp8(p, cfg.P_TYPE, [cfg.BLOCK_M, cfg.BLOCK_N])
            p_scale = ttgl.convert_layout(p_scale, cfg.p_scale_layout)
        else:
            p = self._downcast_fp32_to_fp8(p, cfg.P_TYPE)
            p_scale = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 32], 0x7F, ttgl.uint8, cfg.p_scale_layout)
        p = ttgl.convert_layout(p, cfg.p_layout, cfg.CONVERT_LAYOUT_TRIVIAL)

        return p, p_scale, acc, l_i

    @gluon.jit
    def store_output(self, acc):
        o = acc.to(self.o_mem.dtype)
        self.o_mem.buffer_store(o)

    @gluon.jit
    def _async_wait(self, count):
        if self.cfg.SCALE_PRESHUFFLED:
            tdm.async_wait(count * 2)
        else:
            tdm.async_wait(count)
            cp.wait_group(count)

    @gluon.jit
    def _downcast_fp32_to_mxfp8(self, x, x_format: ttgl.constexpr, shape: ttgl.constexpr):
        block_size: ttgl.constexpr = 32
        outer_dim: ttgl.constexpr = shape[0]
        inner_dim: ttgl.constexpr = shape[1]

        ttgl.static_assert(x_format == 'e4m3' or x_format == 'e5m2')
        dtype: ttgl.constexpr = ttgl.float8e4nv if x_format == 'e4m3' else ttgl.float8e5
        fp8_max: ttgl.constexpr = 57344.0 if dtype == 'e5m2' else 448.0

        ttgl.static_assert(x.dtype == ttgl.float32)
        x = ttgl.reshape(x, [outer_dim, inner_dim // block_size, block_size])
        x_abs = ttgl.abs(x)
        x_max = ttgl.max(x_abs, axis=2)

        dequant_scale = x_max / fp8_max
        dequant_scale = (dequant_scale.to(ttgl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000

        dequant_scale_fp32 = dequant_scale.to(ttgl.float32, bitcast=True)
        quant_scale = ttgl.where(dequant_scale_fp32 == 0.0, 0, 1.0 / dequant_scale_fp32)

        x = x * quant_scale[:, :, None]
        x = ttgl.reshape(x, [outer_dim, inner_dim])
        x = x.to(dtype)

        dequant_scale = (dequant_scale >> 23).to(ttgl.uint8)
        return x, dequant_scale

    @gluon.jit
    def _downcast_fp32_to_fp8(self, x, x_format: ttgl.constexpr):
        if x_format == 'e4m3':
            return x.to(ttgl.float8e4nv)
        else:
            assert x_format == 'e5m2'
            return x.to(ttgl.float8e5)

    @gluon.jit
    def _unshuffle_scale(self, buffer, non_k_dim, k_dim, non_k_div):
        block_non_k: ttgl.constexpr = non_k_dim // non_k_div
        kwidth: ttgl.constexpr = 4 if k_dim >= 4 else k_dim
        return (buffer  #
                .reshape((block_non_k, k_dim // kwidth, non_k_div // 4, 4, kwidth))  #
                .permute((0, 3, 2, 1, 4))  #
                .reshape((non_k_dim, k_dim)))


@gluon.jit
def get_program(q_ptr, k_ptr, v_ptr,  #
                q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                o_ptr,  #
                sm_scale,  #
                Q_TYPE: ttgl.constexpr,  #
                KV_TYPE: ttgl.constexpr,  #
                SEQLEN_Q: ttgl.constexpr,  #
                SEQLEN_K: ttgl.constexpr,  #
                NUM_Q_HEADS: ttgl.constexpr,  #
                NUM_K_HEADS: ttgl.constexpr,  #
                HEAD_SZ: ttgl.constexpr,  #
                BLOCK_M: ttgl.constexpr,  #
                BLOCK_N: ttgl.constexpr,  #
                BLOCK_SCALING: ttgl.constexpr,  #
                P_SCALING: ttgl.constexpr,  #
                SCALE_PRESHUFFLED: ttgl.constexpr,  #
                P_K_WIDTH: ttgl.constexpr,  #
                NUM_BUFFERS: ttgl.constexpr):
    NUM_WARPS: ttgl.constexpr = ttgl.num_warps()
    if BLOCK_SCALING:
        cfg = BlockScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, P_SCALING,
            SCALE_PRESHUFFLED, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS)
        pgm = BlockScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    else:
        cfg = GlobalScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, P_K_WIDTH,
            NUM_BUFFERS, NUM_WARPS)
        pgm = GlobalScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    return pgm


# ===-----------------------------------------------------------------------===#
# Gluon Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr,  #
                    q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                    o_ptr,  #
                    sm_scale,  #
                    Q_TYPE: ttgl.constexpr,  #
                    KV_TYPE: ttgl.constexpr,  #
                    SEQLEN_Q: ttgl.constexpr,  #
                    SEQLEN_K: ttgl.constexpr,  #
                    NUM_Q_HEADS: ttgl.constexpr,  #
                    NUM_K_HEADS: ttgl.constexpr,  #
                    HEAD_SZ: ttgl.constexpr,  #
                    BLOCK_M: ttgl.constexpr,  #
                    BLOCK_N: ttgl.constexpr,  #
                    BLOCK_SCALING: ttgl.constexpr,  #
                    P_SCALING: ttgl.constexpr,  #
                    SCALE_PRESHUFFLED: ttgl.constexpr,  #
                    P_K_WIDTH: ttgl.constexpr):
    # init program
    pgm = get_program(  #
        q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, o_ptr, sm_scale,  #
        Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, BLOCK_SCALING,
        P_SCALING, SCALE_PRESHUFFLED, P_K_WIDTH, NUM_BUFFERS=1)
    cfg = pgm.cfg

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)
    for i in range(0, end):
        pgm.issue_global_load_k(i, buf=0)
        k, k_scale = pgm.shared_load_k(buf=0, wait_count=0)
        p = pgm.compute_qk(k, k_scale)
        p, alpha, m_i = pgm.softmax0(p, m_i)
        p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
        pgm.issue_global_load_v(i, buf=0)
        v, v_scale = pgm.shared_load_v(buf=0, wait_count=0)
        acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)

    acc = acc / l_i[:, None]
    pgm.store_output(acc)


@gluon.jit
def attn_fwd_pipelined_kernel(q_ptr, k_ptr, v_ptr,  #
                              q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                              o_ptr,  #
                              sm_scale,  #
                              Q_TYPE: ttgl.constexpr,  #
                              KV_TYPE: ttgl.constexpr,  #
                              SEQLEN_Q: ttgl.constexpr,  #
                              SEQLEN_K: ttgl.constexpr,  #
                              NUM_Q_HEADS: ttgl.constexpr,  #
                              NUM_K_HEADS: ttgl.constexpr,  #
                              HEAD_SZ: ttgl.constexpr,  #
                              BLOCK_M: ttgl.constexpr,  #
                              BLOCK_N: ttgl.constexpr,  #
                              BLOCK_SCALING: ttgl.constexpr,  #
                              P_SCALING: ttgl.constexpr,  #
                              SCALE_PRESHUFFLED: ttgl.constexpr,  #
                              P_K_WIDTH: ttgl.constexpr):
    # init program
    pgm = get_program(  #
        q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, o_ptr, sm_scale,  #
        Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, BLOCK_SCALING,
        P_SCALING, SCALE_PRESHUFFLED, P_K_WIDTH, NUM_BUFFERS=2)
    cfg = pgm.cfg

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)

    # pipeline prologue, loop -3
    pgm.issue_global_load_k(0, buf=0)

    # pipeline prologue, loop -2
    pgm.issue_global_load_k(1, buf=1)

    k, k_scale = pgm.shared_load_k(buf=0, wait_count=1)

    pgm.issue_global_load_v(0, buf=0)

    # pipeline prologue, loop -1
    qk = pgm.compute_qk(k, k_scale)

    pgm.issue_global_load_k(2, buf=0)

    p, alpha, m_i = pgm.softmax0(qk, m_i)
    k, k_scale = pgm.shared_load_k(buf=1, wait_count=2)

    pgm.issue_global_load_v(1, buf=1)

    # main loop, loop 0 to end-3
    # TODO: Ideally we should unroll the loop by 2 to remove the buffer index
    # update, but our current codegen in llvm does not perform well. Re-enable
    # unroll when fixed.
    for i in range(0, end - 2):
        buf = i % 2

        # loop i
        qk = pgm.compute_qk(k, k_scale)
        p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
        v, v_scale = pgm.shared_load_v(buf, wait_count=2)

        pgm.issue_global_load_k(i + 3, 1 - buf, pred=i != end - 3)

        acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)
        p, alpha, m_i = pgm.softmax0(qk, m_i)
        k, k_scale = pgm.shared_load_k(buf, wait_count=2)

        pgm.issue_global_load_v(i + 2, buf)

    # pipeline epilogue, loop end-2
    qk = pgm.compute_qk(k, k_scale)
    p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
    v, v_scale = pgm.shared_load_v(buf=0, wait_count=2)

    acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)
    p, alpha, m_i = pgm.softmax0(qk, m_i)

    # pipeline epilogue, loop end-1
    # NOTE: in the last iteration, load k is disabled with predicate but still
    # exists in the schedule, so we also take it into wait count.
    p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
    v, v_scale = pgm.shared_load_v(buf=1, wait_count=0)

    acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)

    # write output
    acc = acc / l_i[:, None]
    pgm.store_output(acc)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
             q_scale: torch.Tensor | int, k_scale: torch.Tensor | int, v_scale: torch.Tensor | int,  #
             q_type: str, kv_type: str, block_m: int, block_n: int,  #
             block_scaling: bool, pipelined: bool, p_scaling: bool, scale_preshuffled: bool, p_k_width: int):
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)
    assert head_sz in {64, 128}
    assert block_n == 128

    # q: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
    # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / KV_PACK_DIV]
    # v: [BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K / KV_PACK_DIV]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    if block_scaling:
        # q_scale: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ / 32]
        q_scale = q_scale.permute(0, 2, 1, 3).contiguous()
        if scale_preshuffled:
            # In scaled wmma instruction, scales takes following shapes in global memory:
            # - a_scale: [M, K // 32]
            # - b_scale: [N, K // 32]
            #
            # To have vectorized memory access, it's better to store scales in a packed block scale layout. In this
            # layout, scales are stored in the shape:
            # - a_scale: [M // 32 // 4, K // 32 // 4, 32, 4, 4]
            # - b_scale: [N // 32 // 4, K // 32 // 4, 32, 4, 4]
            #
            # In this way, we can load scales from global memory in a more vectorized way. Then inside the kernel, we
            # permute and reshape scales to canonical shapes required by scaled wmma.
            def _preshuffle_scale(x: torch.Tensor, preshuffle_factor: int):
                b, h, non_k, k = x.shape
                num_chunk_m = non_k // preshuffle_factor
                scale_kwidth = 4 if k >= 4 else k
                num_chunk_k = k // scale_kwidth

                x = x.view(b, h, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
                x = x.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
                return x.view(b, h, non_k // preshuffle_factor, k * preshuffle_factor)

            # k_scale:              [BATCH, NUM_K_HEADS, SEQLEN_K / 128, HEAD_SZ * 4]
            # v_scale(head_sz=128): [BATCH, NUM_K_HEADS, HEAD_SZ / 128, SEQLEN_K * 4]
            # v_scale(head_sz=64):  [BATCH, NUM_K_HEADS, HEAD_SZ / 64, SEQLEN_K * 2]
            k_scale = _preshuffle_scale(k_scale.permute(0, 2, 1, 3), 128)
            v_scale = _preshuffle_scale(v_scale.permute(0, 2, 3, 1), 128 if head_sz == 128 else 64)
        else:
            # In the case of non-preshuffled scales, we will transpose the last 2 dims for better memory access pattern:
            # - a_scale: [K // 32, M]
            # - b_scale: [K // 32, N]

            # k_scale: [BATCH, NUM_K_HEADS, HEAD_SZ / 32, SEQLEN_K]
            # v_scale: [BATCH, NUM_K_HEADS, SEQLEN_K / 32, HEAD_SZ]
            k_scale = k_scale.permute(0, 2, 3, 1).contiguous()
            v_scale = v_scale.permute(0, 2, 1, 3).contiguous()
    else:
        assert scale_preshuffled is False
    # o: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
    o = torch.zeros_like(q, dtype=torch.float32)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    if block_scaling:
        q_scale = q_scale.cuda()
        k_scale = k_scale.cuda()
        v_scale = v_scale.cuda()
    o = o.cuda()

    # Use (NUM_Q_HEADS, NUM_BLOCKS, BATCH) for better xcd locality
    grid = (num_q_heads, cdiv(seqlen_q, block_m), batch)
    args = [
        q, k, v, q_scale, k_scale, v_scale, o, sm_scale,  #
        q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,  #
        block_scaling, p_scaling, scale_preshuffled, p_k_width
    ]
    kwargs = {
        "num_warps": 4,
        "waves_per_eu": 1,
    }
    if pipelined:
        assert cdiv(seqlen_k, block_n) > 4
        assert cdiv(seqlen_k, block_n) % 2 == 0
        kernel = attn_fwd_pipelined_kernel[grid](*args, **kwargs)
    else:
        kernel = attn_fwd_kernel[grid](*args, **kwargs)

    return o.cpu().permute(0, 2, 1, 3), kernel


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _attn_fwd_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
                  q_scale: torch.Tensor | float, k_scale: torch.Tensor | float,
                  v_scale: torch.Tensor | float) -> torch.Tensor:

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    g = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output


def _create_operand(dtype: str, b: int, s: int, h: int, d: int, pack_dim: int = -1):
    size = (b, s, h, d)
    if dtype == 'e4m3':
        sig = torch.randint(0, 2, size, dtype=torch.uint8)
        exp = torch.randint(0, 2**4, size, dtype=torch.uint8)
        man = torch.randint(0, 2**3, size, dtype=torch.uint8)
        v = ((sig << 7) | (exp << 3) | man).type(torch.uint8)
        v[(exp << 3) | man == 0x7F] = 0x00  # avoid NaN
        v = v.view(torch.float8_e4m3fn)
        v_ref = v.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == 'e5m2':
        sig = torch.randint(0, 2, size, dtype=torch.uint8)
        exp = torch.randint(0, 2**5, size, dtype=torch.uint8)
        man = torch.randint(0, 2**2, size, dtype=torch.uint8)
        v = ((sig << 7) | (exp << 2) | man).type(torch.uint8)
        v[(exp << 2) | man >= 0x7C] = 0x00  # avoid NaN and Inf
        v = v.view(torch.float8_e5m2)
        v_ref = v.view(torch.float8_e5m2).to(torch.float32)
    else:
        assert dtype == 'e2m1'
        assert pack_dim >= 0
        v_mxfp4 = MXFP4Tensor(size=size).random()
        v = v_mxfp4.to_packed_tensor(pack_dim)
        v_ref = v_mxfp4.to(torch.float32)
    return v, v_ref


def _create_block_scale(dtype: str, b: int, s: int, h: int, d: int, scale_dim: int):
    # Limit scale to an empirical range for accuracy
    if dtype == 'e4m3':
        low, high = 1 / 16, 2
    elif dtype == 'e5m2':
        low, high = 1 / 16, 2
    else:
        assert dtype == 'e2m1'
        low, high = 1 / 4, 16
    size = [b, s, h, d]
    size[scale_dim] //= 32
    scale = MXScaleTensor(size=tuple(size)).random(low, high)
    scale_ref = scale.to(torch.float32).repeat_interleave(32, dim=scale_dim)
    return scale.data, scale_ref


def _create_global_scale(dtype: str):
    assert dtype in ['e4m3', 'e5m2']
    low, high = (0x7F - 1), 0x7F + 1
    scale = torch.randint(low, high + 1, (), dtype=torch.uint8).item()
    scale_ref = 2**(scale - 0x7F)
    return scale, scale_ref


def static_profile(kernel):
    amdgcn = kernel.asm['amdgcn']

    sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
    sgpr_spill_count = int(re.search(r'\.sgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
    vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    scratch_size = int(re.search(r';\s+ScratchSize:\s+(\d+)', amdgcn).group(1))
    code_len_in_byte = int(re.search(r';\s+codeLenInByte\s+=\s+(\d+)', amdgcn).group(1))
    occupancy = int(re.search(r';\s+Occupancy:\s+(\d+)', amdgcn).group(1))

    print(f"- sgpr_count: {sgpr_count}\n"
          f"- sgpr_spill_count: {sgpr_spill_count}\n"
          f"- vgpr_count: {vgpr_count}\n"
          f"- vgpr_spill_count: {vgpr_spill_count}\n"
          f"- scratch_size: {scratch_size}\n"
          f"- code_len_in_byte: {code_len_in_byte}\n"
          f"- occupancy: {occupancy}\n")


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,pipelined,scale_preshuffled,p_k_width",
    [(*test, *config)  #
     for test in [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
                  for q_type, kv_type in [("e4m3", "e4m3"), ("e4m3", "e2m1")]
                  for batch in [1]
                  for seqlen_q in [1, 1024]  # Prefill, Decode
                  for seqlen_k in [1024]
                  for num_q_heads, num_k_heads in [(1, 1), (4, 1), (4, 2)]  # MHA, MQA, GQA
                  for head_sz in [64, 128]]
     for config in [[128, 128, False, False, 16],  # baseline
                    [128, 128, True, False, 16],  # enable pipeline
                    [128, 128, True, True, 16],  # enable pipeline + scale preshuffle
                    [128, 128, True, True, 8],  # enable pipeline + scale preshuffle + layout optimization
                    ]
     # only run optimized config for decode mha with head_sz=128
     if not (config != [128, 128, False, False, 16] and test[3:] != [1024, 1024, 1, 1, 128])])
def test_block_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                               block_m, block_n, pipelined, scale_preshuffled, p_k_width):
    if kv_type == 'e2m1' and p_k_width == 8:
        pytest.skip("e2m1 can not use k_width=8 for p")

    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = _create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)

    o, kernel = attn_fwd(q, k, v,  #
                         q_scale, k_scale, v_scale,  #
                         q_type, kv_type, block_m, block_n,  #
                         True, pipelined, False, scale_preshuffled, p_k_width)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    amdgcn = kernel.asm['amdgcn']

    # check use correct wmma scaled instruction
    wmma_instrs = re.search(r'v_wmma_[^ ]+', amdgcn)
    for instr in wmma_instrs.groups():
        assert instr == 'v_wmma_scale_f32_16x16x128_f8f6f4'

    # check there is no convert layout for P via shared memory
    ds_store_instrs = re.findall(r'ds_store_[^ ]+', amdgcn)
    assert len(ds_store_instrs) == 0

    # TODO: Reenable this for scale preshuffling after tweaking layouts
    if not scale_preshuffled:
        # check use non-transposed load of k, v and transposed load of k_scale, v_scale from shared memory
        ds_load_instrs = re.findall(r'ds_load_[^ ]+', amdgcn)
        ds_load_instrs = set(ds_load_instrs)

        ds_load_required = {'ds_load_tr8_b64', 'ds_load_b128'}
        if p_k_width == 16:
            assert ds_load_instrs == ds_load_required
        else:
            assert ds_load_instrs == ds_load_required.union({'ds_load_2addr_b64'}) or \
                   ds_load_instrs == ds_load_required.union({'ds_load_2addr_b64', 'ds_load_b64'})

    # check async global load is vectorized with scale preshuffling
    if scale_preshuffled:
        async_load_instrs = re.findall(r'global_load_async_to_lds_[^ ]+', amdgcn)
        if head_sz == 128:
            for instr in async_load_instrs:
                assert instr == "global_load_async_to_lds_b128"
        elif head_sz == 64:
            for instr in async_load_instrs:
                assert instr == "global_load_async_to_lds_b64"

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.1, rtol=0.1)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,pipelined,p_k_width",
    [(*test, *config)  #
     for test in [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
                  for q_type, kv_type in [("e4m3", "e4m3")]
                  for batch in [1]
                  for seqlen_q in [1, 1024]  # Prefill, Decode
                  for seqlen_k in [1024]
                  for num_q_heads, num_k_heads in [(1, 1), (4, 1), (4, 2)]  # MHA, MQA, GQA
                  for head_sz in [64, 128]]
     for config in [[128, 128, False, 16],  # baseline
                    [128, 128, True, 16],  # enable pipeline
                    [128, 128, True, 8],  # enable pipeline + layout optimization
                    ]
     # only run optimized config for decode mha with head_sz=128
     if not (config != [128, 128, False, 16] and test[3:] != [1024, 1024, 1, 1, 128])])
def test_global_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                                block_m, block_n, pipelined, p_k_width):
    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    q_scale, q_scale_ref = _create_global_scale(q_type)
    k_scale, k_scale_ref = _create_global_scale(kv_type)
    v_scale, v_scale_ref = _create_global_scale(kv_type)

    o, kernel = attn_fwd(q, k, v,  #
                         q_scale, k_scale, v_scale,  #
                         q_type, kv_type, block_m, block_n,  #
                         False, pipelined, False, False, p_k_width)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    amdgcn = kernel.asm['amdgcn']

    # check use correct wmma scaled instruction
    wmma_instrs = re.findall(r'v_wmma_[^ ]+', amdgcn)
    assert len(wmma_instrs) > 0 and all(instr == 'v_wmma_scale_f32_16x16x128_f8f6f4' for instr in wmma_instrs)

    # check there is no convert layout for p via shared memory
    ds_store_instrs = re.findall(r'ds_store_[^ ]+', amdgcn)
    assert len(ds_store_instrs) == 0

    # check always use non-transposed load of k and v from shared memory
    ds_load_instrs = re.findall(r'ds_load_[^ ]+', amdgcn)
    ds_load_instrs = set(ds_load_instrs)
    ds_load_required = {'ds_load_b128'}
    if p_k_width == 16:
        assert ds_load_instrs == ds_load_required
    else:
        assert ds_load_instrs == ds_load_required.union({'ds_load_2addr_b64'}) or \
               ds_load_instrs == ds_load_required.union({'ds_load_2addr_b64', 'ds_load_b64'})

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.25, rtol=0.25)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"


if __name__ == "__main__":

    def launch(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,
               scale_type, pipelined, disable_p_scaling, scale_preshuffled, p_k_width):
        if kv_type == 'e2m1' and p_k_width == 8:
            raise RuntimeError("e2m1 can not use k_width=8 for p")

        q, _ = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
        k, _ = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
        v, _ = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
        if scale_type == 'block':
            q_scale, _ = _create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
            k_scale, _ = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
            v_scale, _ = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)
        else:
            assert scale_type == 'global'
            q_scale, _ = _create_global_scale(q_type)
            k_scale, _ = _create_global_scale(kv_type)
            v_scale, _ = _create_global_scale(kv_type)

        _, kernel = attn_fwd(q, k, v,  #
                             q_scale, k_scale, v_scale,  #
                             q_type, kv_type, block_m, block_n,  #
                             scale_type == 'block', pipelined, not disable_p_scaling, scale_preshuffled, p_k_width)
        static_profile(kernel)

    parser = argparse.ArgumentParser()
    parser.add_argument("--q_type", type=str, choices=['e4m3', 'e5m2'], required=True)
    parser.add_argument("--kv_type", type=str, choices=['e4m3', 'e5m2', 'e2m1'], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seqlen_q", type=int, required=True)
    parser.add_argument("--seqlen_k", type=int, required=True)
    parser.add_argument("--num_q_heads", type=int, required=True)
    parser.add_argument("--num_k_heads", type=int, required=True)
    parser.add_argument("--head_sz", type=int, required=True)
    parser.add_argument("--block_m", type=int, required=True)
    parser.add_argument("--block_n", type=int, required=True)
    parser.add_argument(
        "--scale_type", type=str, choices=['block', 'global'], required=True,
        help="`block` = use block scaling where 32 elements share a scale; "
        "`global` = use a single global scale for all elements")
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument(
        "--disable_p_scaling", action="store_true", help="When set, we will use a fixed scale of 1.0 for all P blocks. "
        "Otherwise, we will compute and apply per-block scaling for the P matrix tensor. "
        "Only apply when block scaling is enabled. Ignored for global scaling.")
    parser.add_argument(
        "--scale_preshuffled", action="store_true",
        help="When set, we will preshuffle the K/V scales before passing to the kernel. "
        "Only works for block scaling.")
    parser.add_argument(
        "--p_k_width", type=int, choices=[8, 16], required=True,
        help="The K width (in elements) for p. When set to 8, we can remove the layout conversion for p")
    args = parser.parse_args()
    args = vars(args)
    launch(**args)
