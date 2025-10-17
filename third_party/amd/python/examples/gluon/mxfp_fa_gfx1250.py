"""
Multi-head attention kernel with MXFP data type in Gluon
"""
# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import re
import pytest
import torch

import math
from einops import repeat

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
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def _get_acc_layout():
    wmma_layout = ttgl.amd.AMDWMMALayout(version=3,  #
                                         transposed=True,  #
                                         warps_per_cta=[4, 1],  #
                                         instr_shape=[16, 16, 128])
    return wmma_layout


@gluon.constexpr_function
def _get_operand_reg_layout(operand, packed):
    wmma_layout = _get_acc_layout()
    wmma_layout_packed = _get_acc_layout()
    wmma_layout_packed.instr_shape[-1] //= 2
    return ttgl.DotOperandLayout(operand, wmma_layout_packed if packed else wmma_layout, 16)


@gluon.constexpr_function
def _get_scale_reg_layout(operand):
    a_scale_layout = ttgl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2]],  #
                                                  lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],  #
                                                  warp_bases=[[16, 0], [32, 0]],  #
                                                  block_bases=[],  #
                                                  shape=[64, 4])
    b_scale_layout = ttgl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2]],  #
                                                  lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],  #
                                                  warp_bases=[[0, 0], [0, 0]],  #
                                                  block_bases=[],  #
                                                  shape=[16, 4])
    return a_scale_layout if operand == 0 else b_scale_layout


@gluon.constexpr_function
def _get_operand_smem_layout(shape):
    padding_interval = shape[1]
    padding_amount = 16
    return ttgl.PaddedSharedLayout.with_identity_for([[padding_interval, padding_amount]], shape, [1, 0])


@gluon.constexpr_function
def _get_scale_smem_layout():
    # TODO: improve scale shared layout
    return ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])


@gluon.constexpr_function
def _get_scale_load_layout():
    # TODO: improve scale load layout
    return ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])


# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


@aggregate
class AttentionConfig:
    Q_TYPE: ttgl.constexpr
    P_TYPE: ttgl.constexpr
    KV_TYPE: ttgl.constexpr
    SEQLEN_Q: ttgl.constexpr
    SEQLEN_K: ttgl.constexpr
    NUM_HEADS: ttgl.constexpr
    HEAD_SZ: ttgl.constexpr
    BLOCK_M: ttgl.constexpr
    BLOCK_N: ttgl.constexpr
    KV_PACK_DIV: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr

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

    def __init__(self, Q_TYPE, P_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert P_TYPE == Q_TYPE
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']

        # constants
        self.Q_TYPE = ttgl.constexpr(Q_TYPE)
        self.P_TYPE = ttgl.constexpr(P_TYPE)
        self.KV_TYPE = ttgl.constexpr(KV_TYPE)
        self.SEQLEN_Q = ttgl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = ttgl.constexpr(SEQLEN_K)
        self.NUM_HEADS = ttgl.constexpr(NUM_HEADS)
        self.HEAD_SZ = ttgl.constexpr(HEAD_SZ)
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.BLOCK_N = ttgl.constexpr(BLOCK_N)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)

        KV_PACK_DIV = ttgl.constexpr(2 if KV_TYPE == 'e2m1' else 1)
        self.KV_PACK_DIV = KV_PACK_DIV

        # layouts
        self.q_layout = ttgl.constexpr(_get_operand_reg_layout(0, packed=False))
        self.q_scale_layout = ttgl.constexpr(_get_scale_reg_layout(0))

        self.k_smem_layout = ttgl.constexpr(_get_operand_smem_layout([HEAD_SZ // KV_PACK_DIV, BLOCK_N]))
        self.k_layout = ttgl.constexpr(_get_operand_reg_layout(1, packed=(KV_TYPE == 'e2m1')))
        self.k_scale_load_layout = ttgl.constexpr(_get_scale_load_layout())
        self.k_scale_smem_layout = ttgl.constexpr(_get_scale_smem_layout())
        self.k_scale_layout = ttgl.constexpr(_get_scale_reg_layout(1))

        self.p_layout = ttgl.constexpr(_get_operand_reg_layout(0, packed=False))
        self.p_scale_layout = ttgl.constexpr(_get_scale_reg_layout(0))

        self.v_smem_layout = ttgl.constexpr(_get_operand_smem_layout([BLOCK_N // KV_PACK_DIV, HEAD_SZ]))
        self.v_layout = ttgl.constexpr(_get_operand_reg_layout(1, packed=(KV_TYPE == 'e2m1')))
        self.v_scale_load_layout = ttgl.constexpr(_get_scale_load_layout())
        self.v_scale_smem_layout = ttgl.constexpr(_get_scale_smem_layout())
        self.v_scale_layout = ttgl.constexpr(_get_scale_reg_layout(1))

        self.acc_layout = ttgl.constexpr(_get_acc_layout())


# ===-----------------------------------------------------------------------===#
# Kernel Primitives
# ===-----------------------------------------------------------------------===#


@aggregate
class AttentionProgram:
    cfg: AttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor

    k_desc: tdm.tensor_descriptor
    k_scale_ptr: ttgl.tensor
    k_scale_offs: ttgl.tensor
    k_buffer: ttgl.shared_memory_descriptor
    k_scale_buffer: ttgl.shared_memory_descriptor

    v_desc: tdm.tensor_descriptor
    v_scale_ptr: ttgl.tensor
    v_scale_offs: ttgl.tensor
    v_buffer: ttgl.shared_memory_descriptor
    v_scale_buffer: ttgl.shared_memory_descriptor

    o_ptr: ttgl.tensor
    o_offs: ttgl.tensor
    o_mask: ttgl.tensor

    sm_scale: ttgl.constexpr

    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_desc, k_scale_ptr, k_scale_offs, k_buffer, k_scale_buffer,  #
                 v_desc, v_scale_ptr, v_scale_offs, v_buffer, v_scale_buffer,  #
                 o_ptr, o_offs, o_mask,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_desc = k_desc
        self.k_scale_ptr = k_scale_ptr
        self.k_scale_offs = k_scale_offs
        self.k_buffer = k_buffer
        self.k_scale_buffer = k_scale_buffer
        self.v_desc = v_desc
        self.v_scale_ptr = v_scale_ptr
        self.v_scale_offs = v_scale_offs
        self.v_buffer = v_buffer
        self.v_scale_buffer = v_scale_buffer
        self.o_ptr = o_ptr
        self.o_offs = o_offs
        self.o_mask = o_mask
        self.sm_scale = ttgl.constexpr(sm_scale)

    @gluon.jit
    def initialize(cfg,  #
                   q_ptr, q_scale_ptr,  #
                   k_ptr, k_scale_ptr,  #
                   v_ptr, v_scale_ptr,  #
                   o_ptr,  #
                   sm_scale: ttgl.constexpr):
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_HEADS: ttgl.constexpr = cfg.NUM_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        KV_PACK_DIV: ttgl.constexpr = cfg.KV_PACK_DIV
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        # programs: (NUM_HEADS, NUM_BLOCKS, BATCH)
        off_z = ttgl.program_id(2)
        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)

        # compute offsets for q and q_scale
        # q       [BLOCK_M, HEAD_SZ]
        # q_scale [BLOCK_M, HEAD_SZ / 32]
        q_off_zh = SEQLEN_Q * HEAD_SZ * (NUM_HEADS * off_z + off_h)
        q_offs_m = BLOCK_M * off_m + \
                   ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, cfg.q_layout))
        q_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(0, cfg.q_layout))
        q_offs = q_off_zh + \
                q_offs_m[:, None] * HEAD_SZ + \
                q_offs_d[None, :]

        q_scale_off_zh = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_HEADS * off_z + off_h)
        q_scale_offs_m = BLOCK_M * off_m + \
                        ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, cfg.q_scale_layout))
        q_scale_offs_d = ttgl.arange(0, HEAD_SZ // 32, ttgl.SliceLayout(0, cfg.q_scale_layout))
        q_scale_offs = q_scale_off_zh + \
                    q_scale_offs_m[:, None] * (HEAD_SZ // 32) + \
                    q_scale_offs_d[None, :]

        # create descriptor and buffer for k and k_scale
        # k       [HEAD_SZ / KV_PACK_DIV, BLOCK_N]
        # k_scale [BLOCK_N, HEAD_SZ / 32]
        k_off_zh = SEQLEN_K * (HEAD_SZ // KV_PACK_DIV) * (NUM_HEADS * off_z + off_h)
        k_desc = tdm.make_tensor_descriptor(  #
            base=k_off_zh + k_ptr,  #
            shape=[HEAD_SZ // KV_PACK_DIV, SEQLEN_K],  #
            strides=[SEQLEN_K, 1],  #
            block_shape=[HEAD_SZ // KV_PACK_DIV, BLOCK_N],  #
            layout=cfg.k_smem_layout)
        k_buffer = ttgl.allocate_shared_memory(  #
            k_desc.dtype,  #
            [NUM_BUFFERS] + k_desc.block_shape,  #
            k_desc.layout)

        k_scale_off_zh = SEQLEN_K * (HEAD_SZ // 32) * (NUM_HEADS * off_z + off_h)
        k_scale_offs_n = ttgl.arange(0, BLOCK_N, ttgl.SliceLayout(1, cfg.k_scale_load_layout))
        k_scale_offs_d = ttgl.arange(0, HEAD_SZ // 32, ttgl.SliceLayout(0, cfg.k_scale_load_layout))
        k_scale_offs = k_scale_off_zh + \
                    k_scale_offs_n[:, None] * (HEAD_SZ // 32) + \
                    k_scale_offs_d[None, :]
        k_scale_buffer = ttgl.allocate_shared_memory(  #
            k_scale_ptr.dtype.element_ty,  #
            [NUM_BUFFERS] + [BLOCK_N, HEAD_SZ // 32],  #
            cfg.k_scale_smem_layout)

        # create descriptor and buffer for v and v_scale
        # v       [BLOCK_N / KV_PACK_DIV, HEAD_SZ]
        # v_scale [HEAD_SZ, BLOCK_N / 32]
        v_off_zh = (SEQLEN_K // KV_PACK_DIV) * HEAD_SZ * (NUM_HEADS * off_z + off_h)
        v_desc = tdm.make_tensor_descriptor(  #
            base=v_off_zh + v_ptr,  #
            shape=[SEQLEN_K // KV_PACK_DIV, HEAD_SZ],  #
            strides=[HEAD_SZ, 1],  #
            block_shape=[BLOCK_N // KV_PACK_DIV, HEAD_SZ],  #
            layout=cfg.v_smem_layout)
        v_buffer = ttgl.allocate_shared_memory(  #
            v_desc.dtype,  #
            [NUM_BUFFERS] + v_desc.block_shape,  #
            v_desc.layout)

        v_scale_off_zh = (SEQLEN_K // 32) * HEAD_SZ * (NUM_HEADS * off_z + off_h)
        v_scale_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(1, cfg.v_scale_load_layout))
        v_scale_offs_n = ttgl.arange(0, BLOCK_N // 32, ttgl.SliceLayout(0, cfg.v_scale_load_layout))
        v_scale_offs = v_scale_off_zh + \
                    v_scale_offs_d[:, None] * (SEQLEN_K // 32) + \
                    v_scale_offs_n[None, :]
        v_scale_buffer = ttgl.allocate_shared_memory(  #
            v_scale_ptr.dtype.element_ty,  #
            [NUM_BUFFERS] + [HEAD_SZ, BLOCK_N // 32],  #
            cfg.v_scale_smem_layout)

        # output [BLOCK_M, HEAD_SZ]
        o_offs_zh = SEQLEN_Q * HEAD_SZ * (NUM_HEADS * off_z + off_h)
        o_offs_m = BLOCK_M * off_m + \
                ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, cfg.acc_layout))
        o_offs_n = ttgl.arange(0, HEAD_SZ, layout=ttgl.SliceLayout(0, cfg.acc_layout))
        o_offs = o_offs_zh + \
                o_offs_m[:, None] * HEAD_SZ + \
                o_offs_n[None, :]
        o_mask = o_offs_m[:, None] < SEQLEN_Q

        # load q and q_scale
        q_mask = q_offs_m[:, None] < SEQLEN_Q
        q = buffer_load(q_ptr, q_offs, mask=q_mask, other=0.0)
        q_scale_mask = q_scale_offs_m[:, None] < SEQLEN_Q
        q_scale = buffer_load(q_scale_ptr, q_scale_offs, mask=q_scale_mask, other=0x7F)

        # create the program
        return AttentionProgram(cfg,  #
                                q, q_scale,  #
                                k_desc, k_scale_ptr, k_scale_offs, k_buffer, k_scale_buffer,  #
                                v_desc, v_scale_ptr, v_scale_offs, v_buffer, v_scale_buffer,  #
                                o_ptr, o_offs, o_mask,  #
                                sm_scale)

    @gluon.jit
    def issue_global_load_k(self, i):
        cfg = self.cfg
        k_step: ttgl.constexpr = cfg.BLOCK_N
        k_scale_step: ttgl.constexpr = cfg.BLOCK_N * (cfg.HEAD_SZ // 32)

        buf = i % cfg.NUM_BUFFERS
        k_buffer = self.k_buffer.index(buf)
        k_scale_buffer = self.k_scale_buffer.index(buf)

        tdm.async_load(self.k_desc, [0, i * k_step], k_buffer)

        k_scale_ptrs = (self.k_scale_ptr + i * k_scale_step) + self.k_scale_offs
        cp.async_copy_global_to_shared(k_scale_buffer, k_scale_ptrs)

    @gluon.jit
    def issue_global_load_v(self, i):
        cfg = self.cfg
        v_step: ttgl.constexpr = cfg.BLOCK_N // cfg.KV_PACK_DIV
        v_scale_step: ttgl.constexpr = (cfg.BLOCK_N // 32)

        buf = i % cfg.NUM_BUFFERS
        v_buffer = self.v_buffer.index(buf)
        v_scale_buffer = self.v_scale_buffer.index(buf)

        tdm.async_load(self.v_desc, [i * v_step, 0], v_buffer)

        v_scale_ptrs = (self.v_scale_ptr + i * v_scale_step) + self.v_scale_offs
        cp.async_copy_global_to_shared(v_scale_buffer, v_scale_ptrs)

    @gluon.jit
    def shared_load_k(self, i, wait_count):
        cfg = self.cfg

        buf = i % cfg.NUM_BUFFERS
        k_buffer = self.k_buffer.index(buf)
        k_scale_buffer = self.k_scale_buffer.index(buf)

        self._async_wait(wait_count)
        k = k_buffer.load(cfg.k_layout)
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)
        return k, k_scale

    @gluon.jit
    def shared_load_v(self, i, wait_count):
        cfg = self.cfg

        buf = i % cfg.NUM_BUFFERS
        v_buffer = self.v_buffer.index(buf)
        v_scale_buffer = self.v_scale_buffer.index(buf)

        self._async_wait(wait_count)
        v = v_buffer.load(cfg.v_layout)
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)
        return v, v_scale

    @gluon.jit
    def compute_qk(self, i, k, k_scale):
        cfg = self.cfg

        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, zero)
        return qk

    @gluon.jit
    def compute_pv(self, i, p, v, v_scale, acc):
        cfg = self.cfg

        p, p_scale = self._downcast_to_mxfp(p, cfg.P_TYPE, [cfg.BLOCK_M, cfg.BLOCK_N // 32], cfg.p_scale_layout)
        p = ttgl.convert_layout(p, cfg.p_layout)

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def softmax0(self, i, qk, m_i):
        sm_scale: ttgl.constexpr = self.sm_scale

        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        return p, m_diff, m_ij

    @gluon.jit
    def softmax1(self, i, p, m_diff, acc, l_i):
        alpha = ttgl.exp2(m_diff)
        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        return acc, l_i

    @gluon.jit
    def store_output(self, acc):
        o = acc.to(self.o_ptr.dtype.element_ty)
        buffer_store(o, self.o_ptr, self.o_offs, mask=self.o_mask)

    @gluon.jit
    def _async_wait(self, count):
        tdm.async_wait(count)
        cp.async_wait(count)

    @gluon.jit
    def _downcast_to_mxfp(self, x, dtype: ttgl.constexpr, shape: ttgl.constexpr, layout: ttgl.constexpr):
        # TODO: support better downcast with scale
        if dtype == 'e4m3':
            value = x.to(ttgl.float8e4nv)
        else:
            value = x.to(ttgl.float8e5)
        scale = ttgl.full(shape, 0x7F, ttgl.int8, layout)
        return value, scale


# ===-----------------------------------------------------------------------===#
# Gluon Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr,  #
                    q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                    o_ptr,  #
                    sm_scale: ttgl.constexpr,  #
                    Q_TYPE: ttgl.constexpr,  #
                    KV_TYPE: ttgl.constexpr,  #
                    SEQLEN_Q: ttgl.constexpr,  #
                    SEQLEN_K: ttgl.constexpr,  #
                    NUM_HEADS: ttgl.constexpr,  #
                    HEAD_SZ: ttgl.constexpr,  #
                    BLOCK_M: ttgl.constexpr,  #
                    BLOCK_N: ttgl.constexpr):
    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)

    # init program
    cfg = AttentionConfig(  #
        Q_TYPE, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, 1)
    pgm = AttentionProgram.initialize(  #
        cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    for i in range(0, end):
        pgm.issue_global_load_k(i)
        k, k_scale = pgm.shared_load_k(i, wait_count=0)
        p = pgm.compute_qk(i, k, k_scale)
        p, m, m_i = pgm.softmax0(i, p, m_i)
        acc, l_i = pgm.softmax1(i, p, m, acc, l_i)
        pgm.issue_global_load_v(i)
        v, v_scale = pgm.shared_load_v(i, wait_count=0)
        acc = pgm.compute_pv(i, p, v, v_scale, acc)

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    pgm.store_output(acc)


@gluon.jit
def attn_fwd_pipelined_kernel(q_ptr, k_ptr, v_ptr,  #
                              q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                              o_ptr,  #
                              sm_scale: ttgl.constexpr,  #
                              Q_TYPE: ttgl.constexpr,  #
                              KV_TYPE: ttgl.constexpr,  #
                              SEQLEN_Q: ttgl.constexpr,  #
                              SEQLEN_K: ttgl.constexpr,  #
                              NUM_HEADS: ttgl.constexpr,  #
                              HEAD_SZ: ttgl.constexpr,  #
                              BLOCK_M: ttgl.constexpr,  #
                              BLOCK_N: ttgl.constexpr):
    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)

    # init program
    cfg = AttentionConfig(  #
        Q_TYPE, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, 2)
    pgm = AttentionProgram.initialize(  #
        cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    # pipeline prologue (-4)
    pgm.issue_global_load_k(0)

    # pipeline prologue (-2)
    pgm.issue_global_load_k(1)

    k0, k0_scale = pgm.shared_load_k(0, wait_count=1)

    pgm.issue_global_load_v(0)

    p0 = pgm.compute_qk(0, k0, k0_scale)

    pgm.issue_global_load_k(2)

    p0, m0, m_i = pgm.softmax0(0, p0, m_i)
    k1, k1_scale = pgm.shared_load_k(1, wait_count=2)

    pgm.issue_global_load_v(1)

    # pipeline loop (0 to end-4)
    for i in range(0, end - 2, 2):
        p1 = pgm.compute_qk(i + 1, k1, k1_scale)
        acc, l_i = pgm.softmax1(i, p0, m0, acc, l_i)
        v0, v0_scale = pgm.shared_load_v(i, wait_count=2)

        pgm.issue_global_load_k(i + 3)

        acc = pgm.compute_pv(i, p0, v0, v0_scale, acc)
        p1, m1, m_i = pgm.softmax0(i + 1, p1, m_i)
        k0, k0_scale = pgm.shared_load_k(i + 2, wait_count=2)

        pgm.issue_global_load_v(i + 2)

        p0 = pgm.compute_qk(i + 2, k0, k0_scale)
        acc, l_i = pgm.softmax1(i + 1, p1, m1, acc, l_i)
        v1, v1_scale = pgm.shared_load_v(i + 1, wait_count=2)

        if i + 4 < end:
            pgm.issue_global_load_k(i + 4)

        acc = pgm.compute_pv(i + 1, p1, v1, v1_scale, acc)
        p0, m0, m_i = pgm.softmax0(i + 2, p0, m_i)
        k1, k1_scale = pgm.shared_load_k(i + 3, wait_count=2)

        pgm.issue_global_load_v(i + 3)

    # pipeline epilogue (end-2)
    p1 = pgm.compute_qk(end - 1, k1, k1_scale)
    acc, l_i = pgm.softmax1(end - 2, p0, m0, acc, l_i)
    v0, v0_scale = pgm.shared_load_v(end - 2, wait_count=1)

    acc = pgm.compute_pv(end - 2, p0, v0, v0_scale, acc)
    p1, m1, m_i = pgm.softmax0(end - 1, p1, m_i)

    acc, l_i = pgm.softmax1(end - 1, p1, m1, acc, l_i)
    v1, v1_scale = pgm.shared_load_v(end - 1, wait_count=0)

    acc = pgm.compute_pv(end - 1, p1, v1, v1_scale, acc)

    # write output
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    pgm.store_output(acc)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
             q_scale: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,  #
             q_type: str, kv_type: str, BLOCK_M: int, BLOCK_N: int, pipelined: bool = False):
    batch, seqlen_q, num_heads, head_sz = q.shape
    _, seqlen_k, _, _ = k.shape
    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)

    # q: [BATCH, NUM_HEADS, SEQLEN_Q, HEAD_SZ]
    # k: [BATCH, NUM_HEADS, HEAD_SZ / KV_PACK_DIV, SEQLEN_K]
    # v: [BATCH, NUM_HEADS, SEQLEN_K / KV_PACK_DIV, HEAD_SZ]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 3, 1).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    # q_scale: [BATCH, NUM_HEADS, SEQLEN_Q, HEAD_SZ / 32]
    # k_scale: [BATCH, NUM_HEADS, SEQLEN_K, HEAD_SZ / 32]
    # v_scale: [BATCH, NUM_HEADS, HEAD_SZ, SEQLEN_K / 32]
    q_scale = q_scale.permute(0, 2, 1, 3).contiguous()
    k_scale = k_scale.permute(0, 2, 1, 3).contiguous()
    v_scale = v_scale.permute(0, 2, 3, 1).contiguous()
    # o: [BATCH, NUM_HEADS, SEQLEN_Q, HEAD_SZ]
    o = torch.zeros_like(q, dtype=torch.bfloat16)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    q_scale = q_scale.cuda()
    k_scale = k_scale.cuda()
    v_scale = v_scale.cuda()
    o = o.cuda()

    # Use (NUM_HEADS, NUM_BLOCKS, BATCH) for better xcd locality
    grid = (num_heads, cdiv(seqlen_q, BLOCK_M), batch)
    kargs = [
        q, k, v, q_scale, k_scale, v_scale, o, sm_scale,  #
        q_type, kv_type, seqlen_q, seqlen_k, num_heads, head_sz, BLOCK_M, BLOCK_N
    ]
    if pipelined:
        assert cdiv(seqlen_k, BLOCK_N) > 4
        assert cdiv(seqlen_k, BLOCK_N) % 2 == 0
        kernel = attn_fwd_pipelined_kernel[grid](*kargs, num_warps=4)
    else:
        kernel = attn_fwd_kernel[grid](*kargs, num_warps=4)

    return o.cpu().permute(0, 2, 1, 3), kernel


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _attn_fwd_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
                  q_scale: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
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
        v_ref = v.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == 'e5m2':
        sig = torch.randint(0, 2, size, dtype=torch.uint8)
        exp = torch.randint(0, 2**5, size, dtype=torch.uint8)
        man = torch.randint(0, 2**2, size, dtype=torch.uint8)
        v = ((sig << 7) | (exp << 2) | man).type(torch.uint8)
        v[(exp << 2) | man >= 0x7C] = 0x00  # avoid NaN and Inf
        v_ref = v.view(torch.float8_e5m2).to(torch.float32)
    else:
        assert dtype == 'e2m1'
        assert pack_dim >= 0
        v_mxfp4 = MXFP4Tensor(size=size).random()
        v = v_mxfp4.to_packed_tensor(pack_dim)
        v_ref = v_mxfp4.to(torch.float32)
    return v, v_ref


def _create_scale(dtype: str, b: int, s: int, h: int, d: int, scale_dim: int):
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


def get_variants():
    variants = [
        # q_type, kv_type
        ("e4m3", "e4m3"),
        ("e4m3", "e2m1"),
        # skip e5m2 for now due to accuracy issue
        # ("e5m2", "e5m2"),
        # ("e5m2", "e2m1"),
    ]
    return variants


@pytest.mark.parametrize("q_type,kv_type", get_variants())
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen_q", [256])
@pytest.mark.parametrize("seqlen_k", [1024])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("head_sz", [128])
@pytest.mark.parametrize("block_m", [128])
@pytest.mark.parametrize("block_n", [128])
@pytest.mark.parametrize("pipelined", [False, True])
def test_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_heads, head_sz, block_m, block_n, pipelined):
    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=3)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = _create_scale(q_type, batch, seqlen_q, num_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=1)

    o, _ = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n, pipelined)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=7.5e-2)


if __name__ == "__main__":
    configs = []
    for q_type, kv_type in get_variants():
        for pipelined in [False, True]:
            configs.append({
                "q_type": q_type, "kv_type": kv_type, "batch": 1, "seqlen_q": 256, "seqlen_k": 1024, "num_heads": 1,
                "head_sz": 128, "block_m": 128, "block_n": 128, "pipelined": pipelined
            })

    def launch(q_type, kv_type, batch, seqlen_q, seqlen_k, num_heads, head_sz, block_m, block_n, pipelined):
        q, _ = _create_operand(q_type, batch, seqlen_q, num_heads, head_sz)
        k, _ = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=3)
        v, _ = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=1)
        q_scale, _ = _create_scale(q_type, batch, seqlen_q, num_heads, head_sz, scale_dim=3)
        k_scale, _ = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=3)
        v_scale, _ = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=1)

        _, kernel = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n, pipelined)
        amdgcn = kernel.asm['amdgcn']

        sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
        sgpr_spill_count = int(re.search(r'\.sgpr_spill_count:\s+(\d+)', amdgcn).group(1))
        vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
        vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
        print(f"- sgpr_count: {sgpr_count}\n"
              f"- sgpr_spill_count: {sgpr_spill_count}\n"
              f"- vgpr_count: {vgpr_count}\n"
              f"- vgpr_spill_count: {vgpr_spill_count}\n")

    for config in configs:
        print(f"Config: {config}")
        launch(**config)
