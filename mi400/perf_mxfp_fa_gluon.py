"""
Multi-head attention kernel with MXFP data type in Gluon
"""
import hip

hip.hip.hipInit(0)

import pytest
import torch

import math
from einops import repeat

from triton import cdiv
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def _get_load_layout(packed):
    # tile: [16, 128]
    block_layout = ttgl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    # tile: [32, 64]
    block_layout_packed = ttgl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])

    return block_layout_packed if packed else block_layout


@gluon.constexpr_function
def _get_acc_layout():
    wmma_layout = ttgl.amd.AMDWMMALayout(version=3,  #
                                         transposed=True,  #
                                         warps_per_cta=[4, 1],  #
                                         instr_shape=[16, 16, 128])
    return wmma_layout


@gluon.constexpr_function
def _get_operand_layout(operand, packed):
    wmma_layout = _get_acc_layout()
    wmma_layout_packed = _get_acc_layout()
    wmma_layout_packed.instr_shape[-1] //= 2
    return ttgl.DotOperandLayout(operand, wmma_layout_packed if packed else wmma_layout, 16)


@gluon.constexpr_function
def _get_scale_layout(operand):
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


# ===-----------------------------------------------------------------------===#
# Gluon Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr,  #
                    q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                    o_ptr,  #
                    stride_q_z, stride_q_h, stride_q_m, stride_q_d,  #
                    stride_k_z, stride_k_h, stride_k_n, stride_k_d,  #
                    stride_v_z, stride_v_h, stride_v_n, stride_v_d,  #
                    stride_q_scale_z, stride_q_scale_h, stride_q_scale_m, stride_q_scale_d,  #
                    stride_k_scale_z, stride_k_scale_h, stride_k_scale_n, stride_k_scale_d,  #
                    stride_v_scale_z, stride_v_scale_h, stride_v_scale_d, stride_v_scale_n,  #
                    stride_oz, stride_oh, stride_om, stride_on,  #
                    sm_scale,  #
                    Q_TYPE: ttgl.constexpr,  #
                    KV_TYPE: ttgl.constexpr,  #
                    SEQLEN_Q: ttgl.constexpr,  #
                    SEQLEN_K: ttgl.constexpr,  #
                    NUM_HEADS: ttgl.constexpr,  #
                    HEAD_SZ: ttgl.constexpr,  #
                    BATCH: ttgl.constexpr,  #
                    BLOCK_M: ttgl.constexpr,  #
                    BLOCK_N: ttgl.constexpr):

    ttgl.static_assert(Q_TYPE == 'e5m2' or Q_TYPE == 'e4m3')
    ttgl.static_assert(KV_TYPE == 'e5m2' or KV_TYPE == 'e4m3' or KV_TYPE == 'e2m1')
    P_TYPE: ttgl.constexpr = Q_TYPE

    load_q_layout: ttgl.constexpr = _get_load_layout(packed=False)
    load_k_layout: ttgl.constexpr = _get_load_layout(packed=(KV_TYPE == 'e2m1'))
    load_v_layout: ttgl.constexpr = _get_load_layout(packed=False)

    q_layout: ttgl.constexpr = _get_operand_layout(0, packed=False)
    k_layout: ttgl.constexpr = _get_operand_layout(1, packed=(KV_TYPE == 'e2m1'))
    q_scale_layout: ttgl.constexpr = _get_scale_layout(0)
    k_scale_layout: ttgl.constexpr = _get_scale_layout(1)

    p_layout: ttgl.constexpr = _get_operand_layout(0, packed=False)
    v_layout: ttgl.constexpr = _get_operand_layout(1, packed=(KV_TYPE == 'e2m1'))
    p_scale_layout: ttgl.constexpr = _get_scale_layout(0)
    v_scale_layout: ttgl.constexpr = _get_scale_layout(1)

    acc_layout: ttgl.constexpr = _get_acc_layout()

    KV_PACK_DIV: ttgl.constexpr = 2 if KV_TYPE == 'e2m1' else 1

    # programs: (BATCH, NUM_HEADS, NUM_BLOCKS)
    off_z = ttgl.program_id(0)
    off_h = ttgl.program_id(1)
    off_m = ttgl.program_id(2)

    # offset for q and q_scale:
    # q       [BLOCK_M, HEAD_SZ]
    # q_scale [BLOCK_M, HEAD_SZ / 32]
    q_offs_m = BLOCK_M * off_m + \
               ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, load_q_layout))
    q_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(0, load_q_layout))
    q_offs = stride_q_z * off_z + \
             stride_q_h * off_h + \
             stride_q_m * q_offs_m[:, None] + \
             stride_q_d * q_offs_d[None, :]

    q_scale_offs_m = BLOCK_M * off_m + \
                     ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, q_scale_layout))
    q_scale_offs_d = ttgl.arange(0, HEAD_SZ // 32, ttgl.SliceLayout(0, q_scale_layout))
    q_scale_offs = stride_q_scale_z * off_z + \
                   stride_q_scale_h * off_h + \
                   stride_q_scale_m * q_scale_offs_m[:, None] + \
                   stride_q_scale_d * q_scale_offs_d[None, :]

    # offset for k, k_scale:
    # k       [HEAD_SZ / KV_PACK_DIV, BLOCK_N]
    # k_scale [BLOCK_N, HEAD_SZ / 32]
    k_offs_d = ttgl.arange(0, HEAD_SZ // KV_PACK_DIV, ttgl.SliceLayout(1, load_k_layout))
    k_offs_n = ttgl.arange(0, BLOCK_N, ttgl.SliceLayout(0, load_k_layout))
    k_offs = stride_k_z * off_z + \
             stride_k_h * off_h + \
             stride_k_d * k_offs_d[:, None] + \
             stride_k_n * k_offs_n[None, :]

    k_scale_offs_n = ttgl.arange(0, BLOCK_N, ttgl.SliceLayout(1, k_scale_layout))
    k_scale_offs_d = ttgl.arange(0, HEAD_SZ // 32, ttgl.SliceLayout(0, k_scale_layout))
    k_scale_offs = stride_k_scale_z * off_z + \
                   stride_k_scale_h * off_h + \
                   stride_k_scale_n * k_scale_offs_n[:, None] + \
                   stride_k_scale_d * k_scale_offs_d[None, :]

    # offset for v, v_scale:
    # v       [BLOCK_N / KV_PACK_DIV, HEAD_SZ]
    # v_scale [HEAD_SZ, BLOCK_N / 32]
    v_offs_n = ttgl.arange(0, BLOCK_N // KV_PACK_DIV, ttgl.SliceLayout(1, load_v_layout))
    v_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(0, load_v_layout))
    v_offs = stride_v_z * off_z + \
             stride_v_h * off_h + \
             stride_v_n * v_offs_n[:, None] + \
             stride_v_d * v_offs_d[None, :]

    v_scale_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(1, v_scale_layout))
    v_scale_offs_n = ttgl.arange(0, BLOCK_N // 32, ttgl.SliceLayout(0, v_scale_layout))
    v_scale_offs = stride_v_scale_z * off_z + \
                   stride_v_scale_h * off_h + \
                   stride_v_scale_d * v_scale_offs_d[:, None] + \
                   stride_v_scale_n * v_scale_offs_n[None, :]

    # load q and q_scale
    q_mask = q_offs_m[:, None] < SEQLEN_Q
    q = ttgl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = ttgl.convert_layout(q, q_layout)
    q_scale_mask = q_scale_offs_m[:, None] < SEQLEN_Q
    q_scale = ttgl.load(q_scale_ptr + q_scale_offs, mask=q_scale_mask, other=0x7F)

    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, acc_layout)

    for _ in range(0, ttgl.cdiv(SEQLEN_K, BLOCK_N)):
        # load k and k_scale
        k = ttgl.load(k_ptr + k_offs)
        k = ttgl.convert_layout(k, k_layout)
        k_scale = ttgl.load(k_scale_ptr + k_scale_offs)

        # compute q @ k.T
        qk = ttgl.full([BLOCK_M, BLOCK_N], 0.0, ttgl.float32, acc_layout)
        qk = ttgl.amd.gfx1250.wmma_scaled(q, q_scale, Q_TYPE, k, k_scale, KV_TYPE, qk)

        # get max scores so far
        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))
        m_ij_scaled = m_ij * sm_scale

        # scale and subtract max
        q_shifted = qk * sm_scale - m_ij_scaled[:, None]

        # compute scaled qk and softmax probabilities
        p = ttgl.exp2(q_shifted)

        # compute correction factor
        m_diff_scaled = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff_scaled)
        l_ij = ttgl.sum(p, 1)

        # update accumulator
        acc = acc * alpha[:, None]

        # load v and v_scale
        v = ttgl.load(v_ptr + v_offs)
        v = ttgl.convert_layout(v, v_layout)
        v_scale = ttgl.load(v_scale_ptr + v_scale_offs)

        # downcast p
        p = ttgl.convert_layout(p, p_layout)
        if P_TYPE == 'e4m3':
            p = p.to(ttgl.float8e4nv)
        else:
            p = p.to(ttgl.float8e5)

        # compute p @ v
        p_scale = ttgl.full([BLOCK_M, BLOCK_N // 32], 0x7F, ttgl.int8, p_scale_layout)
        acc = ttgl.amd.gfx1250.wmma_scaled(p, p_scale, P_TYPE, v, v_scale, KV_TYPE, acc)

        # advance k, k_scale, v, v_scale
        k_ptr += BLOCK_N * stride_k_n
        k_scale_ptr += BLOCK_N * stride_k_scale_n

        v_ptr += (BLOCK_N // KV_PACK_DIV) * stride_v_n
        v_scale_ptr += (BLOCK_N // 32) * stride_v_scale_n

        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # epilogue
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # store output
    o_offs_m = BLOCK_M * off_m + \
               ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, acc_layout))
    o_offs_n = ttgl.arange(0, HEAD_SZ, layout=ttgl.SliceLayout(0, acc_layout))
    o_offs = stride_oz * off_z + \
             stride_oh * off_h + \
             stride_om * o_offs_m[:, None] + \
             stride_on * o_offs_n[None, :]
    o_mask = o_offs_m[:, None] < SEQLEN_Q

    o = acc.to(o_ptr.dtype.element_ty)
    ttgl.store(o_ptr + o_offs, o, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
             q_scale: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,  #
             q_type: str, kv_type: str, BLOCK_M: int, BLOCK_N: int) -> torch.Tensor:
    batch, seqlen_q, num_heads, head_sz = q.shape
    _, seqlen_k, _, _ = k.shape
    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)

    # q: [BATCH, NUM_HEADS, SEQLEN_Q, HEAD_SZ]
    # k: [BATCH, NUM_HEADS, SEQLEN_K, HEAD_SZ / KV_PACK_DIV]
    # v: [BATCH, NUM_HEADS, SEQLEN_K / KV_PACK_DIV, HEAD_SZ]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
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

    # Each program holds
    # q: [1, 1, BLOCK_M, HEAD_SZ]
    # k: [1, 1, SEQLEN_K, HEAD_SZ / KV_PACK_DIV]
    # v: [1, 1, SEQLEN_K / KV_PACK_DIV, HEAD_SZ]
    grid = (batch, num_heads, cdiv(seqlen_q, BLOCK_M))
    attn_fwd_kernel[grid](
        q, k, v,  #
        q_scale, k_scale, v_scale,  #
        o,  #
        *q.stride(),  #
        *k.stride(),  #
        *v.stride(),  #
        *q_scale.stride(),  #
        *k_scale.stride(),  #
        *v_scale.stride(),  #
        *o.stride(),  #
        sm_scale,  #
        Q_TYPE=q_type,  #
        KV_TYPE=kv_type,  #
        SEQLEN_Q=seqlen_q,  #
        SEQLEN_K=seqlen_k,  #
        NUM_HEADS=num_heads,  #
        HEAD_SZ=head_sz,  #
        BATCH=batch,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        num_warps=4)

    return o.cpu().permute(0, 2, 1, 3)


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


@pytest.mark.parametrize("q_type,kv_type", [("e4m3", "e4m3"), ("e5m2", "e5m2"), ("e4m3", "e2m1"), ("e5m2", "e2m1")])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen_q", [256])
@pytest.mark.parametrize("seqlen_k", [256])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("head_sz", [128])
@pytest.mark.parametrize("block_m", [128, 64])
@pytest.mark.parametrize("block_n", [128])
def test_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_heads, head_sz, block_m, block_n):
    if q_type == "e5m2":
        pytest.skip("Skip e5m2 for now due to accuracy issue")

    torch.random.manual_seed(0)
    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=3)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = _create_scale(q_type, batch, seqlen_q, num_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = _create_scale(kv_type, batch, seqlen_k, num_heads, head_sz, scale_dim=1)

    o = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
