"""
This file implements a BSHD Flash Attention and tests against torch reference.
"""

# Enabling FFM
import hip

hip.hip.hipInit(0)

# Import ML libs
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import pytest


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr, out_ptr,  #
                    stride_qz, stride_qh, stride_qm, stride_qk,  #
                    stride_kz, stride_kh, stride_kn, stride_kk,  #
                    stride_vz, stride_vh, stride_vn, stride_vk,  #
                    stride_oz, stride_oh, stride_om, stride_on,  #
                    SM_SCALE: gl.constexpr,  #
                    SEQLEN_Q: gl.constexpr,  #
                    SEQLEN_K: gl.constexpr,  #
                    BLOCK_M: gl.constexpr,  #
                    BLOCK_N: gl.constexpr,  #
                    HEAD_SZ: gl.constexpr,  #
                    ):
    # This layout sets vector<8xf16> along fastest dim/head_sz and infer thread distribution needed to fit.
    BLOCK_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [256 // HEAD_SZ, HEAD_SZ // 8], [4, 1], [1, 0])
    K_TRANSPOSE_LAYOUT: gl.constexpr = gl.BlockedLayout([8, 1], [HEAD_SZ // 8, 256 // HEAD_SZ], [1, 4], [0, 1])
    WMMA_QK_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                        instr_shape=[16, 16, 32])
    WMMA_PV_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                        instr_shape=[16, 16, 32])

    seqlen_q = SEQLEN_Q
    seqlen_k = SEQLEN_K

    # workgroup offsets using delinearization
    off_z = gl.program_id(0)
    off_q_head = gl.program_id(1)
    off_k_head = off_q_head
    off_m = gl.program_id(2) * BLOCK_M
    n_blocks_n = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # q [BLOCK_M, HEAD_SZ]
    q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
              (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, BLOCK_LAYOUT)))[:, None] + stride_qk *
              (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, BLOCK_LAYOUT)))[None, :])

    # k [HEAD_SZ, BLOCK_N]
    k_offs = (stride_kz * off_z + stride_kh * off_k_head +
              stride_kk * gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(1, K_TRANSPOSE_LAYOUT))[:, None] +
              stride_kn * gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, K_TRANSPOSE_LAYOUT))[None, :])

    # v [BLOCK_N, BLOCK_DMODEL]
    v_offs = (stride_vz * off_z + stride_vh * off_k_head +
              stride_vn * gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, BLOCK_LAYOUT))[:, None] +
              stride_vk * gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, BLOCK_LAYOUT))[None, :])

    m_i = gl.full([BLOCK_M], float(-1e6), dtype=gl.float32, layout=gl.SliceLayout(1, WMMA_PV_LAYOUT))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, WMMA_PV_LAYOUT))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=WMMA_PV_LAYOUT)

    q_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, BLOCK_LAYOUT)))[:, None] < seqlen_q
    q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)
    q = gl.convert_layout(q, gl.DotOperandLayout(0, WMMA_QK_LAYOUT, 8))

    block_min = 0
    block_max = n_blocks_n * BLOCK_N

    RCP_LN2: gl.constexpr = 1.4426950408889634

    for block_id in range(block_min, block_max, BLOCK_N):
        k_mask = (block_id + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, K_TRANSPOSE_LAYOUT)))[None, :] < seqlen_k
        k = gl.amd.gfx1250.buffer_load(k_ptr, k_offs, mask=k_mask)
        k = gl.convert_layout(k, gl.DotOperandLayout(1, WMMA_QK_LAYOUT, 8))

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=WMMA_QK_LAYOUT)
        qk = gl.amd.gfx1250.wmma(q, k, qk)

        # Handle/pad unaligned M and K2 ids.
        qk_mask = (block_id + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_QK_LAYOUT)))[None, :] < seqlen_k
        qk = gl.where(qk_mask, qk, -1.0e6)

        # get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # scale and subtract max
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(q_shifted)

        # update l_ij before applying dropout
        l_ij = gl.sum(p, 1)

        # update output accumulator
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = gl.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]

        v_mask = (block_id + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, BLOCK_LAYOUT)))[:, None] < seqlen_k
        v = gl.amd.gfx1250.buffer_load(v_ptr, v_offs, mask=v_mask)
        v = gl.convert_layout(v, gl.DotOperandLayout(1, WMMA_PV_LAYOUT, 8))

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        p = gl.convert_layout(p, gl.DotOperandLayout(0, WMMA_PV_LAYOUT, 8))
        acc = gl.amd.gfx1250.wmma(p, v, acc)

        k_ptr += BLOCK_N * stride_kn
        v_ptr += BLOCK_N * stride_vn

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    out_offs = (stride_oz * off_z + stride_oh * off_q_head + stride_om *
                (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_PV_LAYOUT)))[:, None] + stride_on *
                (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, WMMA_PV_LAYOUT)))[None, :])

    op = acc.to(out_ptr.dtype.element_ty)

    out_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_PV_LAYOUT)))[:, None] < seqlen_q
    gl.amd.gfx1250.buffer_store(op, out_ptr, out_offs, mask=out_mask)


def generate_configs():
    base_configs = [
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 512, "SEQLEN_K": 512, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 32
        }),
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 1024, "SEQLEN_K": 1024, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 64,
            "BLOCK_M": 128, "BLOCK_N": 32
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 3, "SEQLEN_K": 32, "NUM_Q_HEADS": 4, "NUM_K_HEADS": 4, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 32
        }),
        pytest.param({
            "BATCH": 4, "SEQLEN_Q": 1, "SEQLEN_K": 100, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 32, "BLOCK_M":
            128, "BLOCK_N": 32
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 1, "SEQLEN_K": 30, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 32, "BLOCK_M":
            128, "BLOCK_N": 32
        }),
    ]
    return base_configs


@pytest.mark.parametrize("config", generate_configs())
def test_attention(config):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    q = torch.randn((BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ), dtype=dtype)
    k = torch.randn((BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ), dtype=dtype)
    v = torch.randn((BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ), dtype=dtype)
    sm_scale = 1.0 / (HEAD_SZ**0.5)

    o = torch.zeros_like(q, dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    o = o.cuda()

    grid = (
        BATCH,
        NUM_Q_HEADS,
        ((SEQLEN_Q + BLOCK_M - 1) // BLOCK_M),
    )

    attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        sm_scale,
        SEQLEN_Q,
        SEQLEN_K,
        BLOCK_M,
        BLOCK_N,
        HEAD_SZ,
        num_warps=4,
    )
    o = o.cpu()
    rtol = 0.004
    atol = 0.004
    torch.cuda.synchronize()
    torch.testing.assert_allclose(o, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    config = {
        "BATCH": 8, "SEQLEN_Q": 1024, "SEQLEN_K": 1024, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 64, "BLOCK_M":
        32, "BLOCK_N": 32
    }
    test_attention(config)
