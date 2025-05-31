import triton
import triton.language as tl
from dropout import fast_dropout_mask

from load_fn import load_fn

# yapf: disable
@triton.jit
def _attn_fwd_inner(
        # Inputs
        acc,
        l_i, m_i, Qk_scale,
        q,
        k_ptrs,
        v_ptrs,
        bias_ptrs,
        stride_kn, stride_vk, stride_bn,
        # Task positions
        start_m, block_min, block_max,
        actual_seqlen_k, actual_seqlen_q, Head_dim,
        # Dropout
        idropout_p, philox_seed, batch_philox_offset, philox_offset_stride,
        encoded_sm_base,
        # CAUSAL (Partial block)
        offs_n_causal,
        masked_blocks,
        n_extra_tokens,
        # Alibi
        alibi_slope,
        q_descale, k_descale, v_descale, p_scale,
        # CAUSAL and BLOCK SIZES
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        OFFS_M: tl.constexpr,
        OFFS_N: tl.constexpr,
        PRE_LOAD_V: tl.constexpr,
        MASK_STEPS: tl.constexpr,
        ENABLE_DROPOUT: tl.constexpr,
        RETURN_ENCODED_SOFTMAX: tl.constexpr,
        PADDED_HEAD: tl.constexpr,
        INT8_GEMM: tl.constexpr,
        INT8_KV: tl.constexpr,
        USE_P_SCALE: tl.constexpr):
    # yapf: enable
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, Head_dim, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, Head_dim)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        if INT8_GEMM:
            qk += ((((tl.dot(q, k).to(tl.float32) * q_descale)) * k_descale) * Qk_scale)
        else:
            if INT8_KV:
                k = (k * k_descale).to(q.type.element_ty)
            qk += (tl.dot(q, k) * Qk_scale)

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk += (alibi_block * 1.44269504089)  # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if MASK_STEPS or IS_CAUSAL:
            if Qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            # philox_offset = batch_philox_offset + start_M * Max_seqlen_k + start_N
            keep = fast_dropout_mask(philox_seed, idropout_p,
                                     batch_philox_offset, start_M, start_N,
                                     BLOCK_M, BLOCK_N, philox_offset_stride)
            if RETURN_ENCODED_SOFTMAX:
                # yapf: disable
                mstore2d(tl.where(keep, p, -p).to(encoded_sm_base.type.element_ty),
                         BLOCK_M,
                         BLOCK_N,
                         o_base=encoded_sm_base,
                         o_start_row=start_m * BLOCK_M,
                         o_start_col=start_n,
                         o_rows=actual_seqlen_q,
                         o_cols=actual_seqlen_k,
                         stride_row=Max_seqlen_k,
                         stride_col=1)
                # yapf: enable
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            # yapf: disable
            mstore2d(p.to(encoded_sm_base.type.element_ty),
                     BLOCK_M,
                     BLOCK_N,
                     o_base=encoded_sm_base,
                     o_start_row=start_m * BLOCK_M,
                     o_start_col=start_n,
                     o_rows=actual_seqlen_q,
                     o_cols=actual_seqlen_k,
                     stride_row=Max_seqlen_k,
                     stride_col=1)
            # yapf: enable
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, Head_dim)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if INT8_GEMM:
            if USE_P_SCALE:
                p = (p * p_scale).to(tl.int8)
                # They are all int8
                acc += tl.dot(p, v)
            else:
                # v is in int8 but p is not, we want the gemm in p's type
                acc += tl.dot(p, v.to(p.type.element_ty))
        else:
            if INT8_KV:
                v = (v * v_descale).to(p.type.element_ty)
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
    return acc, l_i, m_i
