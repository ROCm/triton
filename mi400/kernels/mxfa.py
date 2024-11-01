"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention v2 algorithm
See https://tridao.me/publications/flash2/flash2.pdf
Credits:
AMD Triton kernels team
OpenAI kernel team
Currently only the forward kernel is supported, and contains these features:
1) Arbitrary Q and KV sequence lengths
2) Arbitrary head sizes
3) Multi and grouped query attention
4) Variable sequence lengths
"""

import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_s, k_ptrs, k_scale_ptrs, v_ptrs, v_scale_ptrs, stride_kn, stride_vk, start_m,
                    actual_seqlen_k, actual_seqlen_q, block_min, block_max, BLOCK_M: tl.constexpr,
                    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M: tl.constexpr, OFFS_N: tl.constexpr,
                    PADDED_HEAD: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, QK_SCALE: tl.constexpr):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        # k = tl.load(k_ptrs)#, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        # k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        k = tl.load(k_ptrs)
        k_s = tl.load(k_scale_ptrs)
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        # -- compute qk ----
        qk = tl.dot_scaled(q, q_s, "e5m2", k, k_s, "e2m1")

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * QK_SCALE
        qk = qk * QK_SCALE - m_ij_scaled[:, None]
        p = tl.math.exp2(qk)

        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i * QK_SCALE - m_ij_scaled)
        acc = acc * alpha[:, None]
        v_s = tl.load(v_scale_ptrs)
        # v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        v = tl.load(v_ptrs)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        acc += tl.dot_scaled(p.to(tl.float8e5), None, "e5m2", v, v_s, "e2m1")

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    return acc, l_i


@triton.jit
def mxfa(Q, K, V, qscale, kscale, vscale, SM_SCALE: tl.constexpr, Out, seqlen_q, seqlen_k, stride_qz, stride_qh,
         stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
         stride_oz, stride_oh, stride_om, stride_on, HQ: tl.constexpr, HK: tl.constexpr,
         ACTUAL_BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
         scale_block: tl.constexpr, scale_stride: tl.constexpr):

    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n2 = tl.arange(0, BLOCK_N // 2)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d2 = tl.arange(0, BLOCK_DMODEL // 2)
    offs_db = tl.arange(0, BLOCK_DMODEL // scale_block)

    continue_condition = True  # as we can't have return statements inside while loop in Triton
    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    if start_m * BLOCK_M > seqlen_q:
        continue_condition = False
        # return

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)

    if continue_condition:
        # If MQA / GQA, set the K and V head offsets appropriately.
        GROUP_SIZE: tl.constexpr = HQ // HK
        if GROUP_SIZE != 1:
            off_h_k = off_h_q // GROUP_SIZE
        else:
            off_h_k = off_h_q

        n_extra_tokens = 0
        if seqlen_k < BLOCK_N:
            n_extra_tokens = BLOCK_N - seqlen_k
        elif seqlen_k % BLOCK_N:
            n_extra_tokens = seqlen_k % BLOCK_N
        PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

        # Compute pointers for all the tensors used in this kernel.
        q_offset = off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_scale_ptrs = qscale + q_offset + offs_m[:, None] * scale_stride + offs_db[None, :]

        k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
        k_ptrs = K + k_offset + offs_d2[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_scale_ptrs = kscale + offs_n[:, None] * scale_stride + offs_db[None, :]

        v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
        v_ptrs = V + v_offset + offs_n2[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v_scale_ptrs = vscale + offs_n[:, None] * scale_stride + offs_db[None, :]

        # initialize pointer to m and l
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
        # have native e^x support in HW.
        QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
        # Q is loaded once at the beginning and shared by all N blocks.
        q_ptrs_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
        q_s = tl.load(q_scale_ptrs)

        # Here we compute how many full and masked blocks we have.
        padded_block_k = n_extra_tokens != 0
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
        # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
        # In this case we might exceed n_blocks so pick the min.
        masked_blocks = min(masked_blocks, n_blocks)
        n_full_blocks = n_blocks - masked_blocks
        block_min = 0
        block_max = n_blocks * BLOCK_N
        # Compute for full blocks. Here we set causal to false regardless of its actual
        # value because there is no masking. Similarly we do not need padding.
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_s, k_ptrs, k_scale_ptrs, v_ptrs, v_scale_ptrs, stride_kn,
                                   stride_vk, start_m, seqlen_k, seqlen_q, block_min, block_max, BLOCK_M, BLOCK_DMODEL,
                                   BLOCK_N, offs_m, offs_n, PADDED_HEAD, ACTUAL_BLOCK_DMODEL, QK_SCALE)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

        # epilogue
        # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
        l_recip = 1 / l_i[:, None]
        acc = acc * l_recip

        # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
        # then we have one block with a row of all NaNs which come from computing
        # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
        # and store 0s where there are NaNs as these rows should've been zeroed out.
        end_m_idx = (start_m + 1) * BLOCK_M
        acc = acc.to(Out.type.element_ty)

        # write back O
        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        overflow_size = end_m_idx - seqlen_q
        o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
        o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
        if overflow_size > 0:
            o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
        if PADDED_HEAD:
            o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)
