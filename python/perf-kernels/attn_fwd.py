import triton
import triton.language as tl
from autotune_configs import get_autotune_configs
from jit_or_aot_types import (
    constexpr_or_i32,
    constexpr_or_f32,
    u64,
)
from dropout import PHILOX_RN_PER_OFFSET

autotune_configs, autotune_keys = get_autotune_configs()

from _attn_fwd_inner import _attn_fwd_inner

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

# yapf: disable
@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(
        # Basic SDPA
        Q, K, V, B, A, Sm_scale : constexpr_or_f32, L, Out,
        Q_descale, K_descale, P_scale, P_descale, V_descale,
        # Note: do not apply u64 to last stride
        #       which should be hard-coded to 1 otherwise the performance will be -50%
        stride_qz:u64, stride_qh:u64, stride_qm:u64, stride_qk,
        stride_kz:u64, stride_kh:u64, stride_kn:u64, stride_kk,
        stride_vz:u64, stride_vh:u64, stride_vk:u64, stride_vn,
        stride_oz:u64, stride_oh:u64, stride_om:u64, stride_on,
        stride_bz:u64, stride_bh:u64, stride_bm:u64, stride_bn,
        stride_az, stride_ah,  # Let Triton decide its type/constexpr for this small tensor
        # MQA/GQA
        Num_head_q : constexpr_or_i32,
        Num_head_k : constexpr_or_i32,
        # Varlen
        Num_seqlens : constexpr_or_i32,
        cu_seqlens_q,
        cu_seqlens_k,
        Max_seqlen_q : constexpr_or_i32,
        Max_seqlen_k : constexpr_or_i32,
        # Head Dimensions
        BLOCK_DMODEL: tl.constexpr,
        Head_dim : constexpr_or_i32,
        PADDED_HEAD: tl.constexpr,
        # dropout and PRNG
        ENABLE_DROPOUT: tl.constexpr,
        dropout_p,
        philox_seed_ptr,
        philox_offset1,
        philox_offset2 : tl.int32,  # TODO: move to tl.int64
        philox_seed_output, # Should be '*u64', but code-formatter complains
        philox_offset_output, # Should be '*u64', but code-formatter complains
        RETURN_ENCODED_SOFTMAX: tl.constexpr,
        encoded_softmax,
        # causal, (Planned Feature) windowed attention
        CAUSAL_TYPE: tl.constexpr,
        # bias
        BIAS_TYPE: tl.constexpr,
        # alibi
        USE_ALIBI: tl.constexpr,
        # INT8
        INT8: tl.constexpr,
        INT8_KV: tl.constexpr,
        USE_P_SCALE: tl.constexpr,
        # Persistent related arguments
        PERSISTENT_TYPE: tl.constexpr,  # 0: disable, 1: fixed, 2: dynamic
        persistent_atomic_counter,
        Num_CU : constexpr_or_i32,
        GRID_CU_MULTIP: tl.constexpr,
        Batch : constexpr_or_i32,
        # Performance
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PRE_LOAD_V: tl.constexpr,
        ):
    # yapf: enable
    # Adaptor code for AOTriton, to minimize main body code change
    ## tl.constexpr to variable
    IS_CAUSAL: tl.constexpr = CAUSAL_TYPE != 0
    IS_CAUSAL_BOTTOM_RIGHT: tl.constexpr = CAUSAL_TYPE == 2
    USE_BIAS: tl.constexpr = (BIAS_TYPE == 1)
    PERSISTENT : tl.constexpr = (PERSISTENT_TYPE > 0)
    PERSISTENT_DYNAMIC : tl.constexpr = (PERSISTENT_TYPE == 2)
    tl.static_assert(BIAS_TYPE == 0 or BIAS_TYPE == 1, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    L_not_null = L.cast(dtype=tl.uint64, bitcast=True) != 0  # Allows null L for training=False
    INT8_GEMM: tl.constexpr = INT8 and (not INT8_KV)

    ## philox
    idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32) if ENABLE_DROPOUT else 0
    philox_seed = 0
    philox_offset_base = philox_offset2 if ENABLE_DROPOUT else 0
    philox_offset_stride = tl.cdiv(Max_seqlen_k, PHILOX_RN_PER_OFFSET)
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
        if (tl.program_id(0) == 0 and tl.program_id(1) == 0) and tl.program_id(2) == 0:
            if philox_seed_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_seed_output, philox_seed)
            if philox_offset_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_offset_output,
                         philox_offset_base.to(dtype=philox_seed_output.type.element_ty))

    # Default values for standard kernel
    tile_id = 0
    num_tiles_total = 1
    num_tiles_per_head = 1
    num_tiles_per_sample = 1
    Num_WG = Num_CU * GRID_CU_MULTIP  # number of workgroups launched
    unsupported_by_persistent = Num_seqlens != 0
    if PERSISTENT and not unsupported_by_persistent:  # Only enable for non-varlen
        # if persistent, kernel loops over multiple tiles
        num_tiles_per_head = tl.cdiv(Max_seqlen_q, BLOCK_M)  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * Num_head_q  # times the number of heads
        num_tiles_total = num_tiles_per_sample * Batch  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = persistent_atomic_counter.atomic_add(1)  # returns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)

    continue_condition : tl.int1 = True  # as we can't have return statements inside while loop in Triton

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT_DYNAMIC and not unsupported_by_persistent:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head  # at which head are we inside the sample
            start_m = tile_id % num_tiles_per_sample % num_tiles_per_head  # at which tile are we inside the head
        else:
            start_m = tl.program_id(2)
            off_h_q = tl.program_id(1)
            off_z = tl.program_id(0)
        start_M = start_m * BLOCK_M

        offs_m = start_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        if Num_seqlens > 0:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_M >= seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            batch_index = 0
        elif Num_seqlens < 0:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            if start_M >= seqlen_q:
                continue_condition = False
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            # Varlen, but padded to Rank 4 tensor
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            batch_index = off_z
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            seqlen_q = Max_seqlen_q
            seqlen_k = Max_seqlen_k
            batch_index = off_z

        if continue_condition:
            # Now we compute whether we need to exit early due to causal masking.
            # This is because for seqlen_q > seqlen_k, M rows of the attn scores
            # are completely masked, resulting in 0s written to the output, and
            # inf written to LSE. We don't need to do any GEMMs in this case.
            # This block of code determines what N is, and if this WG is operating
            # on those M rows.
            n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
            if IS_CAUSAL:
                # If seqlen_q == seqlen_k, the attn scores are a square matrix.
                # If seqlen_q != seqlen_k, attn scores are rectangular which means

                if CAUSAL_TYPE == 2:
                    # bottom right aligned causal mask, and ends at either
                    # the top edge (seqlen_q < seqlen_k) or left edge.
                    # This captures the decrease in n_blocks if we have a rectangular attn matrix
                    n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
                else:
                    # top left aligned version
                    n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M, BLOCK_N)

                # This is what adjusts the block_max for the current WG, only
                # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
                n_blocks = min(n_blocks, n_blocks_seqlen)
                # If we have no blocks after adjusting for seqlen deltas, this WG is part of
                # the blocks that are all 0. We exit early.
                if n_blocks <= 0:
                    o_offset = Out + batch_index * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to([BLOCK_M, BLOCK_DMODEL])
                    # We still need to write 0s to the result
                    tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                    # The tensor allocated for L is based on Max_seqlen_q as that is
                    # statically known.
                    if L_not_null:
                        l_ptrs = L + off_z * Num_head_q * Max_seqlen_q + off_h_q * Max_seqlen_q + offs_m
                        # We store inf to LSE, not -inf because in the bwd pass, we subtract this
                        # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
                        l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                        l_ptrs_mask = offs_m < Max_seqlen_q
                        tl.store(l_ptrs, l, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be handled here too?
                    continue_condition = False

            if continue_condition:
                # MQA / GQA has different K and V head offsets.
                # Note: using If-then-else may trigger a compiler bug...
                off_h_k = off_h_q // (Num_head_q // Num_head_k)

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N

                # Compute pointers for all the tensors used in this kernel.
                q_offset = Q + batch_index * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
                q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                k_offset = K + batch_index * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
                k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
                v_offset = V + batch_index * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
                v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
                # Compute pointers for all the scale tensors used in this kernel.

                if INT8:
                    k_descale_ptrs = K_descale + off_h_k
                    v_descale_ptrs = V_descale + off_h_k
                    if not INT8_KV:
                        q_descale_ptrs = Q_descale + off_h_q
                    if USE_P_SCALE:
                        p_scale_ptrs = P_scale + off_h_q
                        p_descale_ptrs = P_descale + off_h_q

                if USE_BIAS:
                    # Note: this might get large enough to overflow on some configs
                    bias_offset = B + batch_index * stride_bz + off_h_q * stride_bh
                    bias_ptrs = bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(A + a_offset)
                else:
                    alibi_slope = None

                off_zh = off_z * Num_head_q + off_h_q
                if ENABLE_DROPOUT:
                    batch_philox_offset = philox_offset_base + off_zh * Max_seqlen_q * Max_seqlen_k
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing any dropout. In
                # this case, we return an invalid pointer so indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_zh * Max_seqlen_q * Max_seqlen_k
                else:
                    encoded_sm_base = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], -3.40282e+38, dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                Qk_scale: constexpr_or_f32 = Sm_scale * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                if PADDED_HEAD:
                    q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < Head_dim)
                q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

                if INT8:
                    k_descale = tl.load(k_descale_ptrs)
                    v_descale = tl.load(v_descale_ptrs)
                    if not INT8_KV:
                        q_descale = tl.load(q_descale_ptrs)
                    else:
                        q_descale = None
                    if USE_P_SCALE:
                        p_scale = tl.load(p_scale_ptrs)
                        p_descale = tl.load(p_descale_ptrs)
                    else:
                        p_scale = None
                        p_descale = None
                else:
                    q_descale = None
                    k_descale = None
                    v_descale = None
                    p_scale = None
                    p_descale = None
                # Here we compute how many full and masked blocks we have.
                padded_block_k = n_extra_tokens != 0
                is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
                if IS_CAUSAL:
                    # There are always at least BLOCK_M // BLOCK_N masked blocks.
                    # Additionally there might be one more due to dissimilar seqlens.
                    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
                else:
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
                if n_full_blocks > 0:
                    block_max = (n_blocks - masked_blocks) * BLOCK_N
                    # yapf: disable
                    acc, l_i, m_i = _attn_fwd_inner(
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
                            seqlen_k, seqlen_q, Head_dim,
                            # Dropout
                            dropout_p, philox_seed, batch_philox_offset, Max_seqlen_k,
                            encoded_sm_base,
                            # offs_n_causal, masked_blocks, n_extra_tokens, _
                            0, 0, 0,
                            # Alibi
                            alibi_slope,
                            # INT8
                            q_descale, k_descale, v_descale, p_scale,
                            # constexpr
                            IS_CAUSAL=False,
                            BLOCK_M=BLOCK_M,
                            BLOCK_DMODEL=BLOCK_DMODEL,
                            BLOCK_N=BLOCK_N,
                            OFFS_M=offs_m,
                            OFFS_N=offs_n,
                            PRE_LOAD_V=PRE_LOAD_V,
                            MASK_STEPS=False,
                            ENABLE_DROPOUT=ENABLE_DROPOUT,
                            RETURN_ENCODED_SOFTMAX=RETURN_ENCODED_SOFTMAX,
                            PADDED_HEAD=PADDED_HEAD,
                            INT8_GEMM=INT8_GEMM,
                            INT8_KV=INT8_KV,
                            USE_P_SCALE=USE_P_SCALE,
                            )
                    # yapf: enable
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if (masked_blocks > 0):
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL_BOTTOM_RIGHT else offs_n
                    else:
                        offs_n_causal = 0
                    k_ptrs += n_full_blocks * BLOCK_N * stride_kn
                    v_ptrs += n_full_blocks * BLOCK_N * stride_vk
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    # yapf: disable
                    acc, l_i, m_i = _attn_fwd_inner(
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
                            seqlen_k, seqlen_q, Head_dim,
                            # Dropout
                            dropout_p, philox_seed, batch_philox_offset, Max_seqlen_k,
                            encoded_sm_base,
                            # CAUSAL: offs_n_causal, masked_blocks, n_extra_tokens, _
                            offs_n_causal, masked_blocks, n_extra_tokens,
                            # Alibi
                            alibi_slope,
                            # INT8
                            q_descale, k_descale, v_descale, p_scale,
                            # constexpr
                            IS_CAUSAL=IS_CAUSAL,
                            BLOCK_M=BLOCK_M,
                            BLOCK_DMODEL=BLOCK_DMODEL,
                            BLOCK_N=BLOCK_N,
                            OFFS_M=offs_m,
                            OFFS_N=offs_n,
                            PRE_LOAD_V=PRE_LOAD_V,
                            MASK_STEPS=True,
                            ENABLE_DROPOUT=ENABLE_DROPOUT,
                            RETURN_ENCODED_SOFTMAX=RETURN_ENCODED_SOFTMAX,
                            PADDED_HEAD=PADDED_HEAD,
                            INT8_GEMM=INT8_GEMM,
                            INT8_KV=INT8_KV,
                            USE_P_SCALE=USE_P_SCALE,
                            )
                    # yapf: enable

                if INT8 and not INT8_KV:
                    if USE_P_SCALE:
                        acc *= p_descale
                    acc *= v_descale

                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
                l_recip = 1 / l_i[:, None]
                if ENABLE_DROPOUT:  # Should make dropout faster?
                    l_recip *= 1.0 / (1 - dropout_p)
                acc = acc * l_recip
                # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
                # then we have one block with a row of all NaNs which come from computing
                # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
                # and store 0s where there are NaNs as these rows should've been zeroed out.
                end_M = start_M + BLOCK_M
                causal_start_idx = seqlen_q - seqlen_k if IS_CAUSAL_BOTTOM_RIGHT else 0
                acc = acc.to(Out.type.element_ty)
                if IS_CAUSAL:
                    if causal_start_idx > start_M and causal_start_idx < end_M:
                        out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
                        mask_m_offsets = start_M + tl.arange(0, BLOCK_M)
                        out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
                        z = 0.0
                        acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

                overflow_size = end_M - seqlen_q
                if L_not_null:
                    # write back LSE
                    l_ptrs = L + off_z * Num_head_q * Max_seqlen_q + off_h_q * Max_seqlen_q + offs_m
                    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
                    # This is only true for the last M block. For others, overflow_size will be -ve
                    if overflow_size > 0:
                        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                    else:
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                # write back O
                o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                if PADDED_HEAD:
                    o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < Head_dim)
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if not PERSISTENT or unsupported_by_persistent:
            tile_id = num_tiles_total  # break after single tile
        else:
            if PERSISTENT_DYNAMIC:
                tile_id = persistent_atomic_counter.atomic_add(1)
            else:
                tile_id += Num_WG
