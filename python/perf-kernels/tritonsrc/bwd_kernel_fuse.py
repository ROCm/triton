#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import triton
import triton.language as tl
from bwd_inner_fuse import bwd_inner_dk_dv_fuse
from bwd_inner_dq import bwd_inner_dq
from dropout import PHILOX_RN_PER_OFFSET
from masked_load_store import (
    load_fn,
    mstore2d,
    is_closed_interval_empty,
    parse_window,
    calculate_intervals,
    closed_interval_size,
)
from composed_tensors import (
    composed_offs_1d,
    composed_zeros_2d,
    composed_ptrs,
    composed_load,
    composed_advance,
    composed_to,
    composed_store,
    composed_mul_lhs,
    composed_dot_both,
    composed_dot_rhs,
    composed_mul_lhs,
    composed_mul_acc,
    composed_inner_product_fp32,
)

@triton.jit
def bwd_kernel_fuse(
    # I/O tensors
    Q, K, V, B, sm_scale,
    Out, DO,
    DK, DV, DQ, DB,
    L,
    # Strides
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    # Problem size
    num_head_q: tl.constexpr,
    num_head_k: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens: tl.constexpr,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    head_dim: tl.constexpr,
    # Dropout
    dropout_p : tl.float32,
    philox_seed_ptr,
    philox_offset1 : '*u64',
    philox_offset2 : 'u64',
    # Windowed Attention
    Window_left : 'i32',
    Window_right : 'i32',
    # Constants
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_DMODEL > 0, 'BLOCK_DMODEL must be greater than 0')
    BLOCK_DMODEL_R0 : tl.constexpr = BLOCK_DMODEL
    BLOCK_DMODEL0 : tl.constexpr = 2 ** (BLOCK_DMODEL_R0.bit_length() - 1)
    BLOCK_DMODEL_R1 : tl.constexpr = BLOCK_DMODEL_R0 - BLOCK_DMODEL0
    BLOCK_DMODEL1 : tl.constexpr = 2 ** (BLOCK_DMODEL_R1.bit_length() - 1) if BLOCK_DMODEL_R1 > 0 else 0
    BLOCK_DMODEL_R2 : tl.constexpr = BLOCK_DMODEL_R1 - BLOCK_DMODEL1
    BLOCK_DMODEL2 : tl.constexpr = 2 ** (BLOCK_DMODEL_R2.bit_length() - 1) if BLOCK_DMODEL_R2 > 0 else 0
    BLOCK_DMODEL_R3 : tl.constexpr = BLOCK_DMODEL_R2 - BLOCK_DMODEL2

    tl.static_assert(BLOCK_DMODEL_R3 == 0, f'BLOCK_DMODEL = {BLOCK_DMODEL} = 0b{BLOCK_DMODEL:b} cannot be factored into <= 3 power of two values')
    tl.static_assert(BLOCK_DMODEL1 > 0 or BLOCK_DMODEL2 == 0, 'Only trailing BLOCK_DMODELx can be 0')
    pid = tl.program_id(2)
    NUM_KV_BLOCKS = tl.cdiv(max_seqlen_k, BLOCK_N)
    NUM_Q_BLOCKS = tl.cdiv(max_seqlen_q, BLOCK_N)
    IS_CAUSAL : tl.constexpr = CAUSAL_TYPE != 0

    if pid >= NUM_KV_BLOCKS:
        # dq compute block
        off_pid = pid - NUM_KV_BLOCKS
        start_q = (off_pid % NUM_Q_BLOCKS) * BLOCK_N
        off_h_k = tl.program_id(1) # kv head index
        group_size = num_head_q // num_head_k
        off_h_q = (off_pid // NUM_Q_BLOCKS) + off_h_k * group_size # q head index

        off_z = tl.program_id(0) # batch index, for varlen it indicates index in cu_seqlens_q/k
        off_zh = off_z * num_head_q + off_h_q * 1
        offs_q_dq = start_q + tl.arange(0, BLOCK_N)
        offs_k_dq = tl.arange(0, BLOCK_M)

        philox_seed = 0
        philox_offset_base = philox_offset2
        philox_offset_stride = tl.cdiv(max_seqlen_k, PHILOX_RN_PER_OFFSET)
        if ENABLE_DROPOUT:
            philox_seed = tl.load(philox_seed_ptr)
            philox_offset_base += tl.load(philox_offset1)
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z

        if num_seqlens > 0:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            if start_q >= seqlen_q:
                return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            batch_index = 0

        if num_seqlens < 0:  # for padded seqlen
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            if start_q >= seqlen_q:
                return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            # Varlen, but padded to Rank 4 tensor
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            batch_index = off_z

        qk_scale = sm_scale * 1.44269504089
        bias_scale = 1.0 / sm_scale
        if num_seqlens > 0:
            if start_q >= seqlen_q:
                return
        if num_seqlens < 0:  # for padded seqlen
            if start_q >= seqlen_q:
                return
        q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(Q,
                                                  stride_qz, stride_qh, stride_qm, stride_qk,
                                                  batch_index, off_h_q, cu_seqlens_q_start + offs_q_dq,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if start_q + BLOCK_N <= seqlen_q:
            q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                       offs_q_dq,
                                       BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                       seqlen_q, head_dim,
                                       other=0.0,
                                       PADDED_ROW=False,
                                       PADDED_COL=PADDED_HEAD,
                                       TRANSPOSED=False)
        else:
            q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                       offs_q_dq,
                                       BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                       seqlen_q, head_dim,
                                       other=0.0,
                                       PADDED_ROW=True,
                                       PADDED_COL=PADDED_HEAD,
                                       TRANSPOSED=False)

        kt_ptrs0, kt_ptrs1, kt_ptrs2 = composed_ptrs(K,
                                                     stride_kz, stride_kh, stride_kn, stride_kk,
                                                     batch_index, off_h_k, cu_seqlens_k_start + offs_k_dq,
                                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                     TRANSPOSED=True)
        vt_ptrs0, vt_ptrs1, vt_ptrs2 = composed_ptrs(V,
                                                     stride_vz, stride_vh, stride_vk, stride_vn,
                                                     batch_index, off_h_k, cu_seqlens_k_start + offs_k_dq,
                                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                     TRANSPOSED=True)

        do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(DO,
                                                     stride_doz, stride_doh, stride_dom, stride_dok,
                                                     batch_index, off_h_q, cu_seqlens_q_start + offs_q_dq,
                                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        o_ptrs0, o_ptrs1, o_ptrs2 = composed_ptrs(Out,
                                                  stride_oz, stride_oh, stride_om, stride_ok,
                                                  batch_index, off_h_q, cu_seqlens_q_start + offs_q_dq,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

        if start_q + BLOCK_N <= seqlen_q:
            do0, do1, do2 = composed_load(do_ptrs0, do_ptrs1, do_ptrs2,
                                          offs_q_dq,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_q, head_dim,
                                          other=0.0,
                                          PADDED_ROW=False,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=False)
            o0, o1, o2 = composed_load(o_ptrs0, o_ptrs1, o_ptrs2,
                                       offs_q_dq,
                                       BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                       seqlen_q, head_dim,
                                       other=0.0,
                                       PADDED_ROW=False,
                                       PADDED_COL=PADDED_HEAD,
                                       TRANSPOSED=False)
        else:
            do0, do1, do2 = composed_load(do_ptrs0, do_ptrs1, do_ptrs2,
                                          offs_q_dq,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_q, head_dim,
                                          other=0.0,
                                          PADDED_ROW=True,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=False)
            o0, o1, o2 = composed_load(o_ptrs0, o_ptrs1, o_ptrs2,
                                       offs_q_dq,
                                       BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                       seqlen_q, head_dim,
                                       other=0.0,
                                       PADDED_ROW=True,
                                       PADDED_COL=PADDED_HEAD,
                                       TRANSPOSED=False)
        # pointer to row-wise quantities in value-like data
        l_ptrs = L + off_zh * max_seqlen_q
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * philox_offset_stride
        else:
            batch_philox_offset = 0

        # initialize pointers to output
        dq_offset = batch_index * stride_dqz + off_h_q * stride_dqh + cu_seqlens_q_start * stride_dqm
        DQ += dq_offset
        store_db = True
        if BIAS_TYPE == 0:
            B_ptr_dq = 0
            DB_ptr = 0
        elif BIAS_TYPE == 1:
            B_ptr_dq = B + off_h_q * stride_bh + batch_index * stride_bz
            if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
                store_db = False
            DB_ptr = DB + off_h_q * stride_dbh + batch_index * stride_dbz
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

        '''
        For Fused kernel BLOCK_N is used for q direction on dq branch
        '''
        window_left, window_right = parse_window(IS_CAUSAL,
                                                 CAUSAL_TYPE,
                                                 Window_left,
                                                 Window_right,
                                                 seqlen_q,
                                                 seqlen_k)
        mask_on_seq_q = (start_q + BLOCK_N > seqlen_q)
        lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = \
                calculate_intervals(IS_CAUSAL,
                                    CAUSAL_TYPE,
                                    window_left,
                                    window_right,
                                    start_q,
                                    seqlen_q,
                                    seqlen_k,
                                    mask_on_seq_q,
                                    BLOCK_N,
                                    BLOCK_M)
        lb_empty = is_closed_interval_empty(lb_lo, lb_hi)
        rb_empty = is_closed_interval_empty(rb_lo, rb_hi)
        fb_empty = is_closed_interval_empty(fb_lo, fb_hi)

        # Check for OOB accesses on D and LSE
        d_lse_ptrs_mask = offs_q_dq < seqlen_q
        Di = composed_inner_product_fp32(o0, o1, o2,
                                         do0, do1, do2,
                                         BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                         axis=1)
        l_i = tl.load(l_ptrs + offs_q_dq, mask=d_lse_ptrs_mask, other=0.0)

        idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32) if ENABLE_DROPOUT else 0
        dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
        dq0, dq1, dq2 = composed_zeros_2d(BLOCK_N, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

        if not fb_empty:
            nblocks_1 = closed_interval_size(fb_lo, fb_hi)
            dq0, dq1, dq2 = bwd_inner_dq(
                dq0, dq1, dq2,
                qk_scale, bias_scale,
                DB_ptr, store_db,
                q0, q1, q2,
                kt_ptrs0, kt_ptrs1, kt_ptrs2,
                stride_kn,
                vt_ptrs0, vt_ptrs1, vt_ptrs2,
                stride_vk,
                B_ptr_dq,
                stride_bn, stride_bm,  stride_dbn, stride_dbm,
                do0, do1, do2,
                Di, l_i,
                seqlen_q, seqlen_k, head_dim,
                start_q, nblocks_1, 0, fb_lo, None,
                idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                window_left,
                window_right,
                BLOCK_N,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_M,
                True,  # FULL_BLOCKS
                False,  # CAUSAL has zero effect for full blocks
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)

        if not (lb_empty and rb_empty):
            tl.debug_barrier()
            nblocks_1 = closed_interval_size(lb_lo, lb_hi)
            nblocks_2 = closed_interval_size(rb_lo, rb_hi)
            dq0, dq1, dq2 = bwd_inner_dq(
                dq0, dq1, dq2,
                qk_scale, bias_scale,
                DB_ptr, store_db,
                q0, q1, q2,
                kt_ptrs0, kt_ptrs1, kt_ptrs2,
                stride_kn,
                vt_ptrs0, vt_ptrs1, vt_ptrs2,
                stride_vk,
                B_ptr_dq,
                stride_bn, stride_bm,  stride_dbn, stride_dbm,
                do0, do1, do2,
                Di, l_i,
                seqlen_q, seqlen_k, head_dim,
                start_q, nblocks_1, nblocks_2, lb_lo, rb_lo,
                idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                window_left,
                window_right,
                BLOCK_N,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_M,
                False,  # FULL_BLOCKS
                IS_CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)
        dq0, dq1, dq2 = composed_mul_lhs(dq0, dq1, dq2,
                                        sm_scale,
                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        dq0, dq1, dq2 = composed_to(dq0, dq1, dq2, dq0.type.element_ty)
        composed_store(dq0, dq1, dq2,
                       BLOCK_N,
                       BLOCK_DMODEL0,
                       BLOCK_DMODEL1,
                       BLOCK_DMODEL2,
                       o_base=DQ,
                       o_start_row=start_q,
                       o_start_col=0,
                       o_rows=seqlen_q,
                       o_cols=head_dim,
                       stride_row=stride_dqm,
                       stride_col=stride_dqk)
    else:
        idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32) if ENABLE_DROPOUT else 0
        philox_seed = 0
        philox_offset_base = philox_offset2
        philox_offset_stride = tl.cdiv(max_seqlen_k, PHILOX_RN_PER_OFFSET)
        if ENABLE_DROPOUT:
            philox_seed = tl.load(philox_seed_ptr)
            philox_offset_base += tl.load(philox_offset1)
        start_k = tl.program_id(2) * BLOCK_N  # start_k partitions seqlen_k
        off_h_k = tl.program_id(1) # head index
        off_z = tl.program_id(0) # batch index, for varlen it indicates index in cu_seqlens_q/k
        offs_q = tl.arange(0, BLOCK_M)
        offs_k = start_k + tl.arange(0, BLOCK_N)
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z

        if num_seqlens > 0:
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            if start_k >= seqlen_k:
                return
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            batch_index = 0

        if num_seqlens < 0:  # for padded seqlen
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            if start_k >= seqlen_k:
                return
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # Varlen, but padded to Rank 4 tensor
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            batch_index = off_z

        k_ptrs0, k_ptrs1, k_ptrs2 = composed_ptrs(K,
                                                  stride_kz, stride_kh, stride_kn, stride_kk,
                                                  batch_index, off_h_k, cu_seqlens_k_start + offs_k,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                  TRANSPOSED=True)
        # kt_offs_k = None if start_k + BLOCK_N <= seqlen_k else start_k + tl.arange(0, BLOCK_N)
        v_ptrs0, v_ptrs1, v_ptrs2 = composed_ptrs(V,
                                                  stride_vz, stride_vh, stride_vk, stride_vn,
                                                  batch_index, off_h_k, cu_seqlens_k_start + offs_k,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                  TRANSPOSED=True)

        if start_k + BLOCK_N <= seqlen_k:
            kt0, kt1, kt2 = composed_load(k_ptrs0, k_ptrs1, k_ptrs2,
                                          offs_k,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_k, head_dim,
                                          other=0.0,
                                          PADDED_ROW=False,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=True)
            vt0, vt1, vt2 = composed_load(v_ptrs0, v_ptrs1, v_ptrs2,
                                          offs_k,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_k, head_dim,
                                          other=0.0,
                                          PADDED_ROW=False,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=True)
        else:
            kt0, kt1, kt2 = composed_load(k_ptrs0, k_ptrs1, k_ptrs2,
                                          offs_k,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_k, head_dim,
                                          other=0.0,
                                          PADDED_ROW=True,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=True)
            vt0, vt1, vt2 = composed_load(v_ptrs0, v_ptrs1, v_ptrs2,
                                          offs_k,
                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                          seqlen_k, head_dim,
                                          other=0.0,
                                          PADDED_ROW=True,
                                          PADDED_COL=PADDED_HEAD,
                                          TRANSPOSED=True)
        if BIAS_TYPE == 0:
            B_ptr = 0
        elif BIAS_TYPE == 1:
            # CAVEAT: bias is incompatible with GQA
            # B_block_ptr = tl.make_block_ptr(
            #         base=B + off_h_k * stride_bh + batch_index * stride_bz,
            #         shape=(seqlen_q, seqlen_k),
            #         strides=(stride_bm, stride_bn),
            #         offsets=(0, start_k),
            #         block_shape=(BLOCK_M, BLOCK_N),
            #         order=(1, 0)
            #         )
            B_ptr = B + off_h_k * stride_bh + batch_index * stride_bz
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

        dk_offset = off_h_k * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
        DK += dk_offset
        dv_offset = off_h_k * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
        DV += dv_offset

        dv0, dv1, dv2 = composed_zeros_2d(BLOCK_N, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        dk0, dk1, dk2 = composed_zeros_2d(BLOCK_N, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        qk_scale = sm_scale * 1.44269504089
        bias_scale = 1.0 / sm_scale
        group_size = num_head_q // num_head_k

        window_left, window_right = parse_window(IS_CAUSAL,
                                                 CAUSAL_TYPE,
                                                 Window_left,
                                                 Window_right,
                                                 seqlen_q,
                                                 seqlen_k)
        mask_on_seq_k = (start_k + BLOCK_N > seqlen_k)
        lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = \
                calculate_intervals(IS_CAUSAL,
                                    CAUSAL_TYPE,
                                    window_right,
                                    window_left,
                                    start_k,
                                    seqlen_k,
                                    seqlen_q,
                                    mask_on_seq_k,
                                    BLOCK_N,
                                    BLOCK_M,
                                    DEBUG=False)
        lb_empty = is_closed_interval_empty(lb_lo, lb_hi)
        rb_empty = is_closed_interval_empty(rb_lo, rb_hi)
        fb_empty = is_closed_interval_empty(fb_lo, fb_hi)

        dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
        for off_h_q in range(off_h_k * group_size, off_h_k * group_size + group_size):
            off_zh = off_z * num_head_q + off_h_q * 1
            # This lower loop bound is because of the causal mask. We create a lower triangular
            # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
            # be ignored in the GEMM.
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * philox_offset_stride
            else:
                batch_philox_offset = 0
            # pointer to row-wise quantities in value-like data
            # Shape (batch, num_heads, max_seqlen_q)
            # In varlen cases, batch == len(cu_seqlens_q) - 1).
            # Hence off_z plays the same role in varlen/non-varlen
            l_ptrs = L + off_zh * max_seqlen_q

            q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(Q,
                                                      stride_qz, stride_qh, stride_qm, stride_qk,
                                                      batch_index, off_h_q, cu_seqlens_q_start + offs_q,
                                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

            do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(DO,
                                                         stride_oz, stride_oh, stride_om, stride_ok,
                                                         batch_index, off_h_q, cu_seqlens_q_start + offs_q,
                                                         BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
            o_ptrs0, o_ptrs1, o_ptrs2 = composed_ptrs(Out,
                                                      stride_oz, stride_oh, stride_om, stride_ok,
                                                      batch_index, off_h_q, cu_seqlens_q_start + offs_q,
                                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

            if not fb_empty:
                nblocks_1 = closed_interval_size(fb_lo, fb_hi)
                dk0, dk1, dk2, dv0, dv1, dv2 = bwd_inner_dk_dv_fuse(
                    dk0, dk1, dk2,
                    dv0, dv1, dv2,
                    qk_scale, bias_scale,
                    q_ptrs0, q_ptrs1, q_ptrs2,
                    stride_qm,
                    kt0, kt1, kt2, vt0, vt1, vt2,
                    B_ptr, stride_bm, stride_bn,
                    do_ptrs0, do_ptrs1, do_ptrs2,
                    stride_dom,
                    o_ptrs0, o_ptrs1, o_ptrs2,
                    stride_om,
                    l_ptrs,
                    seqlen_q, seqlen_k, head_dim,
                    start_k, nblocks_1, 0, fb_lo, None,
                    idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                    window_left,
                    window_right,
                    BLOCK_M,
                    BLOCK_DMODEL0,
                    BLOCK_DMODEL1,
                    BLOCK_DMODEL2,
                    BLOCK_N,
                    True,  # FULL_BLOCKS
                    False,  # CAUSAL has zero effect for full blocks
                    ENABLE_DROPOUT,
                    PADDED_HEAD,
                    BIAS_TYPE)

            if not (lb_empty and rb_empty):
                tl.debug_barrier()
                nblocks_1 = closed_interval_size(lb_lo, lb_hi)
                nblocks_2 = closed_interval_size(rb_lo, rb_hi)
                dk0, dk1, dk2, dv0, dv1, dv2 = bwd_inner_dk_dv_fuse(
                    dk0, dk1, dk2,
                    dv0, dv1, dv2,
                    qk_scale, bias_scale,
                    q_ptrs0, q_ptrs1, q_ptrs2,
                    stride_qm,
                    kt0, kt1, kt2, vt0, vt1, vt2,
                    B_ptr, stride_bm, stride_bn,
                    do_ptrs0, do_ptrs1, do_ptrs2,
                    stride_dom,
                    o_ptrs0, o_ptrs1, o_ptrs2,
                    stride_om,
                    l_ptrs,
                    seqlen_q, seqlen_k, head_dim,
                    start_k, nblocks_1, nblocks_2, lb_lo, rb_lo,
                    idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                    window_left,
                    window_right,
                    BLOCK_M,
                    BLOCK_DMODEL0,
                    BLOCK_DMODEL1,
                    BLOCK_DMODEL2,
                    BLOCK_N,
                    False,  # FULL_BLOCKS
                    IS_CAUSAL,
                    ENABLE_DROPOUT,
                    PADDED_HEAD,
                    BIAS_TYPE)

        dk0, dk1, dk2 = composed_mul_lhs(dk0, dk1, dk2,
                                        sm_scale,
                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        dk0, dk1, dk2 = composed_to(dk0, dk1, dk2, kt0.type.element_ty)
        dv0, dv1, dv2 = composed_to(dv0, dv1, dv2, vt0.type.element_ty)

        composed_store(dk0, dk1, dk2,
                       BLOCK_N,
                       BLOCK_DMODEL0,
                       BLOCK_DMODEL1,
                       BLOCK_DMODEL2,
                       o_base=DK,
                       o_start_row=start_k,
                       o_start_col=0,
                       o_rows=seqlen_k,
                       o_cols=head_dim,
                       stride_row=stride_dkn,
                       stride_col=stride_dkk)

        composed_store(dv0, dv1, dv2,
                       BLOCK_N,
                       BLOCK_DMODEL0,
                       BLOCK_DMODEL1,
                       BLOCK_DMODEL2,
                       o_base=DV,
                       o_start_row=start_k,
                       o_start_col=0,
                       o_rows=seqlen_k,
                       o_cols=head_dim,
                       stride_row=stride_dvk,
                       stride_col=stride_dvn)


