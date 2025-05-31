#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import fast_dropout_mask
from masked_load_store import load_fn
from triton.language.extra import libdevice
from composed_tensors import (
    composed_offs_1d,
    composed_load_with_offset,
    composed_load,
    composed_dot_both,
    composed_dot_rhs,
    composed_mul_lhs,
    composed_mul_acc,
    composed_inner_product_fp32,
)

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)

@triton.jit
def bwd_inner_dk_dv_fuse(
    # I/O Tensor
    dk0, dk1, dk2,
    dv0, dv1, dv2,
    qk_scale, bias_scale,
    # Problem Description
    q_ptrs0, q_ptrs1, q_ptrs2,
    q_stride,
    kt0, kt1, kt2, vt0, vt1, vt2,
    B_ptr, stride_bm, stride_bn,
    do_ptrs0, do_ptrs1, do_ptrs2,
    do_stride,
    o_ptrs0, o_ptrs1, o_ptrs2,
    o_stride,
    l_ptrs,
    seqlen_q, seqlen_k, head_dim,
    # Sub-problem range, (lo, hi) specify the range for seqlen_q
    start_k, nblocks_1, nblocks_2, Block_range_1, Block_range_2,
    ## Dropout
    idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
    ## Sliding Window Attention
    window_left,
    window_right,
    # constexpr starts here
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL0,
    BLOCK_DMODEL1,
    BLOCK_DMODEL2,
    BLOCK_N: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # initialize offsets
    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)

    # q_ptrs0, q_ptrs1, q_ptrs2 = composed_advance(q_ptrs0, q_ptrs1, q_ptrs2,
    #                                              lo * q_stride,
    #                                              BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    # do_ptrs0, do_ptrs1, do_ptrs2 = composed_advance(do_ptrs0, do_ptrs1, do_ptrs2,
    #                                                 lo * do_stride,
    #                                                 BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    # o_ptrs0, o_ptrs1, o_ptrs2 = composed_advance(o_ptrs0, o_ptrs1, o_ptrs2,
    #                                              lo * o_stride,
    #                                              BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)


    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_k: select k and dV
    start_q: select q and dO
    '''
    # for start_q in range(lo, hi, BLOCK_M):
    for block_index in range(nblocks_1+nblocks_2):
        # Seccond Range is invalid (constexpr "None" defined in Full block path)
        if Block_range_2 is None:
            start_qi = block_index + Block_range_1
        else:
            start_qi = block_index + Block_range_1 if block_index < nblocks_1 else (block_index - nblocks_1 + Block_range_2)
        start_q = start_qi * BLOCK_M
        # TODO: Unify the name, the usage of m/n is very confusing
        offs_q_curr = offs_q[:, None] + start_q # (BLOCK_M, 1)
        # -- load q, do --
        if FULL_BLOCKS and not PADDED_HEAD:
            q_offs_m = None
        else:
            q_offs_m = start_q + tl.arange(0, BLOCK_M)

        PADDED_SEQ : tl.constexpr = not FULL_BLOCKS
        q0, q1, q2 = composed_load_with_offset(q_ptrs0, q_ptrs1, q_ptrs2,
                                               start_q, q_stride, offs_q,
                                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                               seqlen_q, head_dim,
                                               other=0.0,
                                               PADDED_ROW=PADDED_SEQ,
                                               PADDED_COL=PADDED_HEAD,
                                               TRANSPOSED=False)
        # do = tl.load(DO_block_ptr)
        # TODO: pre_load_do
        do0, do1, do2 = composed_load_with_offset(do_ptrs0, do_ptrs1, do_ptrs2,
                                                  start_q, do_stride, offs_q,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                  seqlen_q, head_dim,
                                                  other=0.0,
                                                  PADDED_ROW=PADDED_SEQ,
                                                  PADDED_COL=PADDED_HEAD,
                                                  TRANSPOSED=False)
        o0, o1, o2 = composed_load_with_offset(o_ptrs0, o_ptrs1, o_ptrs2,
                                               start_q, o_stride, offs_q,
                                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                               seqlen_q, head_dim,
                                               other=0.0,
                                               PADDED_ROW=PADDED_SEQ,
                                               PADDED_COL=PADDED_HEAD,
                                               TRANSPOSED=False)

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # TODO: These two checks can be optimized to occur on the last iter.
        # -- qk mask ----
        if not FULL_BLOCKS or IS_CAUSAL:
            mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
            MS = offs_q + start_q
            NS = offs_k
            if not FULL_BLOCKS: # Potentially for all blocks in the loop
                q_mask = (MS[:, None] < seqlen_q)
                mask = mask & q_mask
            # isect with n = seqlen_k
            if start_k + BLOCK_N > seqlen_k: # Only one block at most
                k_mask = (NS[None, :] < seqlen_k)
                mask = mask & k_mask
            # isect with both windowed causal lines
            if IS_CAUSAL:
                right_mask = MS[:, None] + window_right >= NS[None, :]
                mask = mask & right_mask
                left_mask = MS[:, None] - window_left <= NS[None, :]
                mask = mask & left_mask
            # tl.device_print('mask', mask)
            qk = tl.where(mask, qk, float("-inf"))

        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: do boundary_check correctly
            bias_ptrs = B_ptr + offs_q_curr * stride_bm + offs_k[None, :] * stride_bn
            if not FULL_BLOCKS:
                mask = (offs_q_curr < seqlen_q) & (offs_k < seqlen_k)[None, :]
                bias = tl.load(bias_ptrs, mask=mask, other=0.0)
                # tl.device_print('mask', mask)
            else:
                bias = tl.load(bias_ptrs)
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        # q.offs = (start_q, 0), k.offs = (0, start_k)
        qk = composed_dot_both(q0, q1, q2,
                               kt0, kt1, kt2,
                               qk,
                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        # Check for OOB accesses on D and LSE
        if FULL_BLOCKS:
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            d_lse_ptrs_mask = offs_q_curr < seqlen_q
            l_i = tl.load(l_ptrs + offs_q_curr,
                          mask=d_lse_ptrs_mask,
                          other=0.0)
        Di = composed_inner_product_fp32(o0, o1, o2,
                                         do0, do1, do2,
                                         BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                         axis=1)
        # FIXME: Potential bug https://github.com/ROCm/aotriton/issues/54
        p = tl.math.exp2(qk_scale * qk - l_i) # (BLOCK_M, BLOCK_N)

        if not FULL_BLOCKS or IS_CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            # philox_offset = batch_philox_offset + start_q * philox_offset_stride + start_k
            keep = fast_dropout_mask(philox_seed, idropout_p,
                                     batch_philox_offset,
                                     start_q, start_k,
                                     BLOCK_M, BLOCK_N, philox_offset_stride)
            # CAVEAT: do NOT update p, ds needs the original p
            p_dropped = tl.where(keep, p * dropout_scale, 0.0)
        else:
            p_dropped = p

        if BLOCK_M == 1:
            # dv += p.to(q_ptrs.dtype.element_ty) * do
            dv0, dv1, dv2 = composed_mul_acc(do0, do1, do2,
                                             p_dropped.to(do0.dtype),
                                             dv0, dv1, dv2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        else:
            # dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dv0, dv1, dv2 = composed_dot_rhs(p_dropped,
                                             do0, do1, do2,
                                             dv0, dv1, dv2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                             TRANSPOSE_LHS=True)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # compute dp = dot(do, vt)
        # dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        # do.shape = (BLOCK_M, BLOCK_DMODEL) vt.shape = (BLOCK_DMODEL, BLOCK_N)
        dp += composed_dot_both(do0, do1, do2,
                                vt0, vt1, vt2,
                                dp,
                                BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp * dropout_scale, 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None]) # (BLOCK_M, BLOCK_N)
        # compute dk
        if BLOCK_M == 1:
            dk0, dk1, dk2 = composed_mul_acc(q0, q1, q2,
                                             ds.to(q_ptrs0.dtype.element_ty),
                                             dk0, dk1, dk2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

        else:
            # ds.shape = (BLOCK_M, BLOCK_N), q.shape = (BLOCK_M, BLOCK_DMODEL)
            dk0, dk1, dk2 = composed_dot_rhs(ds,
                                             q0, q1, q2,
                                             dk0, dk1, dk2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                             TRANSPOSE_LHS=True)
    return dk0, dk1, dk2, dv0, dv1, dv2

