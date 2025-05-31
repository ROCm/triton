#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import fast_philox, fast_dropout_mask, PHILOX_RN_PER_OFFSET
from masked_load_store import mstore2d

@triton.jit
def debug_fill_dropout_rng(R,
                           stride_rz, stride_rh, stride_rm, stride_rn,
                           seqlen_q, seqlen_k,
                           philox_seed : tl.uint64,
                           philox_offset_base : tl.uint64,
                           BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           ):
    start_m = tl.program_id(2)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(0) # batch index
    d_offset = off_h * stride_rh + off_z * stride_rz
    num_h = tl.num_programs(1)
    off_zh = off_z * num_h + off_h * 1
    philox_offset_stride = tl.cdiv(seqlen_k, PHILOX_RN_PER_OFFSET)
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * philox_offset_stride
    R_block_ptr = tl.make_block_ptr(
        base=R + d_offset,
        shape=(seqlen_q, seqlen_k),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    # Minimal SCALE that SCALE * MAX_I32 > 0.5. Tested with
    #   >>> scale = np.array([2.32830657e-10], dtype=np.float32); (scale * max_i32) > 0.5
    #   array([False])
    #   >>> scale = np.array([2.32830658e-10], dtype=np.float32); (scale * max_i32) > 0.5
    #   array([ True])
    #   >>> (np.array([2.328306571e-10], dtype=np.float32) - np.array([2.32830657e-10], dtype=np.float32)) == 0.0
    #   array([ True])  # The final `1` exceeds the precision limit of fp32
    #   >>> (np.array([2.32830658e-10], dtype=np.float32) - np.array([2.328306579e-10], dtype=np.float32)) == 0.0
    #   array([ True])  # The final digits exceeds the precision limit of fp32
    # Note1: Do not use tl.uint_to_uniform_float, it translates max_u32 to 0.0
    # Note2: Do not pick SCALE * MAX_I32 < 0.5 (== is not possible due to fp
    #        errors), because it cannot generate 1.0 otherwise
    SCALE : tl.constexpr = 2.32830658e-10
    for start_n in range(0, seqlen_k, BLOCK_N):
        philox_offset = batch_philox_offset + start_m * BLOCK_M * philox_offset_stride + start_n // PHILOX_RN_PER_OFFSET
        rng = fast_philox(philox_seed, philox_offset, BLOCK_M, BLOCK_N // PHILOX_RN_PER_OFFSET, philox_offset_stride)
        if R.type.element_ty == tl.float32:
            # Attept to translate [-MAX_I32-1, MAX_I32] to [0.0, 1.0]
            # 0 should be translated to 0.0
            rng = (rng * SCALE + 0.5).to(tl.float32)
            tl.clamp(rng, 0.0, 1.0) # In case |rng| >= MAX_I32
            tl.store(R_block_ptr,
                     rng,
                     boundary_check=(0,1))
        elif R.type.element_ty == tl.int32:
            tl.store(R_block_ptr, rng, boundary_check=(0,1))
        else:
            tl.static_assert(False, 'R.type.element_ty must be either float32 or int32')
        R_block_ptr = tl.advance(R_block_ptr, (0, BLOCK_N))

@triton.jit
def debug_fill_dropout_rng_tensor(R,
                                  stride_rz, stride_rh, stride_rm, stride_rn,
                                  seqlen_q, seqlen_k,
                                  philox_seed_ptr: '*u64',
                                  philox_offset_base_ptr : '*u64',
                                  BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr,
                                  ):
    philox_seed = tl.load(philox_seed_ptr)
    philox_offset_base = tl.load(philox_offset_base_ptr)
    debug_fill_dropout_rng(R,
                           stride_rz, stride_rh, stride_rm, stride_rn,
                           seqlen_q, seqlen_k,
                           philox_seed,
                           philox_offset_base,
                           BLOCK_M,
                           BLOCK_N,
                           )

@triton.jit
def debug_simulate_encoded_softmax(R,
                                   stride_rz, stride_rh, stride_rm, stride_rn,
                                   dropout_p, Num_head_q, Max_seqlen_q, Max_seqlen_k,
                                   philox_seed_ptr : '*u64',
                                   philox_offset1 : '*u64',
                                   philox_offset2 : tl.uint64,  # TODO: move to tl.int64
                                   BLOCK_M: tl.constexpr,
                                   BLOCK_N: tl.constexpr,
                                   ):
    # philox_seed = 0 sets philox_seed's dtype to i32
    philox_seed = philox_seed_ptr.cast(dtype=tl.uint64, bitcast=True)
    if philox_seed_ptr.cast(dtype=tl.uint64, bitcast=True) != 0:
        philox_seed = tl.load(philox_seed_ptr)
    philox_offset_base = philox_offset2
    if philox_offset1.cast(dtype=tl.uint64, bitcast=True) != 0:
        philox_offset_base += tl.load(philox_offset1)
    idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)
    philox_offset_stride = tl.cdiv(Max_seqlen_k, PHILOX_RN_PER_OFFSET)

    start_m = tl.program_id(2)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(0)
    off_zh = off_z * Num_head_q + off_h_q
    encoded_sm_base = R + off_z * stride_rz + off_h_q * stride_rh
    batch_philox_offset = philox_offset_base + off_zh * Max_seqlen_q * philox_offset_stride
    # Simulated value, should not be used to validate qk's correctness
    p = tl.full([BLOCK_M, BLOCK_N], 0.5, dtype=R.type.element_ty)
    for start_n in range(0, Max_seqlen_k, BLOCK_N):
        keep = fast_dropout_mask(philox_seed, idropout_p,
                                 batch_philox_offset, start_m * BLOCK_M, start_n,
                                 BLOCK_M, BLOCK_N, philox_offset_stride)
        mstore2d(tl.where(keep, p, -p),
                 BLOCK_M,
                 BLOCK_N,
                 o_base=encoded_sm_base,
                 o_start_row=start_m * BLOCK_M,
                 o_start_col=start_n,
                 o_rows=Max_seqlen_q,
                 o_cols=Max_seqlen_k,
                 stride_row=stride_rm,
                 stride_col=1)
