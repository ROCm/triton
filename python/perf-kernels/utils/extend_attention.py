# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

import pytest

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"



is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

is_hip_ = is_hip()


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q.to(k.dtype), k)
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        STORE_TRANSPOSE=is_hip_,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


@triton.jit
def _fwd_fused_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    ####
    W_KC,
    W_VC,
    stride_w_h,
    stride_w_c,
    stride_w_d,
    BLOCK_C: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    FUSE_GEMMS: tl.constexpr,
    ABSORB_W_KC: tl.constexpr,
    DPE: tl.constexpr,
    ####
    logit_cap: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, D)
    offs_block_d = tl.arange(0, BLOCK_D)

    offs_c = tl.arange(0, C)
    offs_block_c = tl.arange(0, BLOCK_C)

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    offs_dv = tl.arange(0, DV)

    mask_d = offs_d < D
    mask_c = offs_c < C
    mask_dv = offs_dv < DV

    TILE_D: tl.constexpr = BLOCK_D < D

    if FUSE_GEMMS:
        if not TILE_D: # if not tiling along D, load q only once outside the inner loop
            offs_q_d = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qbs
                + cur_head * stride_qh
                + offs_block_d[None, :]
            )
            q_d = tl.load(
                Q_Extend + offs_q, mask=(mask_m[:, None]), other=0.0
            )
    else:
        offs_q = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q_Extend + offs_q, mask=(mask_m[:, None]), other=0.0
        )

    if DPE > 0:
        offs_dpe = tl.arange(0, DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :] + D
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")


    if FUSE_GEMMS:
        if ABSORB_W_KC: # absorb the w_kc into q
            # load q and w_kc in BLOCK_D parts
            if TILE_D:
                offs_q_d = (
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                    * stride_qbs
                    + cur_head * stride_qh
                    + offs_block_d[None, :]
                )
            offs_w_kc_d = (
                cur_head * stride_w_h + offs_block_d[:, None] * stride_w_d + offs_c[None, :] * stride_w_c
            )
            q_acc = tl.zeros((BLOCK_M, C), dtype=tl.float32)
            for d in range(0, tl.cdiv(D, BLOCK_D)):
                w_kc_d = tl.load(W_KC + offs_w_kc_d + d * BLOCK_D * stride_w_d)
                if TILE_D:
                    q_d = tl.load(
                        Q_Extend + offs_q_d + d * BLOCK_D, mask=(mask_m[:, None]), other=0.0
                    )
                q_acc += tl.dot(q_d, w_kc_d)        
            q_acc = q_acc.to(qpe.dtype)

        # load w_kc and w_vc in BLOCK_C parts
        offs_w_kc = (
            cur_head * stride_w_h + offs_block_d[:, None] * stride_w_d + offs_block_c[None, :] * stride_w_c
        )
        offs_w_vc = (
            cur_head * stride_w_h + offs_d[None, :] * stride_w_d + offs_block_c[:, None] * stride_w_c
        )
    else:
        offs_w_kc = 0
        offs_w_vc = 0


    # stage 1: compute scores with prefix
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )

        if ABSORB_W_KC:
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_c[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_c[:, None]), other=0.0
            )
            qk = tl.dot(q_acc.to(k.dtype), k)
        elif FUSE_GEMMS:
            offs_buf_k_c = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_block_c[:, None]
            )
            if TILE_D:
                offs_q_d = (
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                    * stride_qbs
                    + cur_head * stride_qh
                    + offs_block_d[None, :]
                )
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for d in range(0, tl.cdiv(D, BLOCK_D)):
                for c in range(0, tl.cdiv(C, BLOCK_C)):
                    if TILE_D:
                        q_d = tl.load(
                            Q_Extend + offs_q_d + d * BLOCK_D, mask=(mask_m[:, None]), other=0.0
                        )
                    w_kc_c = tl.load(W_KC + offs_w_kc + c * BLOCK_C * stride_w_c + d * BLOCK_D * stride_w_d)
                    kv_c = tl.load(
                        K_Buffer + offs_buf_k_c + c * BLOCK_C, mask=(mask_n[None, :]), other=0.0
                    )
                    k_c = tl.dot(w_kc_c, kv_c) # projected keys
                    qk += tl.dot(q_d, k_c.to(kv_c.dtype))
        else:
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_c[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_c[:, None]), other=0.0
            )
            qk = tl.dot(q.to(k.dtype), k)
        
        
        if DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None] + C
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        e_max = n_e_max

        if FUSE_GEMMS:
            offs_buf_v_c = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_block_c[None, :]
            )
            acc = acc * re_scale[:, None]
            for c in range(0, tl.cdiv(C, BLOCK_C)):
                w_vc_c = tl.load(W_VC + offs_w_vc + c * BLOCK_C * stride_w_c)
                kv_c = tl.load(
                    V_Buffer + offs_buf_v_c + c * BLOCK_C, mask=mask_n[:, None], other=0.0
                )
                v_c = tl.dot(kv_c, w_vc_c) # projected values
                acc += tl.dot(p.to(kv_c.dtype), v_c.to(kv_c.dtype))
        else:
            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_c[None, :]
            )       
            v = tl.load(
                V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_c[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)


    # stage 2: compute the triangle part
    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        if ABSORB_W_KC:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_c[:, None]
            )
            
            k =  tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_c[:, None]), other=0.0
            )

            qk = tl.dot(q_acc, k, out_dtype=tl.float32)
        elif FUSE_GEMMS:
            offs_k_c = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_block_c[:, None]
            )
            if TILE_D:
                offs_q_d = (
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                    * stride_qbs
                    + cur_head * stride_qh
                    + offs_block_d[None, :]
                )
            # stage 2
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for d in range(0, tl.cdiv(D, BLOCK_D)):
                for c in range(0, tl.cdiv(C, BLOCK_C)):
                    if TILE_D:
                        q_d = tl.load(
                            Q_Extend + offs_q_d + d * BLOCK_D, mask=(mask_m[:, None]), other=0.0
                        )
                    w_kc_c = tl.load(W_KC + offs_w_kc + c * BLOCK_C * stride_w_c + d * BLOCK_D * stride_w_d)
                    kv_c =  tl.load(
                        K_Extend + offs_k_c + c * BLOCK_C, mask=(mask_n[None, :]), other=0.0
                    )
                    k_c = tl.dot(w_kc_c, kv_c) # projected keys
                    qk += tl.dot(q_d, k_c.to(kv_c.dtype), out_dtype=tl.float32)
        else:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_c[:, None]
            )
            
            k =  tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_c[:, None]), other=0.0
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)


        if DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None] + C
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        
        if FUSE_GEMMS:
            offs_v_c = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_block_c[None, :]
            )
            acc = acc * re_scale[:, None]
            for c in range(0, tl.cdiv(C, BLOCK_C)):
                w_vc_c = tl.load(W_VC + offs_w_vc + c * BLOCK_C * stride_w_c)
                kv_c = tl.load(
                    V_Extend + offs_v_c + c * BLOCK_C, mask=mask_n[:, None], other=0.0
                )
                
                v_c = tl.dot(kv_c, w_vc_c) # projected values

                acc += tl.dot(p.to(kv_c.dtype), v_c.to(kv_c.dtype))
        else:
            offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_c[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_c[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)



    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )

    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def extend_fused_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    fuse_gemms=False,
    w_kc=None,
    w_vc=None,
    absorb_w_kc=False,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """

    C = v_buffer.shape[-1] 
    DPE = k_buffer.shape[-1] - C
    D = q_extend.shape[-1] - DPE

    BLOCK_C = min(256, C)
    BLOCK_D = min(64, D)
    # tl.dots inside the kernel are of size
    # first gemm
    # (BLOCK_M, BLOCK_D) x (BLOCK_D, C) = (BLOCK_M, C) # accumulate over BLOCK_D. With absorb_w_kc 
    # or
    # (BLOCK_M, BLOCK_D) x (BLOCK_D, BLOCK_C) x (BLOCK_C, BLOCK_N) = (BLOCK_M, BLOCK_N) # accumulate over BLOCK_D and BLOCK_C. With absorb_w_kc = False.
    # second gemm
    # (BLOCK_M, BLOCK_N) x (BLOCK_N, BLOCK_C) x (BLOCK_C, DV) = (BLOCK_M, DV) # accumulate over BLOCK_C

    # We could probably do 
    # (BLOCK_M, BLOCK_N) x (BLOCK_N, BLOCK_C) x (BLOCK_C, BLOCK_D) = (BLOCK_M, BLOCK_D) # accumulate for BLOCK_C, and some partial store for BLOCK_D

    if fuse_gemms:
        assert w_kc is not None and w_vc is not None, "w_kc and w_vc must be provided when fusing gemms"
        DV = w_vc.shape[-1]
    else:
        DV = v_buffer.shape[-1] # no projection

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    sm_scale = sm_scale or 1.0 / ((k_buffer.shape[-1])**0.5) # TODO: check that this is correct here
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}


    _fwd_fused_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        # FUSE_GEMMS arguments
        w_kc if fuse_gemms else None,
        w_vc if fuse_gemms else None,
        w_kc.stride(0) if fuse_gemms else None,
        w_kc.stride(1) if fuse_gemms else None,
        w_kc.stride(2) if fuse_gemms else None,
        BLOCK_C=BLOCK_C,
        C=C,
        BLOCK_D=BLOCK_D,
        D=D,
        DV=DV,
        DPE=DPE,
        FUSE_GEMMS=fuse_gemms,
        ABSORB_W_KC=absorb_w_kc and fuse_gemms,
        ##################
        logit_cap=logit_cap,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
        STORE_TRANSPOSE=is_hip_,
        **extra_kargs,
    )