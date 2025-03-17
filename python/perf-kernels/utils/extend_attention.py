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

3 versions:

1. extend_attention_fwd: the reference version from sglang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L291.
2. extend_fused_attention: a version which fuses the computation of Q = Q * W_KC and O = O * W_VC inside the kernel. fp8 is supported for the gemms.
3. extend_persistent_attention: a persistent kernel version of the reference version.

"""

import torch
import triton
import triton.language as tl
from typing import Tuple

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

is_hip_ = is_hip()

def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    fp8_max = finfo.max
    if is_hip_:
        fp8_max = 224.0
    scale = fp8_max / amax
    x_scl_sat = (x * scale).clamp(min=-fp8_max, max=fp8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


fused_autotune_configs =  [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_C': 32, 'BLOCK_D': 32, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_C': 64, 'BLOCK_D': 64, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_C': 128, 'BLOCK_D': 32, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_C': 256, 'BLOCK_D': 128, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
    ], 

autotune_keys = [
        'logit_cap',
        'Lq',
        'Lv',
        'BLOCK_DMODEL',
        'BLOCK_DPE',
        'BLOCK_DV',
        'BLOCK_M',
        'BLOCK_N',
        'USE_CUSTOM_MASK',
        'SKIP_PREFIX_CUSTOM_MASK',
        'STORE_TRANSPOSE',
    ]

autotune_configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1},
                      num_stages=1, num_warps=4),
    ]



# @triton.autotune(
#     configs=autotune_configs,
#     key=autotune_keys,
#     use_cuda_graph=True,
# )
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


# @triton.autotune(
#     configs=fused_autotune_configs,
#     key=autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def _fwd_fused_kernel(
    Q_NOPE,
    Q_PE,
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
    stride_qpe_bs,
    stride_qpe_h,
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
    stride_w_kc_h,
    stride_w_kc_c,
    stride_w_kc_d,
    stride_w_vc_h,
    stride_w_vc_c,
    stride_w_vc_d,
    Q_descale,
    W_descale,
    FP8: tl.constexpr,
    FP8_max,
    BLOCK_C: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    #
    BLOCK_DQ: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DO: tl.constexpr,
    DQ: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    DO: tl.constexpr,
    DACC: tl.constexpr,
    FUSE_W_KC: tl.constexpr,
    FUSE_W_VC: tl.constexpr,
    DPE: tl.constexpr,
    ####
    logit_cap: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    PERSISTENT: tl.constexpr,
    NUM_WG: tl.constexpr,
    B,S,H,
    atomic_counter,
):
    """
    Q_NOPE: [T, H, D]
    Q_PE: [T, H, DPE]
    K_Extend: [T, C+DPE]
    V_Extend: [T, C]
    O_Extend: [T, H, DV]
    K_Buffer: [T_, H, C+DPE]
    V_Buffer: [T_, H, C]
    
    T: B * extend_lens, note that they can be varlen
    T_: B * prefix_lens, note that they can be varlen

    index pointers to access correct sequences in above tensors:
    qo_indptr: [B+1]
    kv_indptr: [B+1]

    gemms: 
    Q_Extend[..., :D] * W_KC * K_Buffer/Extend[..., :C]
    P * V_Buffer/Extend * W_VC
    
    W_KC: [H, D, C]
    W_VC: [H, C, D]
    FUSE_W_KC: do we fuse the gemm with w_kc
    FUSE_W_VC: do we fuse the gemm with w_vc
    """

    TILE_D: tl.constexpr = BLOCK_D < DQ # do we tile the w_kc fusion along D?
    TRANS: tl.constexpr = FUSE_W_KC and FUSE_W_VC # can we replace v with k.trans()?

    if PERSISTENT: # if persistent, kernel loops over multiple pids (tiles along Q)
        pid = atomic_counter.atomic_add(1)
        num_pids_per_head = tl.cdiv(S, BLOCK_M)
        num_pids_per_seq = num_pids_per_head * H
        num_pids_total = num_pids_per_seq * B
    else:  # standard, kernel processes only one pid
        pid = 0
        num_pids_total = 1
    
    while pid < num_pids_total:
        if PERSISTENT:
            cur_seq = pid // num_pids_per_seq
            cur_head = pid % num_pids_per_seq // num_pids_per_head
            cur_block_m = pid % num_pids_per_seq % num_pids_per_head
        else:
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

        offs_m = tl.arange(0, BLOCK_M)
        mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

        # weight matrix dimensions
        offs_d = tl.arange(0, BLOCK_D)
        offs_c = tl.arange(0, BLOCK_C)

        # tensor dimensions    
        offs_dq = tl.arange(0, BLOCK_DQ)
        offs_dk = tl.arange(0, BLOCK_DK)
        offs_dv = tl.arange(0, BLOCK_DV)
        offs_dpe = tl.arange(0, DPE)

        offs_do = tl.arange(0, BLOCK_DO)
        mask_do = offs_do < DO

        mask_d = offs_d < D
        mask_c = offs_c < C
        mask_dq = offs_dq < DQ
        mask_dk = offs_dk < DK
        mask_dv = offs_dv < DV

        offs_n = tl.arange(0, BLOCK_N)
        
        acc = tl.zeros([BLOCK_M, DACC], dtype=tl.float32)
        deno = tl.zeros([BLOCK_M], dtype=tl.float32)
        e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

        if FUSE_W_KC:
            offs_q_d = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qbs
                + cur_head * stride_qh
                + offs_d[None, :]
            )
            offs_w_kc_d = (
                cur_head * stride_w_kc_h + offs_d[:, None] * stride_w_kc_d + offs_c[None, :] * stride_w_kc_c
            )
            if TILE_D:
                q = tl.zeros((BLOCK_M, C), dtype=tl.float32)
                for d in range(0, tl.cdiv(D, BLOCK_D)):
                    w_kc_d = tl.load(W_KC + offs_w_kc_d + d * BLOCK_D * stride_w_kc_d, mask= ((offs_d + d * BLOCK_D)[:, None] < D) & mask_c[None,:], other=0.0)                        
                    q_d = tl.load(
                        Q_NOPE + offs_q_d + d * BLOCK_D, mask=(mask_m[:, None]) & ( (offs_d + d * BLOCK_D)[None, :] < D), other=0.0
                    )
                    # (BLOCK_M, BLOCK_D) * (BLOCK_D, C)
                    q += tl.dot(q_d, w_kc_d, out_dtype=tl.float32)
                
            else: # no tiling along d
                w_kc_d = tl.load(W_KC + offs_w_kc_d, mask=mask_d[:, None] & mask_c[None, :], other=0.0)
                q_d = tl.load(
                    Q_NOPE + offs_q_d, mask=(mask_m[:, None]) & (mask_dq[None, :]), other=0.0
                )
                # (BLOCK_M, D) * (D, C)
                q = tl.dot(q_d, w_kc_d, out_dtype=tl.float32)       
            
            if FP8:
                q = q * Q_descale * W_descale

            q = q.to(K_Extend.type.element_ty) 

        else: # reference
            offs_q = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qbs
                + cur_head * stride_qh
                + offs_dq[None, :]
            )
            q = tl.load(
                Q_NOPE + offs_q, mask=(mask_m[:, None]) & (mask_dq[None, :]), other=0.0
            )
        
        if DPE > 0:
            offs_qpe = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qpe_bs
                + cur_head * stride_qpe_h
                + offs_dpe[None, :]
            )
            qpe = tl.load(Q_PE + offs_qpe, mask=mask_m[:, None], other=0.0)
            qpe = qpe.to(K_Extend.type.element_ty)

        

        # stage 1: compute scores with prefix
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
                + offs_dk[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_dk[:, None]), other=0.0
            )
            # if ref (absorb) or FUSE_W_KC
            # (BLOCK_M, 512) * (512, BLOCK_N)
            # if ref (normal)
            # (BLOCK_M, 128) * (128, BLOCK_N)
            qk = tl.dot(q.to(k.dtype), k)
            
            if DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None] + DK
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

            # (BLOCK_M, BLOCK_N) * (BLOCK_N, C if ref (absorb) or FUSE_W_VC else D)
            if TRANS:
                p = p.to(k.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p, k.trans()) 
            else:
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
                


        # stage 2: compute the triangle part
        cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
        for start_n in range(0, cur_block_m_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            mask_n = (start_n + offs_n) < cur_block_m_end
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dk[:, None]
            )
            k =  tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_dk[:, None]), other=0.0
            )
            # if ref (absorb) or FUSE_W_KC
            # (BLOCK_M, 512) * (512, BLOCK_N)
            # else
            # (BLOCK_M, 128) * (128, BLOCK_N)

            qk = tl.dot(q, k, out_dtype=tl.float32)

            if DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None] + DK
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
            
            # (BLOCK_M, BLOCK_N) * (BLOCK_N, C if ref (absorb) or FUSE_W_VC else D)
            if TRANS:
                p = p.to(k.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p, k.trans()) 
            else:
                offs_v = (
                    (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                    + cur_kv_head * stride_vh
                    + offs_dv[None, :]
                )
                v = tl.load(
                    V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
                )

        # We can absorb the w_vc outside the loop
        if FUSE_W_VC:
            offs_w_vc_d = (
                cur_head * stride_w_vc_h + offs_c[:, None] * stride_w_vc_c + offs_do[None, :] * stride_w_vc_d 
            )
            if FP8:
                max_val = tl.max(tl.abs(acc)).to(tl.float32)
                max_val = tl.clamp(max_val, 1e-12, max_val)
                scale = FP8_max / max_val
                descale = max_val / FP8_max
                acc = tl.clamp(acc * scale, -FP8_max, FP8_max)
            
            acc = acc.to(W_VC.type.element_ty)
            for d in range(0, tl.cdiv(DO, BLOCK_DO)):
                offs_o_block = offs_do + d * BLOCK_DO
                w_vc_d = tl.load(W_VC + offs_w_vc_d + d * BLOCK_DO * stride_w_vc_d, mask= mask_c[:, None] & (offs_o_block[None, :] < DO), other=0.0)
                # (BLOCK_M, C) * (C, BLOCK_D)
                acc_d = tl.dot(acc, w_vc_d)
                if FP8:
                    acc_d = acc_d * descale * W_descale
                acc_d = acc_d.to(O_Extend.type.element_ty)
                
                offs_o_d = (
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                    * stride_obs
                    + cur_head * stride_oh
                    + offs_o_block[None, :]
                )
                if STORE_TRANSPOSE:
                    tl.store(
                        O_Extend + offs_o_d.T,
                        (acc_d / deno[:, None]).T,
                        mask=((mask_m[:, None] & (offs_o_block[None, :] < DO))).T,
                    )
                else:
                    tl.store(
                        O_Extend + offs_o_d,
                        acc_d / deno[:, None],
                        mask=(mask_m[:, None] & (offs_o_block[None, :] < DO)),
                    )
        else:
            offs_o = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_obs
                + cur_head * stride_oh
                + offs_do[None, :]
            )
            if STORE_TRANSPOSE:
                tl.store(
                    O_Extend + offs_o.T,
                    (acc / deno[:, None]).T,
                    mask=(mask_m[:, None] & mask_do[None, :]).T,
                )
            else:
                tl.store(
                    O_Extend + offs_o,
                    acc / deno[:, None],
                    mask=mask_m[:, None] & mask_do[None, :],
                )
        
        if PERSISTENT:
            pid = atomic_counter.atomic_add(1)
        else:
            pid = num_pids_total # break the while loop


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
    fuse_w_kc=False,
    fuse_w_vc=False,
    w_kc=None,
    w_vc=None,
    w_descale=None,
    fp8=False,
    persistent=False,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager

    tl.dots inside the kernel
    first gemm: q * w_kc * kv : (BLOCK_M, D) x (D, C) x (C, BLOCK_N) = (BLOCK_M, BLOCK_N)
    
    outside loop: q * w_kc: (BLOCK_M, BLOCK_D) x (BLOCK_D, C) = (BLOCK_M, C) # tile across D
    inside the loop: (BLOCK_M, C) x (C, BLOCK_N) = (BLOCK_M, BLOCK_N). This is the same as in ref.
    """

    # persistent only when all the sequences are prefill
    # persistent = False if (kv_indptr[1:] - kv_indptr[:-1]).sum() > 0 else persistent # TODO: find a better heuristic

    TILE_D = 128 if fp8 else 32 # q = q * w_kc = (BLOCK_M, BLOCK_D) x (BLOCK_D, C) = (BLOCK_M, C)
    TILE_DO = 128 if fp8 else 16 # o = o * w_vc = (BLOCK_M, C) x (C, BLOCK_DO) = (BLOCK_M, BLOCK_DO)
    
    DV = v_buffer.shape[-1]

    DPE, C, D = qk_rope_head_dim, kv_lora_rank, qk_nope_head_dim

    DQ = q_extend.shape[-1] - DPE
    DK = k_extend.shape[-1] - DPE
    
    BLOCK_DQ = triton.next_power_of_2(DQ)
    BLOCK_DK = triton.next_power_of_2(DK)
    BLOCK_DV = triton.next_power_of_2(DV)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_D = triton.next_power_of_2(D)
    
    if fuse_w_kc:
        assert w_kc is not None, "w_kc must be provided"
        BLOCK_D = min(TILE_D, BLOCK_D)
    else:
        BLOCK_D = min(BLOCK_D, DV)

    if fuse_w_vc:
        assert w_vc is not None, "w_vc must be provided"
        DACC = v_extend.shape[-1]
        DO = w_vc.shape[-1] 
        BLOCK_DO = min(TILE_DO, triton.next_power_of_2(DO))
    else:
        DACC = v_extend.shape[-1] # no projection
        DO = v_extend.shape[-1] # no projection
        BLOCK_DO = triton.next_power_of_2(DO)
            
    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    sm_scale = sm_scale or 1.0 / ((k_extend.shape[-1])**0.5) # TODO: check that this is correct here
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    
    # k heads can be num heads or 1, v heads can be num heads or 1 (depending on if they are projected or not)
    kv_heads = v_extend.shape[1] if fuse_w_kc else k_extend.shape[1]
    kv_group_num = q_extend.shape[1] // kv_heads

    USE_CUSTOM_MASK = custom_mask is not None

    if persistent:
        NUM_WG = torch.cuda.get_device_properties("cuda").multi_processor_count
        atomic_counter = torch.zeros([1], device=q_extend.device, dtype=torch.int32)
        grid = (min(NUM_WG, batch_size * head_num * triton.cdiv(max_len_extend, BLOCK_M)),)
    else:
        NUM_WG = 0
        atomic_counter = None
        grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    q_nope = q_extend[..., :DQ]
    q_pe = q_extend[..., DQ:]

    q_descale = None

    # FP8
    if not fp8 and fuse_w_kc: # descale w_kc
        w_kc = (w_kc.to(torch.bfloat16) * w_descale).to(q_nope.dtype)
           
    if not fp8 and fuse_w_vc: # descale w_vc
        w_vc = (w_vc.to(torch.bfloat16) * w_descale).to(q_nope.dtype) 

    if fp8 and fuse_w_kc: # quantize q_nope part
        q_nope, q_descale = input_to_float8(q_nope, w_kc.dtype)
        q_descale = q_descale.item()

    fp8_e4m3fnuz_max = torch.finfo(torch.float8_e4m3fnuz).max

    _fwd_fused_kernel[grid](
        # input tensors
        q_nope,
        q_pe,
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
        # strides
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        k_extend.stride(0),
        0 if fuse_w_kc else k_extend.stride(1),
        v_extend.stride(0),
        0 if fuse_w_vc else v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        0 if fuse_w_kc else k_buffer.stride(1), 
        v_buffer.stride(0),
        0 if fuse_w_vc else v_buffer.stride(1),
        # fuse gemm arguments
        w_kc,
        w_vc,
        w_kc.stride(0) if w_kc is not None else 0,
        w_kc.stride(1) if w_kc is not None else 0,
        w_kc.stride(2) if w_kc is not None else 0,
        w_vc.stride(0) if w_vc is not None else 0,
        w_vc.stride(1) if w_vc is not None else 0,
        w_vc.stride(2) if w_vc is not None else 0,
        q_descale,
        w_descale,
        FP8=fp8,
        FP8_max=fp8_e4m3fnuz_max,
        FUSE_W_KC=fuse_w_kc,
        FUSE_W_VC=fuse_w_vc,
        C=C,
        D=D,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        # shape params
        DQ=DQ,
        DK=DK,
        DV=DV,
        DO=DO,
        BLOCK_DQ=BLOCK_DQ,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DO=BLOCK_DO,
        DPE=DPE,
        DACC=DACC,
        logit_cap=logit_cap,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
        STORE_TRANSPOSE=is_hip_,
        # persistent kernels arguments
        PERSISTENT=persistent,
        atomic_counter=atomic_counter,
        B=batch_size, S=max_len_extend, H=head_num, NUM_WG=NUM_WG,
        #
        **extra_kargs,
    )




@triton.jit
def _fwd_persistent_kernel(
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
    PERSISTENT: tl.constexpr,
    NUM_WG: tl.constexpr,
    B,S,H,
    atomic_counter,
):
    if PERSISTENT: # if persistent, kernel loops over multiple pids (tiles along Q)
        pid = atomic_counter.atomic_add(1)
        num_pids_per_head = tl.cdiv(S, BLOCK_M)
        num_pids_per_seq = num_pids_per_head * H
        num_pids_total = num_pids_per_seq * B
    else:  # standard, kernel processes only one pid
        pid = 0
        num_pids_total = 1
    
    while pid < num_pids_total:
        if PERSISTENT:
            cur_seq = pid // num_pids_per_seq
            cur_head = pid % num_pids_per_seq // num_pids_per_head
            cur_block_m = pid % num_pids_per_seq % num_pids_per_head
        else:
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
        
        if PERSISTENT:
            pid = atomic_counter.atomic_add(1)
        else:
            pid = num_pids_total # break the while loop


def extend_persistent_attention_fwd(
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
    persistent=True,
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

    if persistent:
        NUM_WG = torch.cuda.get_device_properties("cuda").multi_processor_count
        atomic_counter = torch.zeros([1], device=q_extend.device, dtype=torch.int32)
        grid = (min(NUM_WG, batch_size * head_num * triton.cdiv(max_len_extend, BLOCK_M)),)
    else:
        NUM_WG = 0
        atomic_counter = None
        grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
    

    _fwd_persistent_kernel[grid](
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
        PERSISTENT=persistent,
        atomic_counter=atomic_counter,
        B=batch_size, S=max_len_extend, H=head_num, NUM_WG=NUM_WG,
    )