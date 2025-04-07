from typing import Dict, Optional
import torch.test
import triton
import torch
import triton.language as tl
import pytest
import os
import functools
import argparse
import sys
from moe_gemm import moe_align_block_size, silu_and_mul, try_get_optimal_moe_config, get_config_dtype_str, quantize_tensor, moe_gemm, MetaData as MoEMetaData, get_moe_configs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # This goes one level up from fused-moe/
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils.benchmark_utils import get_available_models, get_model_configs  # noqa: E402

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]
}

supported_fp8 = [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]


class MetaData():
    use_fp8_w8a8 = False
    use_int8_w8a16 = False

    def __init__(self, top_k, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config):
        self.top_k = top_k
        self.topk_weights = topk_weights
        self.topk_ids = topk_ids
        self.sorted_token_ids = sorted_token_ids
        self.expert_ids = expert_ids
        self.num_tokens_post_padded = num_tokens_post_padded
        self.config = config

    def set_use_fp8_w8a8(self, a_descale, w1_descale, w2_descale, fp8_type):
        self.use_fp8_w8a8 = True
        self.a_descale = a_descale
        self.w1_descale = w1_descale
        self.w2_descale = w2_descale
        self.fp8_type = fp8_type

    def set_use_int8_w8a16(self, w1_descale, w2_descale):
        self.use_int8_w8a16 = True
        self.w1_descale = w1_descale
        self.w2_descale = w2_descale
        self.a_descale = None

    def check_args(self, a, w1, w2, o):
        assert a.shape[-1] == w1.shape[-1]
        assert w1.shape[-1] == w2.shape[-2]
        assert w1.shape[-2] // 2 == w2.shape[-1]
        assert o.shape[-1] == a.shape[-1] and o.shape[-1] == w1.shape[-1]

        assert not (self.use_fp8_w8a8 and self.use_int8_w8a16)
        if self.use_fp8_w8a8:
            assert self.fp8_type in supported_fp8, f"fp8 type {self.fp8_type} not supported"


@triton.jit
def e2e_moe_persistent_kernel(
    A,
    W1,
    W2,
    Out,
    A_scale,
    W1_scale,
    W2_scale,
    stride_am,
    stride_ak,
    stride_w1e,
    stride_w1n,
    stride_w1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_cm,
    stride_w1se,
    stride_w1sn,
    stride_w2se,
    stride_w2sk,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    intermediate_ptr,
    stride_im,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr, # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr, # outputs (EM, BLOCK_SIZE_K2)
    NUM_SMS: tl.constexpr,
    ):
    start_m = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N1)
    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K2)
    m_tile_per_sm = num_pid_m // NUM_SMS

    if start_m < num_pid_m % NUM_SMS:
        m_tile_per_sm += 1

    N_HALF: tl.constexpr = N // 2
    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N1 // 2

    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)
    offs_n1 = tl.arange(0, BLOCK_SIZE_N1)
    offs_n1_half = tl.arange(0, BLOCK_SIZE_HALF)
    offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    i = offs_n1.to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2

    dtype = Out.dtype.element_ty

    pid_m = start_m - NUM_SMS

    for _ in range(0, m_tile_per_sm):
        pid_m += NUM_SMS
        # pid_m = pid_m_start + m_off
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

        # Here we assume that valid tokens are in the range [0, M).
        token_mask = (offs_token >= 0) & (offs_token < EM)

        off_experts = tl.load(expert_ids_ptr + pid_m)
        # tl.device_print("pid_m", pid_m)
        # TODO mem fault when when pid_n != 0
        for pid_n in range(0, num_pid_n):
            offs_half = (pid_n * BLOCK_SIZE_HALF + i_floor) % N_HALF
            # (i % 2): [0, 1, 0, 1, ...] (alternating)
            # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
            # So offs_w1n now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
            offs_w1n = (offs_half + (i % 2) * (N_HALF)) % N

            mask_w1n = (pid_n * BLOCK_SIZE_N1 + i) < N

            a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k1[None, :] * stride_ak)
            w1_ptrs = W1 + off_experts * stride_w1e + (offs_k1[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N1), dtype=tl.float32)

            if use_int8_w8a16:
                w1_scale_ptrs = W1_scale + off_experts * stride_w1se + offs_w1n[None, :] * stride_w1sn
                w1_scale = tl.load(w1_scale_ptrs)
            if use_fp8_w8a8:
                a_scale = tl.load(A_scale)
                w1_scale = tl.load(W1_scale + off_experts)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K1)):
                # Masking ensures we don't load from invalid tokens or indices
                if EVEN_K:
                    a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
                    # TODO memory fault N dim, might be k as well
                    w1 = tl.load(w1_ptrs, mask=mask_w1n[None, :], other=0.0)
                else:
                    a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k1[None, :] < K - k * BLOCK_SIZE_K1)), other=0.0)
                    w1 = tl.load(w1_ptrs, mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1) & mask_w1n[None, :], other=0.0)

                if use_int8_w8a16:
                    accumulator = tl.dot(a, w1.to(a.type), acc=accumulator)
                elif use_fp8_w8a8:
                    accumulator += tl.dot(a, w1)
                else:
                    accumulator = tl.dot(a, w1, acc=accumulator)
                a_ptrs += BLOCK_SIZE_K1 * stride_ak
                w1_ptrs += BLOCK_SIZE_K1 * stride_w1k

            if use_int8_w8a16:
                accumulator = (accumulator * w1_scale)
            elif use_fp8_w8a8:
                accumulator = (accumulator * a_scale * w1_scale)

            silu_acc, mul_acc = accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
            silu_acc = (silu_acc / (1.0 + tl.exp2(-(silu_acc * 1.44269504089))))
            acc = (silu_acc * mul_acc).to(dtype)

            offs_in = pid_n * BLOCK_SIZE_HALF + offs_n1_half
            i_mask = token_mask[:, None] & (offs_in[None, :] < N_HALF)
            i_ptrs = intermediate_ptr + stride_im * offs_token[:, None] + offs_in[None, :]
            # TODO dtye??
            tl.atomic_add(i_ptrs, acc, mask=i_mask, sem="release")
            # TODO quantization

        for pid_k in range(0, num_pid_k):
            offs_w2k = (pid_k * BLOCK_SIZE_K2 + offs_k2) % K
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

            intermediate_ptrs = intermediate_ptr + (offs_token[:, None] * stride_im + offs_n2[None, :])
            w2_ptrs = W2 + off_experts * stride_w2e + (offs_n2[:, None] * stride_w2n + offs_w2k[None, :] * stride_w2k)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K2), dtype=tl.float32)

            mask_w2k = (pid_k * BLOCK_SIZE_K2 + offs_k2) < K

            if use_int8_w8a16:
                w2_scale_ptrs = W2_scale + off_experts * stride_w2se + offs_k2[None, :] * stride_w2sk
                w2_scale = tl.load(w2_scale_ptrs)

            if use_fp8_w8a8:
                # TODO calculate the intermediate scale and scale intermediate
                # a_scale = tl.load(A_scale)
                i_scale = 1
                w2_scale = tl.load(W2_scale + off_experts)

            for n in range(0, tl.cdiv(N_HALF, BLOCK_SIZE_N2)):
                # Masking ensures we don't load from invalid tokens or indices

                if EVEN_N:
                    intermediate = tl.load(intermediate_ptrs, mask=(token_mask[:, None]), other=0.0)
                    w2 = tl.load(w2_ptrs)
                else:
                    intermediate = tl.load(intermediate_ptrs, mask=(token_mask[:, None] & (offs_n2[None, :] < N_HALF - n * BLOCK_SIZE_N2)), other=0.0)
                    w2 = tl.load(w2_ptrs, mask=(offs_n2[:, None] < N_HALF - n * BLOCK_SIZE_N2) & mask_w2k[None, :], other=0.0)

                if use_int8_w8a16:
                    accumulator = tl.dot(intermediate, w2.to(intermediate.type), acc=accumulator)
                elif use_fp8_w8a8:
                    accumulator += tl.dot(intermediate, w2)
                else:
                    accumulator = tl.dot(intermediate, w2, acc=accumulator)
                intermediate_ptrs += BLOCK_SIZE_N2
                w2_ptrs += BLOCK_SIZE_N2 * stride_w2n

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                accumulator = accumulator * moe_weight[:, None]

            if use_int8_w8a16:
                accumulator = (accumulator * w2_scale)
            elif use_fp8_w8a8:
                accumulator = (accumulator * i_scale * w2_scale)

            offs_ck = pid_k * BLOCK_SIZE_K2 + offs_k2
            c_mask = token_mask[:, None] & (offs_ck[None, :] < K)
            out_ptrs = Out + stride_cm * offs_token[:, None] + offs_ck[None, :]
            tl.store(out_ptrs, accumulator.to(dtype), mask=c_mask)


@triton.heuristics({
'GRID_MN':
    lambda args: triton.cdiv(args['EM'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
def e2e_moe_kernel(
    A,
    W1,
    W2,
    Out,
    A_scale,
    W1_scale,
    W2_scale,
    stride_am,
    stride_ak,
    stride_w1e,
    stride_w1n,
    stride_w1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_cm,
    stride_w1se,
    stride_w1sn,
    stride_w2se,
    stride_w2sk,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr, # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr, # outputs (EM, BLOCK_SIZE_K2)
    GROUP_SIZE_M: tl.constexpr,
    GRID_MN: tl.constexpr,
    atomic_num_stages: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - W1: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - W2: The stacked MOE weight tensor with shape (E, K, N // 2), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, K), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """


    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_w1e > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w2e > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_cm > 0)
    if use_int8_w8a16:
        tl.assume(stride_w1se > 0)
        tl.assume(stride_w1sn > 0)
        tl.assume(stride_w2se > 0)
        tl.assume(stride_w2sk > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_XCDS: tl.constexpr = 8

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    dtype = Out.dtype.element_ty

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    # Here we assume that valid tokens are in the range [0, M).
    token_mask = (offs_token >= 0) & (offs_token < EM)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    # (i % 2): [0, 1, 0, 1, ...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_w1n now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_w1n = (offs_half + (i % 2) * (N // 2)) % N

    mask_w1n = (pid_n * BLOCK_SIZE_N + i) < N

    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k1[None, :] * stride_ak)
    w1_ptrs = W1 + off_experts * stride_w1e + (offs_k1[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if use_int8_w8a16:
        w1_scale_ptrs = W1_scale + off_experts * stride_w1se + offs_w1n[None, :] * stride_w1sn
        w1_scale = tl.load(w1_scale_ptrs)

    if use_fp8_w8a8:
        a_scale = tl.load(A_scale)
        w1_scale = tl.load(W1_scale + off_experts)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K1)):
        # Masking ensures we don't load from invalid tokens or indices
        if EVEN_K:
            a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
            w1 = tl.load(w1_ptrs, mask=mask_w1n[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k1[None, :] < K - k * BLOCK_SIZE_K1)), other=0.0)
            w1 = tl.load(w1_ptrs, mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1) & mask_w1n[None, :], other=0.0)

        if use_int8_w8a16:
            accumulator = tl.dot(a, w1.to(a.type), acc=accumulator)
        elif use_fp8_w8a8:
            accumulator += tl.dot(a, w1)
        else:
            accumulator = tl.dot(a, w1, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K1 * stride_ak
        w1_ptrs += BLOCK_SIZE_K1 * stride_w1k

    if use_int8_w8a16:
        accumulator = (accumulator * w1_scale)
    elif use_fp8_w8a8:
        accumulator = (accumulator * a_scale * w1_scale)

    silu_acc, mul_acc = accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    silu_acc = (silu_acc / (1.0 + tl.exp2(-(silu_acc * 1.44269504089))))
    acc = (silu_acc * mul_acc).to(dtype)

    # TODO scale acc
    acc_scale = 1.0
    # TODO scale acc
    # -------------------------------

    offs_w2n = tl.arange(0, BLOCK_SIZE_N // 2) + pid_n * (BLOCK_SIZE_N // 2)

    w2_ptrs = W2 + off_experts * stride_w2e + (offs_k2[None, :] * stride_w2k + offs_w2n[:, None] * stride_w2n)
    out_ptrs = Out + stride_cm * offs_token[:, None] + offs_k2[None, :]

    if use_fp8_w8a8:
        w2_scale = tl.load(W2_scale + off_experts)

    # minus if pid_m is even otherwise positive
    k_sign = (pid_m % 2) * 2 - 1
    num_k = tl.cdiv(K, BLOCK_SIZE_K2)
    for _k in tl.range(0, num_k, num_stages=atomic_num_stages):
        k = (num_k + (_k * k_sign)) % num_k
        k = ((k + pid_n * 4)) % num_k
        # k = _k

        if use_int8_w8a16:
            w2_scale_ptrs = W2_scale + off_experts * stride_w2se + (offs_k2 + k * BLOCK_SIZE_K2)[None, :] * stride_w2sk
            w2_scale = tl.load(w2_scale_ptrs)

        if EVEN_K:
            w2 = tl.load(w2_ptrs + k * BLOCK_SIZE_K2 * stride_w2k, mask=(offs_w2n[:, None] < N), other=0.0)
        else:
            w2 = tl.load(w2_ptrs + k * BLOCK_SIZE_K2 * stride_w2k, mask=((offs_w2n[:, None] < N) & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)), other=0.0)

        if use_int8_w8a16:
            out = tl.dot(acc, w2.to(a.type))
        else:
            out = tl.dot(acc, w2)

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
            out = out * moe_weight[:, None]

        if use_int8_w8a16:
            out = (out * w2_scale)
        elif use_fp8_w8a8:
            out = (out * acc_scale * w2_scale)

        # # atomic add
        if EVEN_K:
            c_mask = token_mask[:, None]
        else:
            c_mask = token_mask[:, None] & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)

        # TODO check scope
        tl.atomic_add(out_ptrs + k * BLOCK_SIZE_K2, out, mask=c_mask, sem="relaxed", scope="cta")
        # tl.store(out_ptrs + k * BLOCK_SIZE_K2, out, mask=c_mask)

def e2e_moe(a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, c: torch.Tensor, metadata: MetaData) -> torch.Tensor:
    metadata.check_args(a, w1, w2, c)

    topk_ids, num_tokens_post_padded, topk_weights, sorted_token_ids, expert_ids, config = metadata.topk_ids, metadata.num_tokens_post_padded, metadata.topk_weights, metadata.sorted_token_ids, metadata.expert_ids, metadata.config

    use_fp8_w8a8, use_int8_w8a16 = metadata.use_fp8_w8a8, metadata.use_int8_w8a16
    a_descale, w1_descale, w2_descale = None, None, None
    stride_w1se = None
    stride_w1sn = None
    stride_w2se = None
    stride_w2sk = None
    if use_fp8_w8a8 or use_int8_w8a16:
        a_descale, w1_descale, w2_descale = metadata.a_descale, metadata.w1_descale, metadata.w2_descale
        if use_int8_w8a16:
            stride_w1se = w1_descale.stride(0)
            stride_w1sn = w1_descale.stride(1)
            stride_w2se = w2_descale.stride(0)
            stride_w2sk = w2_descale.stride(1)

    _, top_k = topk_ids.shape

    EM = num_tokens_post_padded.item()
    _, N, K = w1.shape

    BLOCK_SIZE_K1 = config["BLOCK_SIZE_K1"]

    EVEN_K = K % BLOCK_SIZE_K1 == 0
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    stride_cm = c.stride(1)

    if EM > 1024:
        atomic_num_stages = 2
    else:
        atomic_num_stages = 1

    e2e_moe_kernel[grid](a, w1, w2, c, a_descale, w1_descale, w2_descale, a.stride(0), a.stride(1), w1.stride(0), w1.stride(1),
                          w1.stride(2), w2.stride(0), w2.stride(2), w2.stride(1), stride_cm, stride_w1se, stride_w1sn, stride_w2se, stride_w2sk, top_k, topk_weights,
                          sorted_token_ids, expert_ids, EM, N, K, EVEN_K, MUL_ROUTED_WEIGHT=topk_weights is not None,
                          use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, atomic_num_stages=atomic_num_stages, **config
                          )
    return c


def e2e_moe_persistent(a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, intermediate: torch.Tensor, c: torch.Tensor, metadata: MetaData) -> torch.Tensor:
    metadata.check_args(a, w1, w2, c)

    topk_ids, num_tokens_post_padded, topk_weights, sorted_token_ids, expert_ids, config = metadata.topk_ids, metadata.num_tokens_post_padded, metadata.topk_weights, metadata.sorted_token_ids, metadata.expert_ids, metadata.config

    use_fp8_w8a8, use_int8_w8a16 = metadata.use_fp8_w8a8, metadata.use_int8_w8a16
    a_descale, w1_descale, w2_descale = None, None, None
    stride_w1se = None
    stride_w1sn = None
    stride_w2se = None
    stride_w2sk = None
    if use_fp8_w8a8 or use_int8_w8a16:
        a_descale, w1_descale, w2_descale = metadata.a_descale, metadata.w1_descale, metadata.w2_descale
        if use_int8_w8a16:
            stride_w1se = w1_descale.stride(0)
            stride_w1sn = w1_descale.stride(1)
            stride_w2se = w2_descale.stride(0)
            stride_w2sk = w2_descale.stride(1)

    _, top_k = topk_ids.shape

    EM = num_tokens_post_padded.item()
    _, N, K = w1.shape


    BLOCK_SIZE_K1 = config["BLOCK_SIZE_K1"]
    BLOCK_SIZE_N2 = config["BLOCK_SIZE_N2"]

    EVEN_K = K % BLOCK_SIZE_K1 == 0
    EVEN_N = (N // 2) % BLOCK_SIZE_N2 == 0
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2

    num_pid_m = triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"])

    grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            ), )
    stride_cm = c.stride(1)
    stride_im = intermediate.stride(0)

    e2e_moe_persistent_kernel[grid](a, w1, w2, c, a_descale, w1_descale, w2_descale, a.stride(0), a.stride(1), w1.stride(0), w1.stride(1),
                          w1.stride(2), w2.stride(0), w2.stride(2), w2.stride(1), stride_cm, stride_w1se, stride_w1sn, stride_w2se, stride_w2sk, top_k, topk_weights,
                          sorted_token_ids, expert_ids, intermediate, stride_im, EM, N, K, EVEN_K, EVEN_N, MUL_ROUTED_WEIGHT=topk_weights is not None,
                          use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, NUM_SMS=NUM_SMS, **config,
                          )

    
    return c, intermediate


def quantize_input(a, w1, w2, use_fp8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr, metatdata: MetaData, fp8_type=None):
    assert not (use_fp8_w8a8 and use_int8_w8a16)
    assert not (use_fp8_w8a8 and fp8_type is None)

    if use_fp8_w8a8:
        a_quantized, _, a_descale = quantize_tensor(a, dtype=fp8_type)
        w1_quantized, _, w1_descale = quantize_tensor(w1, dim=(0, ), dtype=fp8_type)
        w2_quantized, _, w2_descale = quantize_tensor(w2, dim=(0, ), dtype=fp8_type)
        metatdata.set_use_fp8_w8a8(a_descale, w1_descale, w2_descale, fp8_type)
        return a_quantized, w1_quantized, w2_quantized

    if use_int8_w8a16:
        w1_quantized, _, w1_descale = quantize_tensor(w1, dim=(0, 1), dtype=torch.int8)
        w2_quantized, _, w2_descale = quantize_tensor(w2, dim=(0, 1), dtype=torch.int8)
        metatdata.set_use_int8_w8a16(w1_quantized, w2_quantized)
        return a, w1_quantized, w2_quantized


def get_default_e2e_config(
    M: int,
    E: int,
    is_marlin: bool,
) -> Dict[str, int]:
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K1': 32,'BLOCK_SIZE_K2': 32, 'GROUP_SIZE_M': 8}
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K1': 64, 'BLOCK_SIZE_K2': 64, 'GROUP_SIZE_M': 1}
    return config

def try_get_optimal_e2e_moe_config(
    E: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    config_dir = "e2e_configs"
):
    configs = get_moe_configs(dtype, config_dir)

    if configs:
        if configs:
            if M < M_THRESHOLD_SMALL:
                config = configs["small_M"]
            elif M < M_THRESHOLD_MEDIUM:
                config = configs["medium_M"]
            else:
                config = configs["large_M"]
    else:
        # Else use the default config
        config = get_default_e2e_config(M, E, is_marlin)

    return config


def input_helper(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_fp8_w8a8: bool,
                 use_int8_w8a16: bool, fp8_type, dtype, persistent=None):
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    w1 = torch.randn((E, N, K), dtype=dtype, device='cuda')
    w2 = torch.randn((E, K, N // 2), dtype=dtype, device='cuda')

    out = torch.zeros((M, top_k, K), dtype=dtype, device='cuda')

    values = torch.randn(M, E, device='cuda')

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    config_dir = "e2e_configs_persistent" if persistent else "e2e_configs"

    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, dtype=dtype)
    get_config_func = functools.partial(
        try_get_optimal_e2e_moe_config,
        E,
        config_dtype,
    )
    config = get_config_func(M, config_dir=config_dir)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    metadata = MetaData(top_k, topk_weights if routed_weight else None, topk_ids, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, config)

    if use_fp8_w8a8 or use_int8_w8a16:
        a, w1, w2 = quantize_input(a, w1, w2, use_fp8_w8a8, use_int8_w8a16, metadata, fp8_type)

    if persistent:
        intermediate = torch.zeros((M * top_k, N // 2), dtype=dtype, device='cuda')
        return a, w1, w2, intermediate, out, metadata

    return a, w1, w2, out, metadata


def silu_and_mul_torch(input):
    """
    Performs the SiLU activation on the first half of the input tensor and
    multiplies it element-wise with the second half.

    Args:
        input (torch.Tensor): Input tensor of shape [..., 2 * d].
        param (float): Parameter for the SiLU activation function.

    Returns:
        torch.Tensor: Output tensor of shape [..., d].
    """
    d = input.size(-1) // 2
    A, B = input[:, :d], input[:, d:]

    silu_A = A / (1.0 + torch.exp(-A.float()))

    output = silu_A * B

    return output


def e2e_moe_ref(a, w1, w2, c, M, E, top_k, N, metadata: MetaData):
    config_dtype = get_config_dtype_str(use_fp8_w8a8=metadata.use_fp8_w8a8, use_int8_w8a16=metadata.use_int8_w8a16, dtype=c.dtype)

    config = try_get_optimal_moe_config(E, config_dtype, M)

    
    moe_metadata1 = MoEMetaData(
        top_k=metadata.top_k,
        topk_weights=None,
        topk_ids=metadata.topk_ids,
        sorted_token_ids=metadata.sorted_token_ids,
        expert_ids=metadata.expert_ids,
        num_tokens_post_padded=metadata.num_tokens_post_padded,
        config=config
    )
    moe_metadata2 = MoEMetaData(
        top_k=1,
        topk_weights=metadata.topk_weights,
        topk_ids=metadata.topk_ids,
        sorted_token_ids=metadata.sorted_token_ids,
        expert_ids=metadata.expert_ids,
        num_tokens_post_padded=metadata.num_tokens_post_padded,
        config=config
    )

    # TODO quantization support. How to get the scale for the intermediate result?

    intermediate_cache1 = torch.zeros([M, top_k, N], dtype=a.dtype, device=a.device)
    intermediate_cache2 = torch.zeros([M * top_k, N // 2], dtype=a.dtype, device=a.device)

    moe_gemm(a, w1, intermediate_cache1, moe_metadata1)
    silu_and_mul(intermediate_cache1.view(M * top_k, N), intermediate_cache2)

    moe_gemm(intermediate_cache2, w2, c, moe_metadata2)

    return c, intermediate_cache2


@pytest.mark.parametrize("M, N, K, top_k, E", [
    (1, 14336, 4096, 2, 8),
    # TODO doesn't work check k mask
    (16, 14336, 1, 2, 4),
    (256, 14336, 1, 2, 4),
    (2048, 14336, 1, 2, 4),
    # ------
    (1, 14336, 128, 2, 4),
    (16, 14336, 128, 1, 4),
    (16, 14336, 128, 1, 1),
    (64, 70, 128, 2, 8),
    (64, 30, 128, 2, 8),
    (64, 32, 128, 2, 8),
    (64, 7186, 128, 2, 8),
    (64, 3584, 128, 2, 8),
    (64, 1792, 128, 2, 8),
    (64, 64, 128, 2, 8),
])
# @pytest.mark.parametrize('routed_weight', [True, False])
@pytest.mark.parametrize('routed_weight', [False])
def test_correctness(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool,
                     dtype=torch.float32):
    torch.manual_seed(20)
    a, w1, w2, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,
                                     use_int8_w8a16=False, fp8_type=None,
                                     dtype=dtype)

    tri_out = e2e_moe(a, w1, w2, c, metadata)

    topk_ids = metadata.topk_ids
    topk_weights = metadata.topk_weights
    ref_out = torch.empty_like(c)

    ref_out = e2e_moe_ref(a, w1, w2, ref_out, M, E, top_k, N, metadata)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M, N, K, top_k, E", [
    (1, 14336, 4096, 2, 8),
    (2048, 14336, 4096, 2, 8),
    # TODO doesn't work check k mask
    (16, 14336, 1, 2, 4),
    (256, 14336, 1, 2, 4),
    (2048, 14336, 1, 2, 4),
    # ------
    (1, 14336, 128, 2, 4),
    (16, 14336, 128, 1, 4),
    (16, 14336, 128, 1, 1),
    (64, 70, 128, 2, 8),
    (64, 30, 128, 2, 8),
    (64, 32, 128, 2, 8),
    (64, 7186, 128, 2, 8),
    (64, 3584, 128, 2, 8),
    (64, 1792, 128, 2, 8),
    (64, 64, 128, 2, 8),
])
# @pytest.mark.parametrize('routed_weight', [True, False])
@pytest.mark.parametrize('routed_weight', [False])
def test_correctness_persistent(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool,
                     dtype=torch.float16):
    torch.manual_seed(20)
    a, w1, w2, intermediate, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,
                                     use_int8_w8a16=False, fp8_type=None,
                                     dtype=dtype, persistent=True)

    tri_out, tri_intermediate = e2e_moe_persistent(a, w1, w2, intermediate, c, metadata)

    topk_ids = metadata.topk_ids
    topk_weights = metadata.topk_weights
    ref_out = torch.empty_like(c)

    ref_out, ref_intermediate = e2e_moe_ref(a, w1, w2, ref_out, M, E, top_k, N, metadata)

    torch.testing.assert_close(tri_intermediate, ref_intermediate, atol=2e-2, rtol=2e-2)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=2e-2, rtol=2e-2)
    print("all tests pass")

# @pytest.mark.parametrize("M, N, K, top_k, E", [
#     (64, 14336, 4096, 2, 8),
#     (16, 14336, 1, 2, 4),
#     (1, 14336, 128, 2, 4),
#     (16, 14336, 128, 1, 4),
#     (16, 14336, 128, 1, 1),
#     (64, 7186, 128, 2, 8),
#     (64, 3584, 128, 2, 8),
#     (64, 1792, 128, 2, 8),
#     (64, 64, 128, 2, 8),
# ])
# @pytest.mark.parametrize('routed_weight', [True, False])
# @pytest.mark.parametrize('use_fp8_w8a8', [True])
# @pytest.mark.parametrize('use_silu_activation', [True, False])
# @pytest.mark.parametrize('fp8_type', [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])
# def test_correctness_fp8(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_silu_activation: bool,
#                          use_fp8_w8a8, fp8_type, dtype=torch.float16):
#     torch.manual_seed(20)
#     a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=use_fp8_w8a8,
#                                      use_int8_w8a16=False, fp8_type=fp8_type, use_silu_activation=use_silu_activation,
#                                      dtype=dtype)

#     tri_out = moe_gemm(a, b, c, metadata)

#     topk_ids = metadata.topk_ids
#     topk_weights = metadata.topk_weights
#     ref_out = torch.empty_like(c)
#     # Repeat a -> (M, top_k, K)
#     a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
#     # (M, top_k, N, K)
#     b_indexed = b.half()[topk_ids]
#     ref_out = torch.einsum("mek,menk->men", a_expanded.float(), b_indexed.float())

#     if routed_weight:
#         ref_out *= topk_weights.unsqueeze(-1)

#     ref_out = ref_out * metadata.b_descale[topk_ids].unsqueeze(-1)
#     ref_out = ref_out * metadata.a_descale
#     ref_out = ref_out.to(dtype)

#     if use_silu_activation:
#         ref_out = silu_and_mul_torch(ref_out.reshape(M * top_k, N)).to(dtype)
#     # Validate correctness
#     torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


# @pytest.mark.parametrize("M, N, K, top_k, E", [
#     (64, 14336, 4096, 2, 8),
#     (16, 14336, 1, 2, 4),
#     (1, 14336, 128, 2, 4),
#     (16, 14336, 128, 1, 4),
#     (16, 14336, 128, 1, 1),
#     (64, 7186, 128, 2, 8),
#     (64, 3584, 128, 2, 8),
#     (64, 1792, 128, 2, 8),
#     (64, 64, 128, 2, 8),
# ])
# @pytest.mark.parametrize('routed_weight', [True, False])
# @pytest.mark.parametrize('use_int8_w8a16', [True])
# @pytest.mark.parametrize('use_silu_activation', [True, False])
# def test_correctness_int8(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_silu_activation: bool,
#                           use_int8_w8a16, dtype=torch.float16):
#     torch.manual_seed(20)
#     a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,
#                                      use_int8_w8a16=use_int8_w8a16, fp8_type=None,
#                                      use_silu_activation=use_silu_activation, dtype=dtype)

#     tri_out = moe_gemm(a, b, c, metadata)

#     topk_ids = metadata.topk_ids
#     topk_weights = metadata.topk_weights
#     ref_out = torch.empty_like(c)
#     # Repeat a -> (M, top_k, K)
#     a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
#     # (M, top_k, N, K)
#     b_indexed = b[topk_ids]
#     ref_out = torch.einsum("mek,menk->men", a_expanded.to(torch.float32), b_indexed.to(torch.float32))
#     if routed_weight:
#         ref_out *= topk_weights.unsqueeze(-1)

#     ref_out = ref_out * metadata.b_descale[topk_ids, :]
#     ref_out = ref_out.to(dtype)

#     if use_silu_activation:
#         ref_out = silu_and_mul_torch(ref_out.reshape(M * top_k, N)).to(dtype)
#     # Validate correctness
#     torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


def get_configs():
    configs = [
        {"M": 64, "N": 256, "K": 128, "E": 8, "top_k": 2},
        {"M": 64, "N": 1792, "K": 1024, "E": 8, "top_k": 2},
        {"M": 64, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 128, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 1024, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 4096, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 64, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 128, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 256, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 512, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 1024, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 2048, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 4096, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["mistral"], model=args.model)
    moe_configs = []
    M = args.M if args.M else 4096  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]

        E = 8
        top_k = 2

        moe_configs.append((model_name, M, N1, K1, E, top_k))

    return moe_configs


def run_benchmark(custom, args):
    routed_weight = args.routed_weight
    use_int8_w8a16 = args.int8_w8a16
    use_fp8_w8a8 = args.fp8_w8a8
    dtype = arg_to_torch_dtype[args.dtype]
    fp8_type = arg_to_torch_dtype[args.fp8_type]

    x_names = ['M', 'N', 'K', 'E', 'top_k']

    if custom:
        assert args.M and args.N and args.K and args.E and args.top_k, \
            "Please provide M, N, K, E, top_k for custom runs."
        x_vals_list = [(args.M, args.N, args.K, args.E, args.top_k)]
    else:
        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ['model', 'M', 'N', 'K', 'E', 'top_k']
        else:
            configs = get_configs()
            x_vals_list = [(cfg['M'], cfg['N'], cfg['K'], cfg['E'], cfg['top_k']) for cfg in configs]

    line_names = ["ref", "fused"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_names, line_names=line_names,
        styles=[('red', '-'), ('green', '-')], plot_name='e2e-moe-benchmark', args={
                    'dtype': dtype, 'routed_weight': routed_weight, 'use_fp8_w8a8': use_fp8_w8a8, 'use_int8_w8a16':
                    use_int8_w8a16, 'fp8_type': fp8_type}
                )

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, N, K, E, top_k, dtype, routed_weight, provider, use_fp8_w8a8, use_int8_w8a16, fp8_type, model=None):
        a, w1, w2, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,
                                     use_int8_w8a16=False, fp8_type=None,
                                     dtype=dtype)

        if "fused" in provider:
            fn = lambda: e2e_moe(a, w1, w2, c, metadata)

        if "ref" in provider:
            fn = lambda: e2e_moe_ref(a, w1, w2, c, M, E, top_k, N, metadata)
        ms = triton.testing.do_bench(fn)

        return ms

    bench_moe_gemm.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE GEMM",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")
    available_models = get_available_models(model_families=["mistral"])  # Dynamically load model names
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument("-K", type=int, default=0, help="K dimension")
    parser.add_argument("-N", type=int, default=0, help="N dimension")
    parser.add_argument("-E", type=int, default=0, help="Number of experts")
    parser.add_argument("-top_k", type=int, default=0, help="top_k experts per token")
    parser.add_argument("-routed_weight", action='store_true', default=False)
    parser.add_argument("-int8_w8a16", action='store_true', default=False)
    parser.add_argument("-fp8_w8a8", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-fp8_type", default='e5m2fnuz')
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {
    'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, "e5m2fnuz": torch.float8_e5m2fnuz, "e4m3fnuz":
    torch.float8_e4m3fnuz
}


def main():
    # args = parse_args()
    # custom_config = False
    # # If user provides all M,K,N,E,top_k we consider it custom
    # if args.M and args.K and args.N and args.E and args.top_k:
    #     custom_config = True

    
    # run_benchmark(custom_config, args)

    # test_correctness_persistent(4096, 4096, 4096, 2, 4, False, torch.float32)


if __name__ == '__main__':
    sys.exit(main())
