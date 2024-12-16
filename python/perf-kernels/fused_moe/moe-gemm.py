import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional, Tuple
import os
import json
import functools


@triton.jit
def moe_gemm_kernel(
    A,
    B,
    Out,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # MUL_ROUTED_WEIGHT: tl.constexpr
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    # Here we assume that valid tokens are in the range [0, M).
    token_mask = (offs_token >= 0) & (offs_token < M)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # TODO why is there // top_k here???
    a_ptrs = A + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Masking ensures we don't load from invalid tokens or indices
        a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Load the MoE weights using the token_mask to avoid invalid memory accesses.
    # TODO enable the weight
    # moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    accumulator = accumulator
    # accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = Out + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    # TODO why it doesn't do accumulation over experts???
    # tl.store(out_ptrs, accumulator, mask=c_mask)
    tl.atomic_add(out_ptrs, accumulator, mask=c_mask)


def _moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, block_size: int, sorted_token_ids: torch.Tensor,
                          expert_ids: torch.Tensor, num_tokens_post_pad: torch.Tensor) -> None:
    """
    Rearrange token-expert assignments so that each expert's token IDs
    are grouped in multiples of `block_size`.

    Parameters
    ----------
    topk_ids : torch.Tensor
        Shape [M, top_k], expert IDs for each token.
    num_experts : int
        Total number of experts (E).
    block_size : int
        Alignment boundary.
    sorted_token_ids : torch.Tensor (to be written in-place)
        1D array to store the re-ordered token IDs (length >= M*top_k).
    expert_ids : torch.Tensor (to be written in-place)
        1D array to store the re-ordered expert IDs (same length as sorted_token_ids).
    num_tokens_post_pad : torch.Tensor (scalar, to be written in-place)
        Will contain the total (padded) number of token-expert entries after alignment.
    """
    M, top_k = topk_ids.shape

    # 1) Build a list of tokens for each expert
    expert_to_tokens = [[] for _ in range(num_experts)]
    # For each token, for each selected expert, we append (token_id, expert)
    for token_id in range(M):
        for j in range(top_k):
            e_id = topk_ids[token_id, j].item()
            expert_to_tokens[e_id].append(token_id)

    # 2) Reorder tokens block by block, padding if needed
    reordered_token_ids = []
    reordered_expert_ids = []

    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)

        n_blocks = ((num_tokens + block_size - 1) // block_size)
        # If not a multiple of block_size, pad up to the next multiple
        padded_size = n_blocks * block_size

        # Reorder all actual tokens for expert e_id
        reordered_token_ids.extend(tokens_for_expert)
        # reordered_expert_ids.extend([e_id]*num_tokens)
        reordered_expert_ids.extend([e_id]*n_blocks)

        # Pad with dummy token_id = -1 (or any sentinel), if needed
        if padded_size > num_tokens:
            pad_count = padded_size - num_tokens
            reordered_token_ids.extend([-1] * pad_count)

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)
    # 3) Write data back in-place into sorted_token_ids, expert_ids
    #    sorted_token_ids / expert_ids are expected to have shape >= [M*top_k]
    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype,
                                                   device=sorted_token_ids.device)
    expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)

    # Fill remainder with -1 if these arrays are bigger than total_length
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = -1
    if expert_length < expert_ids.numel():
        expert_ids[expert_length:] = -1

    # 4) Set num_tokens_post_pad to the new total length
    num_tokens_post_pad.fill_(token_length)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1), ), dtype=torch.int32,
                             device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    # TODO do we need to predefine the vars? may be they can be defined in the function and reurned
    _moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

    return sorted_ids, expert_ids, num_tokens_post_pad


def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int, dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype)

    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
) -> Dict[str, int]:
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}
    return config


def try_get_optimal_moe_config(
    b_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
):
    E, K, N = b_shape
    configs = get_moe_configs(E, N, dtype)

    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Else use the default config
        config = get_default_config(M, E, N, K, top_k, dtype, is_marlin)
    return config


def get_config_dtype_str(dtype: torch.dtype, use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def moe_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk_weights: torch.Tensor,
             topk_ids: torch.Tensor) -> None:
    config_dtype = None
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        b.shape,
        topk_ids.shape[1],
        config_dtype,
    )
    M, top_k = topk_ids.shape
    E, _, _ = b.shape

    config = get_config_func(M)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)
    print(sorted_token_ids)
    print(expert_ids)

    EM = num_tokens_post_padded.item()
    _, N, K = b.shape
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    moe_gemm_kernel[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(0),
                          c.stride(1), top_k, topk_weights, sorted_token_ids, expert_ids, EM, N,
                          K, M, **config)
    return c


def input_helper(M: int, K: int, N: int, top_k: int, E: int):
    """
    Parameters:
    - M: number of tokens after sorting and routing (this may be total tokens * top_k if each token is duplicated for each expert)
    - K: input feature dimension
    - N: output feature dimension
    - top_k: number of experts per token
    - E: number of experts

    Returns:
    (a, b, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded)
    ready to be passed into moe_gemm.
    """

    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((E, N, K), dtype=torch.float32, device='cuda')
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    # config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
    #                                     use_int8_w8a16=use_int8_w8a16,
    #                                     dtype=hidden_states.dtype)
    config_dtype = None

    values = torch.randn(M, E, device='cuda')

    # Option 1: Sorted values
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    return a, b, c, topk_weights, topk_ids


@pytest.mark.parametrize("M, K, N, top_k, E", [
    (16, 64, 256, 2, 4),
    (64, 32, 128, 1, 2),
    (256, 128, 512, 2, 8)
])
def test_correctness(M: int, K: int, N: int, top_k: int, E: int):
    a, b, c, topk_weights, topk_ids = input_helper(M, K, N, top_k, E)

    # TODO weights and other features
    tri_out = moe_gemm(a, b, c, topk_weights, topk_ids)

    ref_out = torch.empty_like(c)
    for token_id in range(M):
        expert_ids = topk_ids[token_id, :]
        experts = b[expert_ids, :, :]
        token = a[token_id, :]
        ref_out[token_id, :] = torch.einsum('k,enk->n', token, experts)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)
