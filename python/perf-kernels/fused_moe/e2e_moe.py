import triton
import torch
import triton.language as tl
import pytest
import os
import functools
import argparse
import sys
from moe_gemm import moe_align_block_size, silu_and_mul, try_get_optimal_moe_config, get_config_dtype_str, quantize_tensor, moe_gemm, MetaData as MoEMetaData


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

    def __init__(self, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config):
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
    EVEN_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr, # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr, # outputs (EM, BLOCK_SIZE_K2)
    GROUP_SIZE_M: tl.constexpr,
    # NUM_XCDS: tl.constexpr,
    # GRID_MN: tl.constexpr
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
    # TODO NUM_XCD
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

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

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K2)):
        if use_int8_w8a16:
            w2_scale_ptrs = W2_scale + off_experts * stride_w2se + (offs_k2 + k * BLOCK_SIZE_K2)[None, :] * stride_w2sk
            w2_scale = tl.load(w2_scale_ptrs)

        # TODO can we do mask free load?????
        # if EVEN_N:
        #     w2 = tl.load(w2_ptrs)
        # else:
        w2 = tl.load(w2_ptrs, mask=(offs_w2n[:, None] < N) & (offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K, other=0.0)

        if use_int8_w8a16:
            out = tl.dot(acc, w2.to(a.type))
        else:
            out = tl.dot(acc, w2,)

        # TODO check do we need two of this?
        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
            out = out * moe_weight[:, None]

        if use_int8_w8a16:
            out = (out * w2_scale)
        elif use_fp8_w8a8:
            out = (out * acc_scale * w2_scale)

        # # atomic add
        c_mask = token_mask[:, None] & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)

        # TODO check scope
        # tl.store(out_ptrs, out, mask=c_mask)
        tl.atomic_add(out_ptrs, out, mask=c_mask, sem="relaxed")

        w2_ptrs += BLOCK_SIZE_K2 * stride_w2k
        out_ptrs += BLOCK_SIZE_K2


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

    BLOCK_SIZE_K1 = config["BLOCK_SIZE_K"]
    # TODO tune
    BLOCK_SIZE_K2 = 128

    EVEN_K = K % BLOCK_SIZE_K1 == 0
    EVEN_N = N % config["BLOCK_SIZE_N"] == 0
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    stride_cm = c.stride(1)

    e2e_moe_kernel[grid](a, w1, w2, c, a_descale, w1_descale, w2_descale, a.stride(0), a.stride(1), w1.stride(0), w1.stride(1),
                          w1.stride(2), w2.stride(0), w2.stride(2), w2.stride(1), stride_cm, stride_w1se, stride_w1sn, stride_w2se, stride_w2sk, top_k, topk_weights,
                          sorted_token_ids, expert_ids, EM, N, K, EVEN_K, EVEN_N, MUL_ROUTED_WEIGHT=topk_weights is not None,
                          use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
                          BLOCK_SIZE_K1=BLOCK_SIZE_K1, BLOCK_SIZE_K2=BLOCK_SIZE_K2, GROUP_SIZE_M=config["GROUP_SIZE_M"]
                          )
    return c


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


def input_helper(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_fp8_w8a8: bool,
                 use_int8_w8a16: bool, fp8_type, dtype):
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    w1 = torch.randn((E, N, K), dtype=dtype, device='cuda')
    w2 = torch.randn((E, K, N // 2), dtype=dtype, device='cuda')

    out = torch.zeros((M, top_k, K), dtype=dtype, device='cuda')

    values = torch.randn(M, E, device='cuda')

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, dtype=dtype)
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        E,
        config_dtype,
    )
    config = get_config_func(M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    metadata = MetaData(topk_weights if routed_weight else None, topk_ids, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, config)

    if use_fp8_w8a8 or use_int8_w8a16:
        a, w1, w2 = quantize_input(a, w1, w2, use_fp8_w8a8, use_int8_w8a16, metadata, fp8_type)

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


def e2e_moe_ref(a, w1, w2, c, M, top_k, N, metadata: MetaData):
    moe_metadata1 = MoEMetaData(topk_weights=metadata.topk_weights,
    topk_ids=metadata.topk_ids,
    sorted_token_ids=metadata.sorted_token_ids,
    expert_ids=metadata.expert_ids,
    num_tokens_post_padded=metadata.num_tokens_post_padded,
    config=metadata.config)
    # TODO quantization support. How to get the scale for the intermediate result?

    intermediate_cache1 = torch.zeros([M, top_k, N], dtype=a.dtype, device=a.device)
    intermediate_cache2 = torch.zeros([M* top_k, N // 2], dtype=a.dtype, device=a.device)

    moe_gemm(a, w1, intermediate_cache1, moe_metadata1)
    silu_and_mul(intermediate_cache1.view(M * top_k, N), intermediate_cache2)
    moe_gemm(intermediate_cache2, w2, c, moe_metadata1)

    return c


@pytest.mark.parametrize("M, N, K, top_k, E", [
    (64, 14336, 4096, 2, 8),
    # TODO doesn't work check k mask
    # (16, 14336, 1, 2, 4),
    # (256, 14336, 1, 2, 4),
    # (2048, 14336, 1, 2, 4),
    # ------
    # (1, 14336, 128, 2, 4),
    # (16, 14336, 128, 1, 4),
    # (16, 14336, 128, 1, 1),
    # (64, 70, 128, 2, 8),
    # (64, 30, 128, 2, 8),
    # (64, 32, 128, 2, 8),
    # (64, 7186, 128, 2, 8),
    # (64, 3584, 128, 2, 8),
    # (64, 1792, 128, 2, 8),
    # (64, 64, 128, 2, 8),
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
    # Repeat a -> (M, top_k, K)
    # a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    # # (M, top_k, N, K)
    # w1_indexed = w1[topk_ids]
    # w2_indexed = w2[topk_ids]

    # ref_out = torch.einsum("mek,menk->men", a_expanded, w1_indexed)
    # if routed_weight:
    #     ref_out *= topk_weights.unsqueeze(-1)

    # ref_out = silu_and_mul_torch(ref_out.reshape(M * top_k, N)).to(dtype).view(M, top_k, N // 2)
    # ref_out = torch.einsum("men,mekn->mek", ref_out, w2_indexed)
    # if routed_weight:
    #     ref_out *= topk_weights.unsqueeze(-1)

    ref_out = e2e_moe_ref(a, w1, w2, c, M, top_k, N, metadata)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


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


@pytest.mark.parametrize("M, N, top_k", [
    (64, 14336, 2),
    (16, 14336, 2),
    (1, 14336, 2),
    (16, 14336, 1),
    (64, 7186, 2),
    (64, 3584, 2),
    (64, 1792, 2),
    (64, 64, 2),
])
def test_silu_correctness(M: int, N: int, top_k: int, dtype=torch.float16):
    torch.manual_seed(20)
    a = torch.randn((M * top_k, N), dtype=dtype, device='cuda')
    out = torch.zeros((M * top_k, N // 2), dtype=dtype, device='cuda')

    tri_out = silu_and_mul(a, out)
    ref_out = silu_and_mul_torch(a).to(dtype)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


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

        N2 = config["hidden_size"]
        K2 = config["intermediate_size"] // 2

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

    line_names = ['ref', 'fused']

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
            fn = lambda: e2e_moe_ref(a, w1, w2, c, M, top_k, N, metadata)
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
    args = parse_args()
    custom_config = False
    # If user provides all M,K,N,E,top_k we consider it custom
    if args.M and args.K and args.N and args.E and args.top_k:
        custom_config = True
    run_benchmark(custom_config, args)


if __name__ == '__main__':
    sys.exit(main())
