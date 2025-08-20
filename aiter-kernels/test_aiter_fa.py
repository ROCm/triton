#import hip
#hip.hip.hipInit(0)

import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import logging
import numpy as np
import math
from einops import repeat
import os


ATOL_fp8 = 2.5e-1
RTOL_fp8 = 2.5e-1


def fp8_assert_close(
    tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5
):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))

    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)

    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100

    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True

    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()

    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    print(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )
    return False


def ref_attn(q, k, v, sm_scale, dtype=torch.float32):
    #scale_block = metadata.scale_block

    #q_scale = fp8e8m0_to_float32(q_scale).repeat_interleave(scale_block, dim=-1)
    #k_scale = fp8e8m0_to_float32(k_scale).repeat_interleave(scale_block, dim=-1)
    #v_scale = fp8e8m0_to_float32(v_scale).repeat_interleave(scale_block, dim=-1)

    #q = q.to(dtype) * q_scale.to(dtype)
    #k = k.to(dtype) * k_scale.to(dtype)
    #v = v.to(dtype) * v_scale.to(dtype)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float())
    if sm_scale:
      scores *= sm_scale

    p = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bkhd->bqhd', p, v.float())
    return ref_out


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
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
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    start_m,
    seqlen_k,
    seqlen_q,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    descale_q,
    descale_k,
    descale_v,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator

    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.

            # remove the old if condition
            # if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            # Though this will unconditionally compute mask_partial at runtime,
            # the causal for loop does not have the if-else block any more, which
            # helps instruction scheduling and register pressure.
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + OFFS_N[None, :]
            mask_partial = size_n < boundary_m[:, None]
            mask = tl.where(bound_cond, mask_partial, mask)

        # compute masks
        q_mask = OFFS_M[:, None] < seqlen_q
        k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
        p_mask = q_mask & k_mask

        # -- compute qk ----
        qk += tl.dot(q, k) * descale_q * descale_k

        qk = tl.where(mask, qk, float("-inf"))

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # scale and subtract max
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]
        v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
        acc += (
            tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v
        )

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    descale_q_ptr: torch.Tensor,
    descale_k_ptr: torch.Tensor,
    descale_v_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    sm_scale,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BATCH,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = tl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    # num blocks along seqlen

    off_q_head = wid % NUM_Q_HEADS
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    stride_qz = tl.cast(stride_qz_in, tl.int64)
    stride_qh = tl.cast(stride_qh_in, tl.int64)
    stride_qm = tl.cast(stride_qm_in, tl.int64)
    stride_qk = tl.cast(stride_qk_in, tl.int64)
    stride_kz = tl.cast(stride_kz_in, tl.int64)
    stride_kh = tl.cast(stride_kh_in, tl.int64)
    stride_kn = tl.cast(stride_kn_in, tl.int64)
    stride_kk = tl.cast(stride_kk_in, tl.int64)
    stride_vz = tl.cast(stride_vz_in, tl.int64)
    stride_vh = tl.cast(stride_vh_in, tl.int64)
    stride_vn = tl.cast(stride_vn_in, tl.int64)
    stride_vk = tl.cast(stride_vk_in, tl.int64)
    stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
    stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
    stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
    stride_oz = tl.cast(stride_oz_in, tl.int64)
    stride_oh = tl.cast(stride_oh_in, tl.int64)
    stride_om = tl.cast(stride_om_in, tl.int64)
    stride_on = tl.cast(stride_on_in, tl.int64)

    seqlen_q = SEQLEN_Q
    seqlen_k = SEQLEN_K

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    if grp_sz != 1:  # Grouped Query Attention
        off_k_head = off_q_head // grp_sz
    else:
        off_k_head = off_q_head

    # q,k,v offsets
    q_offs = (
        off_z * stride_qz
        + off_q_head * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q_ptrs = q_ptr + q_offs

    k_offs = (
        off_z * stride_kz
        + off_k_head * stride_kh
        + offs_d[:, None] * stride_kk
        + offs_n[None, :] * stride_kn
    )
    k_ptrs = k_ptr + k_offs

    v_offs = (
        off_z * stride_vz
        + off_k_head * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )
    v_ptrs = v_ptr + v_offs

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if BLOCK_DMODEL == BLOCK_DMODEL_POW2:
        q_mask = offs_m[:, None] < seqlen_q
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    descale_q = tl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
    descale_k = tl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
    descale_v = tl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # if CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
 
    # Padding on Q does not need to be masked in the FA loop.
    masked_blocks = padded_block_k

    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            seqlen_k,
            seqlen_q,
            block_min,
            block_max,
            0,
            0,
            0,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            MASK_STEPS=False,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            FP8_MAX=FP8_MAX,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks, if any, are full / not masked.
    if masked_blocks > 0:
        offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            seqlen_k,
            seqlen_q,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            MASK_STEPS=True,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            FP8_MAX=FP8_MAX,
        )
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    overflow_size = end_m_idx - seqlen_q

    # write back O
    offs_out = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL_POW2], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


def _flash_attn_forward(
    q,
    k,
    v,
    softmax_scale,
    max_seqlen_q,
    max_seqlen_k,
    config,
    descale_q=None,
    descale_k=None,
    descale_v=None):

    # FP8
    FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max

    o = torch.zeros(q.shape, dtype=torch.float32)

    # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    num_k_heads = k.shape[2]
    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_DMODEL_POW2 = max(BLOCK_DMODEL_POW2, 16)

    grid = lambda META: (
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
    )

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    o = o.cuda()
    descale_q = descale_q.cuda()
    descale_k = descale_k.cuda()
    descale_v = descale_v.cuda()
    

    handle = _attn_fwd[grid](
        q,
        k,
        v,
        descale_q,
        descale_k,
        descale_v,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        descale_q.stride(0) if descale_q is not None else 0,
        descale_k.stride(0) if descale_k is not None else 0,
        descale_v.stride(0) if descale_v is not None else 0,
        *o_strides,
        softmax_scale,
        SEQLEN_Q=max_seqlen_q,
        SEQLEN_K=max_seqlen_k,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=head_sz,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        FP8_MAX=FP8_MAX,
        BATCH=batch,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"]
    )
    
    dump_ir = config['dump-ir']
    if dump_ir != 'none':
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'{handle.name}.{dump_ir}'
        with open(os.path.join(curr_dir, filename), "w") as file:
            file.write(handle.asm[dump_ir])

    return o.cpu()


def _cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype,
    layout,
    clamp_val=1e-9):
    """
    Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.
    Args:
        - x (torch.Tensor): shape [batch, seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): FP8 tensor with the same shape as x
        - descale_factor (torch.Tensor): tensor of shape [batch, 1, heads, 1]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )
    reduce_dims = (1, 3)  # seq_len and dim dimensions

    # Compute the absolute max along reduce_dims, clamped to avoid 0-scale
    x_abs_max = x.abs().amax(dim=reduce_dims)
    x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

    # Unsqueeze back to a shape suitable for broadcast
    unsqueeze_dims = sorted(reduce_dims)
    for d in unsqueeze_dims:
        x_abs_max = x_abs_max.unsqueeze(d)

    # compute scale and descale
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / x_abs_max
    descale_factor = x_abs_max / fp8_max

    # cast to FP8, optionally setting requires_grad
    x_fp8 = (x * scale).to(fp8_dtype)

    return x_fp8, descale_factor


def apply_flash_attn_fp8(
    q,
    k,
    v,
    softmax_scale,
    config):
  
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_og = q.size(3)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
  
    # cast input to fp8
    #fp8_dtype = torch.float8_e4m3fn
    fp8_dtype = torch.float8_e5m2
    q_fp8, descale_q = _cast_to_fp8(q, fp8_dtype, "bshd")
    k_fp8, descale_k = _cast_to_fp8(k, fp8_dtype, "bshd")
    v_fp8, descale_v = _cast_to_fp8(v, fp8_dtype, "bshd")
  
    out_padded = (
        _flash_attn_forward(
            q_fp8,
            k_fp8,
            v_fp8,
            softmax_scale,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            config=config,
        )
    )
  
    out = out_padded[..., :head_size_og]
    result = [out]

    return result[0] if len(result) == 1 else tuple(result)


def generate_tensor(dtype, B, H, N, D, min=1, max=3):
    if dtype == 'float4':
        a = MXFP4Tensor(data = torch.randint(min, max, (B, H, N, D)))
    else:
        torch_type = getattr(torch, dtype)
        a = (torch.randint(min, max, (B, H, N, D))).to(torch_type)
    return a


def test_mha(config):
    BATCH = config["BATCH"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    HEAD_SZ = config["HEAD_SZ"]
    verbose = config["verbose"]
    dtype = "float16"
    
    torch.manual_seed(0)
    q = generate_tensor(dtype, BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, min=1, max=3)
    k = generate_tensor(dtype, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, min=1, max=3)
    v = generate_tensor(dtype, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, min=1, max=3)

    softmax_scale = q.shape[-1] ** (-0.5)
    triton_out = apply_flash_attn_fp8(
        q,
        k,
        v,
        softmax_scale,
        config)

    torch_out = ref_attn(q, k, v, softmax_scale)
    is_ok = fp8_assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=ATOL_fp8, rtol=RTOL_fp8
    )
    
    if is_ok:
      print("Passed ✅")
      if verbose:
        #print(f'{triton_out=}')
        pass
    else:
      print("Fail ❌")
      if verbose:
        print(f'{triton_out=}')
        print(f'{torch_out=}')


def generate_configs():
    MAX_BATCH = 64
    base_configs = [
        #{"BATCH": 1, "NUM_Q_HEADS": 1, "NUM_K_HEADS": 1, "SEQLEN_Q": 1024, "SEQLEN_K": 1024, "HEAD_SZ": 128, "BLOCK_M": 32, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},
        #{"BATCH": 1, "NUM_Q_HEADS": 1, "NUM_K_HEADS": 1, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 128, "BLOCK_M": 32, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},
        {"BATCH": 1, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 128, "BLOCK_M": 32, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},
        {"BATCH": MAX_BATCH, "NUM_Q_HEADS": 1, "NUM_K_HEADS": 1, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 128, "BLOCK_M": 32, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},

        # WMMA K-padding is not implemented, Triton compiler error (use types: "float8_e5m2", "float4")
        {"BATCH": 1, "NUM_Q_HEADS": 1, "NUM_K_HEADS": 1, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 64, "BLOCK_M": 32, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},
        {"BATCH": MAX_BATCH, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 64, "BLOCK_M": 32, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": 1},
    ]
    return base_configs


if __name__ == "__main__":
  configs = generate_configs()
  for config in configs:
    config["verbose"] = True
    #config["dump-ir"] = "ttgir"
    config["dump-ir"] = "amdgcn"
    print(f'testing: {config}')
    test_mha(config)
