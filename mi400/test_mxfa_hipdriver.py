# References:
# - triton
#   - python/test/unit/language/test_matmul.py
# - aiter (commit: aab726524b952a23cb71577bd48c91a2db21c983)
#   - aiter/ops/triton/mha.py
#   - aiter/test_mha_common.py

import os

if 'FFM_PATH' in os.environ:
    import hip
    hip.hip.hipInit(0)

import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import argparse
import math
from einops import repeat

ATOL_fp8 = 2.5e-1
RTOL_fp8 = 2.5e-1


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
    q_scale,
    k_scale_ptrs,
    v_scale_ptrs,
    stride_k_scale_n,
    stride_v_scale_n,
    block_min,
    block_max,
    q_type: tl.constexpr,
    kv_type: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
    DISABLE_MASKING: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    KV_PACK_DIV: tl.constexpr = 2 if kv_type == 'e2m1' else 1

    for _ in range(block_min, block_max, BLOCK_N):
        k = tl.load(k_ptrs)
        k_scale = tl.load(k_scale_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot_scaled(q, q_scale, q_type, k, k_scale, kv_type)

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # scale and subtract max
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        # update output accumulator
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs)
        v_scale = tl.load(v_scale_ptrs)

        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc += tl.dot_scaled(p.to(tl.float8e4nv), None, 'e4m3', v, v_scale, kv_type)

        k_ptrs += BLOCK_N * stride_kn
        k_scale_ptrs += BLOCK_N * stride_k_scale_n

        v_ptrs += (BLOCK_N // KV_PACK_DIV) * stride_vk
        v_scale_ptrs += (BLOCK_N // 32) * stride_v_scale_n

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    q_scale_ptr: torch.Tensor,
    k_scale_ptr: torch.Tensor,
    v_scale_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_q_scale_z,
    stride_q_scale_h,
    stride_q_scale_m,
    stride_q_scale_k,
    stride_k_scale_z,
    stride_k_scale_h,
    stride_k_scale_n,
    stride_k_scale_k,
    stride_v_scale_z,
    stride_v_scale_h,
    stride_v_scale_n,
    stride_v_scale_k,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    sm_scale,
    q_type: tl.constexpr,
    kv_type: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BATCH,
    DISABLE_MASKING: tl.constexpr,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    seqlen_q = SEQLEN_Q
    seqlen_k = SEQLEN_K

    KV_PACK_DIV: tl.constexpr = 2 if kv_type == 'e2m1' else 1

    # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    wid = tl.program_id(0)
    n_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # offsets
    off_q_head = wid % NUM_Q_HEADS
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n_packed = tl.arange(0, BLOCK_N // KV_PACK_DIV)
    offs_n_scale = tl.arange(0, BLOCK_N // 32)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d_packed = tl.arange(0, BLOCK_DMODEL // KV_PACK_DIV)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL // 32)
    off_k_head = off_q_head

    # q       [BLOCK_M, BLOCK_DMODEL]
    # q_scale [BLOCK_M, BLOCK_DMODEL / 32]
    q_offs = (off_z * stride_qz + off_q_head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q_ptrs = q_ptr + q_offs

    q_scale_offs = (off_z * stride_q_scale_z + off_q_head * stride_q_scale_h + offs_m[:, None] * stride_q_scale_m +
                    offs_d_scale[None, :] * stride_q_scale_k)
    q_scale_ptrs = q_scale_ptr + q_scale_offs

    # k       [BLOCK_DMODEL / KV_PACK_DIV, BLOCK_N]
    # k_scale [BLOCK_N, BLOCK_DMODEL / 32]
    k_offs = (off_z * stride_kz + off_k_head * stride_kh + offs_d_packed[:, None] * stride_kk +
              offs_n[None, :] * stride_kn)
    k_ptrs = k_ptr + k_offs

    k_scale_offs = (off_z * stride_k_scale_z + off_k_head * stride_k_scale_h + offs_n[:, None] * stride_k_scale_n +
                    offs_d_scale[None, :] * stride_k_scale_k)
    k_scale_ptrs = k_scale_ptr + k_scale_offs

    # v       [BLOCK_N / KV_PACK_DIV, BLOCK_DMODEL]
    # v_scale [BLOCK_DMODEL, BLOCK_N / 32]
    v_offs = (off_z * stride_vz + off_k_head * stride_vh + offs_n_packed[:, None] * stride_vn +
              offs_d[None, :] * stride_vk)
    v_ptrs = v_ptr + v_offs

    v_scale_offs = (off_z * stride_v_scale_z + off_k_head * stride_v_scale_h + offs_d[:, None] * stride_v_scale_k +
                    offs_n_scale[None, :] * stride_v_scale_n)
    v_scale_ptrs = v_scale_ptr + v_scale_offs

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q_mask = True if DISABLE_MASKING else offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q_scale = tl.load(q_scale_ptrs, mask=q_mask, other=0x7F)

    block_min = 0
    block_max = n_blocks * BLOCK_N
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn, q_scale, k_scale_ptrs,
                                    v_scale_ptrs, stride_k_scale_n, stride_v_scale_n, block_min, block_max, q_type,
                                    kv_type, BLOCK_M, BLOCK_N, sm_scale, DISABLE_MASKING)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M

    # write back O
    overflow_size = end_m_idx - seqlen_q

    offs_out = (off_z * stride_oz + off_q_head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)

    out_mask = True if DISABLE_MASKING else out_mask
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


def attn_fwd(q, k, v, q_scale, k_scale, v_scale, config, args):
    softmax_scale = q.shape[-1]**(-0.5)

    o = torch.zeros_like(q, dtype=torch.float32)

    batch, seqlen_q, num_q_heads, head_sz = q.shape
    num_k_heads = k.shape[2]
    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    q_scale_strides = (q_scale.stride(0), q_scale.stride(2), q_scale.stride(1), q_scale.stride(3))
    k_scale_strides = (k_scale.stride(0), k_scale.stride(2), k_scale.stride(1), k_scale.stride(3))
    v_scale_strides = (v_scale.stride(0), v_scale.stride(2), v_scale.stride(1), v_scale.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    q_scale = q_scale.cuda()
    k_scale = k_scale.cuda()
    v_scale = v_scale.cuda()
    o = o.cuda()

    q_type = args.q_type
    kv_type = args.kv_type

    grid = lambda META: (batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]), )

    handle = _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *q_scale_strides,
        *k_scale_strides,
        *v_scale_strides,
        *o_strides,
        softmax_scale,
        q_type,
        kv_type,
        SEQLEN_Q=q.shape[1],
        SEQLEN_K=k.shape[1],
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=head_sz,
        BATCH=batch,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        DISABLE_MASKING=args.disable_masking,
        num_warps=config["NUM_WARPS"],
        num_stages=config["NUM_STAGES"],
    )

    if args.dump_ir != 'none':
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'attn_kernel.{args.dump_ir}'
        with open(os.path.join(curr_dir, filename), "w") as file:
            file.write(handle.asm[args.dump_ir])

    return o.cpu()


def attn_ref(q, k, v, q_scale, k_scale, v_scale):
    dtype_og = q.dtype

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og)


def test_mha(config, args):
    BATCH = config['BATCH']
    SEQLEN_Q = config['SEQLEN_Q']
    SEQLEN_K = config['SEQLEN_K']
    NUM_Q_HEADS = config['NUM_Q_HEADS']
    NUM_K_HEADS = config['NUM_K_HEADS']
    HEAD_SZ = config['HEAD_SZ']

    def create_operand(dtype: str, b: int, s: int, h: int, d: int, pack_dim: int = -1):
        if dtype == 'e4m3':
            v = torch.randint(20, 40, (b, s, h, d), dtype=torch.uint8)
            v_ref = v.view(torch.float8_e4m3fn).to(torch.float32)
        elif dtype == 'e5m2':
            v = torch.randint(20, 40, (b, s, h, d), dtype=torch.uint8)
            v_ref = v.view(torch.float8_e5m2).to(torch.float32)
        else:
            assert dtype == 'e2m1'
            assert pack_dim >= 0
            v_mxfp4 = MXFP4Tensor(size=(b, s, h, d)).random()
            v = v_mxfp4.to_packed_tensor(pack_dim)
            v_ref = v_mxfp4.to(torch.float32)
        return v, v_ref

    def create_scale(b: int, s: int, h: int, d: int, scale_dim: int):
        size = [b, s, h, d]
        size[scale_dim] //= 32
        scale = MXScaleTensor(size=tuple(size)).random(high=24)
        scale_ref = scale.to(torch.float32).repeat_interleave(32, dim=scale_dim)
        return scale.data, scale_ref

    torch.random.manual_seed(0)
    q, q_ref = create_operand(args.q_type, BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ)
    k, k_ref = create_operand(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, pack_dim=3)
    v, v_ref = create_operand(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, pack_dim=1)
    q_scale, q_scale_ref = create_scale(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, scale_dim=3)
    k_scale, k_scale_ref = create_scale(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, scale_dim=3)
    v_scale, v_scale_ref = create_scale(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, scale_dim=1)

    triton_out = attn_fwd(q, k, v, q_scale, k_scale, v_scale, config, args)
    torch_out = attn_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)

    try:
        torch.testing.assert_close(triton_out, torch_out, atol=ATOL_fp8, rtol=RTOL_fp8)
    except Exception as err:
        print("❌ Triton and Torch differ")
        print(err)
        if args.verbose:
            print(f"{triton_out=}")
            print(f"{torch_out=}")
        return

    print("✅ Triton and Torch match")


def generate_configs(args):
    MAX_BATCH = 64
    num_stages = args.num_stages if args.num_stages != -1 else 3
    BLOCK_M_FOR_HEAD_SIZE_128 = 128 if args.kv_type == 'e4m3' or args.kv_type == "e5m2" else 256
    base_configs = [
        # HEAD_SZ == 128
        {
            "BATCH": 1, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 128,
            "BLOCK_M": BLOCK_M_FOR_HEAD_SIZE_128, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "NUM_STAGES": num_stages
        },
        {
            "BATCH": MAX_BATCH, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 128,
            "BLOCK_M": BLOCK_M_FOR_HEAD_SIZE_128, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "NUM_STAGES": num_stages
        },
        # HEAD_SZ == 64
        {
            "BATCH": 1, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 64,
            "BLOCK_M": 512, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": num_stages
        },
        {
            "BATCH": MAX_BATCH, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 64,
            "BLOCK_M": 512, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": num_stages
        },
    ]
    return base_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-type", choices=["e5m2", "e4m3"], default="e4m3",
                        help="data type for K and V (default e4m3)")
    parser.add_argument("--kv-type", choices=["e5m2", "e4m3", "e2m1"], default="e4m3",
                        help="data type for K and V (default e2m1)")
    parser.add_argument("--dump-ir", choices=['none', 'ttir', 'ttgir', 'llir', 'amdgcn'], default="none",
                        help="dump IR format")
    parser.add_argument("-c", "--case", type=int, required=True, help='case id')
    parser.add_argument("--num-stages", type=int, default=-1, required=False, help='num stages')
    parser.add_argument("-m", "--disable-masking", action='store_true', help='use masked loads')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    args = parser.parse_args()

    print(f'{args.q_type=}; {args.kv_type=}; {args.disable_masking=}')
    print(f'Testing with {ATOL_fp8=}; {RTOL_fp8=}')

    configs = generate_configs(args)
    config = configs[args.case]

    print(f'{config=}')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f'mxfa-curr-config.txt'
    with open(os.path.join(curr_dir, filename), "w") as file:
        file.write(f'{config=}\n')
        file.write(f'{args.q_type=}\n')
        file.write(f'{args.kv_type=}\n')
        file.write(f'{args.disable_masking=}\n')

    test_mha(config, args)
