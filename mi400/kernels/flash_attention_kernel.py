import triton
import triton.language as tl
import torch
import numpy as np
from kernels.test_common import allclose_numpy


def soft(x, axis=-1):
    """
    Calculates the softmax of an array.

    Args:
        x (np.ndarray): Input array.
        axis (int): Axis along which to calculate softmax.

    Returns:
        np.ndarray: Softmax of the input array.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attn(query, key, value):
    """
    Calculates the scaled dot-product attention.

    Args:
        query (np.ndarray): Query vectors. Shape: (batch_size, num_queries, query_dim).
        key (np.ndarray): Key vectors. Shape: (batch_size, num_key, key_dim).
        value (np.ndarray): Value vectors. Shape: (batch_size, num_value, value_dim).
        mask (np.ndarray, optional): Mask for the attention. Shape: (batch_size, num_queries, num_key).

    Returns:
        np.ndarray: Output vectors. Shape: (batch_size, num_queries, value_dim).
    """
    scores = np.matmul(query, key.transpose())

    attention_weights = soft(scores, axis=-1)

    output = np.matmul(attention_weights, value)

    return output


def shouldFilter(dtype, config):
    if dtype in ["float8_e4m3fn", "float8_e5m2"]:
        return config["HEAD_DIM"] < 64 or config["BLOCK_N"] < 64
    return False


def generate_configs():
    base_configs = [
        {"N_CTX": 64, "BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 2},
        {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 4},
        {"N_CTX": 64, "BLOCK_M": 64, "BLOCK_N": 32, "HEAD_DIM": 32, "NUM_WARPS": 2, "NUM_CTAS": 2},
        {"N_CTX": 64, "BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 128, "NUM_WARPS": 4, "NUM_CTAS": 1},
        {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS": 4, "NUM_CTAS": 1},
        {"N_CTX": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 128, "NUM_WARPS": 2, "NUM_CTAS": 1},
        {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 128, "NUM_WARPS": 4, "NUM_CTAS": 1},
        {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 128, "HEAD_DIM": 128, "NUM_WARPS": 4, "NUM_CTAS": 1},
    ]
    configs = []
    for dtype in ["bfloat16", "float8_e5m2"]:
        for config in base_configs:
            if shouldFilter(dtype, config):
                continue
            new_config = config.copy()
            new_config["DTYPE"] = dtype
            configs.append(new_config)
    return configs


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    m = torch.max(x, axis=1).values[:, None]
    # e_x = x-m
    e_x = torch.exp(x - m)
    return e_x / e_x.sum(axis=1)[:, None]  # only difference


@triton.jit
def flash_attention_kernel(Q, K, V, Out,  #
                           sm_scale, stride_qz, stride_qh, stride_qm, stride_qk,  #
                           stride_kz, stride_kh, stride_kn, stride_kk,  #
                           stride_vz, stride_vh, stride_vk, stride_vn,  #
                           stride_oz, stride_oh, stride_om, stride_on,  #
                           Z, H, N_CTX,  #
                           BLOCK_M: tl.constexpr,  #
                           BLOCK_N: tl.constexpr,  #
                           HEAD_DIM: tl.constexpr,  #
                           STAGE: tl.constexpr  #
                           ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)

    # block pointers
    Kdesc = tl.make_tensor_descriptor(
        base=K,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        block_shape=(BLOCK_N, HEAD_DIM),
    )
    Vdesc = tl.make_tensor_descriptor(
        base=V,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        block_shape=(BLOCK_N, HEAD_DIM),
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout, so we don't necessarily
    # need to load directly in LDS. We can simply have it in registers
    # (however, another possibility might be to have it in LDS to save
    # register)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    q_ptrs = Q + stride_qm * offs_m[:, None] + stride_qk * offs_k[None, :]
    q = tl.load(q_ptrs, mask=mask)

    lo, hi = 0, N_CTX
    offsetkv_y = lo
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = Kdesc.load([offsetkv_y, 0]).T
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = Vdesc.load([offsetkv_y, 0])
        p = p.to(V.type.element_ty)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij
        offsetkv_y += BLOCK_N
    acc = acc / l_i[:, None]

    # Store C directly
    o_ptrs = Out + stride_om * offs_m[:, None] + stride_on * offs_k[None, :]
    tl.store(o_ptrs, acc, mask=mask)


def test_fa(config):
    print(f"Compiling {config}")

    N_CTX = config["N_CTX"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    HEAD_DIM = config["HEAD_DIM"]
    NUM_WAPRS = config["NUM_WARPS"]
    NUM_CTAS = config["NUM_CTAS"]
    DTYPE = config["DTYPE"]
    H = 1
    STAGE = 1

    torch_type = getattr(torch, DTYPE)
    q = (torch.randn(N_CTX, HEAD_DIM)).to(torch_type)
    k = (torch.randn(N_CTX, HEAD_DIM)).to(torch_type)
    v = (torch.randn(N_CTX, HEAD_DIM)).to(torch_type)
    q_d = q.cuda()
    k_d = k.cuda()
    v_d = v.cuda()

    o_numpy = attn(q.to(torch.float32).numpy(), k.to(torch.float32).numpy(), v.to(torch.float32).numpy())
    o_d = torch.empty(N_CTX, HEAD_DIM, dtype=torch.float32).cuda()
    numBlocks = int(N_CTX / BLOCK_M)
    grid = [numBlocks, 1, 1]

    # Please note, FFM does not support scratch memory, so if the kernel uses scratch it'll segfault
    flash_attention_kernel[grid](q_d, k_d, v_d, o_d, 1.0, 1, 1, q.stride(0), q.stride(1), 1, 1,
                                 k.stride(0), k.stride(1), 1, 1, v.stride(0), v.stride(1), 1, 1, o_d.stride(0),
                                 o_d.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=NUM_WAPRS,
                                 num_ctas=NUM_CTAS, num_stages=3, Z=1, H=1, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM, STAGE=3)
    o_triton = o_d.cpu().numpy()
    rtol = 0.02
    atol = 0.02
    if DTYPE == "float8_e5m2":
        rtol = 0.03
        atol = 0.1
    if not allclose_numpy(o_triton, o_numpy, rtol=rtol, atol=atol):
        print("FAIL")
    else:
        print("OK")


if __name__ == "__main__":
    for config in generate_configs():
        test_fa(config)
