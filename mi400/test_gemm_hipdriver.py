import hip

hip.hip.hipInit(0)

import torch
import triton
import triton.language as tl
import numpy as np
from kernels.test_common import allclose_numpy


def shouldFilter(dtype, config):
    if dtype in ["float8_e4m3fn", "float8_e5m2"]:
        return config["BLOCK_K"] < 64
    return False


def generate_configs():
    base_configs = [
        {
            "M": 16, "N": 16, "K": 32, "BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "NUM_WARPS": 1, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128, "NUM_WARPS": 1, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 64, "N": 64, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 128, "N": 128, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 256, "N": 256, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 256, "N": 256, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 32, "N": 32, "K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "NUM_WARPS": 1, "NUM_CTAS": 2,
            "USE_TDM": 1
        },
        {
            "M": 32, "N": 32, "K": 32, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "NUM_WARPS": 4, "NUM_CTAS": 2,
            "USE_TDM": 1
        },
        {
            "M": 64, "N": 64, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_WARPS": 1, "NUM_CTAS": 4,
            "USE_TDM": 1
        },
        {
            "M": 128, "N": 128, "K": 128, "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_WARPS": 1, "NUM_CTAS": 4,
            "USE_TDM": 1
        },
        {
            "M": 128, "N": 128, "K": 128, "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_WARPS": 1, "NUM_CTAS": 4,
            "USE_TDM": 0
        },
        {
            "M": 64, "N": 64, "K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "NUM_WARPS": 1, "NUM_CTAS": 2,
            "USE_TDM": 1
        },
        {
            "M": 64, "N": 64, "K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "NUM_WARPS": 1, "NUM_CTAS": 2,
            "USE_TDM": 0
        },
        {
            "M": 1, "N": 2 * 43, "K": 512, "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_WARPS": 8, "NUM_CTAS":
            1, "USE_TDM": 1
        },
    ]
    configs = []
    for dtype in ["bfloat16", "float8_e5m2"]:
        for config in base_configs:
            new_config = config.copy()
            new_config["DTYPE"] = dtype
            if shouldFilter(dtype, config):
                continue
            configs.append(new_config)
    return configs


@triton.jit
def gemm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        M, N, K, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        USE_TDM: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if USE_TDM:
        a_desc = tl._experimental_make_tensor_descriptor(base=a_ptr + (pid_m * BLOCK_SIZE_M) * K, shape=(M, K),
                                                         strides=(K, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
        b_desc = tl._experimental_make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_SIZE_N, shape=(K, N),
                                                         strides=(N, 1), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_offset = 0
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if USE_TDM:
            a = a_desc.load([0, k])
            b = b_desc.load([k, 0])
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, acc=accumulator)

        if not USE_TDM:
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * N

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def testGemm(config):
    print(config)
    DTYPE = config["DTYPE"]
    M = config["M"]
    N = config["N"]
    K = config["K"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_WARPS = config["NUM_WARPS"]
    NUM_CTAS = config["NUM_CTAS"]
    USE_TDM = config["USE_TDM"]
    groupSizeM = 1

    torch.manual_seed(42)
    torch_type = getattr(torch, DTYPE)
    a_h = torch.randint(1, 6, (M, K)).to(torch_type)
    b_h = torch.randint(1, 6, (K, N)).to(torch_type)
    c_d = torch.empty(M, N, dtype=torch.float32).cuda()
    a_d = a_h.cuda()
    b_d = b_h.cuda()
    numBlocks = int((M + BLOCK_M - 1) / BLOCK_M) * int((N + BLOCK_N - 1) / BLOCK_N)
    grid = [numBlocks, 1, 1]
    gemm_kernel[grid](a_d, b_d, c_d, M, N, K, N, 1, BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
                      GROUP_SIZE_M=groupSizeM, USE_TDM=USE_TDM, num_warps=NUM_WARPS, num_ctas=NUM_CTAS)
    c_triton = c_d.cpu().numpy()
    c_numpy = a_h.to(torch.float32).numpy() @ b_h.to(torch.float32).numpy()
    if not allclose_numpy(c_triton, c_numpy):
        print("FAIL")
    else:
        print("OK")


if __name__ == "__main__":
    for config in generate_configs():
        testGemm(config)
