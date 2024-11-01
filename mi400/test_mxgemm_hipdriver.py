import hip

hip.hip.hipInit(0)

import triton
import triton.language as tl
import torch
import triton
import triton.language as tl
import numpy as np
from triton.tools.mxfp import MXFP4Tensor, MXFP6Tensor, MXScaleTensor
from kernels.test_common import allclose_numpy
from test_mxgemm import generate_configs, get_ptype, torch_gemm_mxfp


@triton.jit
def mxgemm_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale: tl.constexpr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        fpflag: tl.constexpr, SCALE_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, USE_TDM: tl.constexpr):
    DIV_FACTOR: tl.constexpr = 2 if fpflag == 4 else 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K // DIV_FACTOR)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_BLOCK)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    if USE_TDM:
        a_desc = tl._experimental_make_tensor_descriptor(base=a_ptr + (pid_m * BLOCK_M) * stride_am, shape=(M, K),
                                                         strides=(stride_am, 1),
                                                         block_shape=(BLOCK_M, BLOCK_K // DIV_FACTOR))
        b_desc = tl._experimental_make_tensor_descriptor(base=b_ptr + (pid_n * BLOCK_N) * stride_bn, shape=(K, N),
                                                         strides=(stride_bk, 1),
                                                         block_shape=(BLOCK_K // DIV_FACTOR, BLOCK_N))
    else:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K // DIV_FACTOR)):
        k_remaining = K - k * (BLOCK_K // DIV_FACTOR)
        valid_k = offs_k < k_remaining
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)

        if USE_TDM:
            a = a_desc.load([0, k * (BLOCK_K // DIV_FACTOR)])
            b = b_desc.load([k * (BLOCK_K // DIV_FACTOR), 0])
        else:
            a = tl.load(a_ptrs, mask=valid_k[None, :], other=0.)
            b = tl.load(b_ptrs, mask=valid_k[:, None], other=0.)
            a_ptrs += (BLOCK_K // DIV_FACTOR) * stride_ak
            b_ptrs += (BLOCK_K // DIV_FACTOR) * stride_bk

        if fpflag == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
        elif fpflag == 62:
            accumulator = tl.dot_scaled(a, scale_a, "e2m3", b, scale_b, "e2m3", accumulator)
        elif fpflag == 63:
            accumulator = tl.dot_scaled(a, scale_a, "e3m2", b, scale_b, "e3m2", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)

        a_scale_ptr += BLOCK_K // SCALE_BLOCK
        b_scale_ptr += BLOCK_K // SCALE_BLOCK

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


def testGemm(config):
    print(config)
    M = config['M']
    N = config['N']
    K = config['K']
    blockSizeM = config['BLOCK_M']
    blockSizeN = config['BLOCK_N']
    blockSizeK = config['BLOCK_K']
    numCtas = config['NUM_CTAS']
    numWarps = config['NUM_WARPS']
    dtype = config['DTYPE']
    scale_block = config['SCALE_BLOCK']

    kernel_file = "mxgemm_kernel"
    outdir = "mxgemm_kernel"
    num_stages = 3

    fpflag = 8
    if dtype == 'float4':
        fpflag = 4
    elif dtype == 'float6_e2m3':
        fpflag = 62
    elif dtype == 'float6_e3m2':
        fpflag = 63

    ptype = get_ptype(dtype=dtype)

    ## For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    if dtype == 'float4':
        dataa = torch.randint(1, 6, (M, K))
        a = MXFP4Tensor(data=dataa)
        # print(a.to(torch.float32))
        # print(a.data)
        datab = torch.randint(1, 6, (K, N))
        b = MXFP4Tensor(data=datab)
    elif dtype == "float6_e2m3":
        a = MXFP6Tensor(data=torch.randint(1, 5, (M, K)), e=2)
        b = MXFP6Tensor(data=torch.randint(1, 5, (K, N)), e=2)
    elif dtype == "float6_e3m2":
        a = MXFP6Tensor(data=torch.randint(1, 5, (M, K)), e=3)
        b = MXFP6Tensor(data=torch.randint(1, 5, (K, N)), e=3)
    else:
        torch_type = getattr(torch, dtype)
        a = (torch.randint(1, 6, (M, K))).to(torch_type)
        b = (torch.randint(1, 6, (K, N))).to(torch_type)

    a_scale = torch.randint(127, 130, (M, K // scale_block), dtype=torch.uint8)
    b_scale = torch.randint(127, 130, (N, K // scale_block), dtype=torch.uint8)
    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, dtype)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
        b = b.to_packed_tensor(dim=0)

    c_triton = torch.empty(M, N, dtype=torch.float32).cuda()
    a_d = a.data.cuda()
    b_d = b.data.cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1
    USE_TDM = 1

    mxgemm_kernel[grid](a_d, b_d, c_triton, a_scale_d, b_scale_d, M, N, K, K // scale_block, a_d.stride(0), 1,
                        b_d.stride(0), 1, c_triton.stride(0), 1, fpflag, scale_block, blockSizeM, blockSizeN,
                        blockSizeK, group_size_m, USE_TDM=USE_TDM, num_warps=numWarps, num_ctas=numCtas)

    c_ref_numpy = c_ref.cpu().numpy()
    c_triton_numpy = c_triton.cpu().numpy()

    if not allclose_numpy(c_triton_numpy, c_ref_numpy):
        print("FAIL")
    else:
        print("OK")


if __name__ == "__main__":
    for config in generate_configs():
        testGemm(config)
