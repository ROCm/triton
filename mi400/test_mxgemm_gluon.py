import hip

hip.hip.hipInit(0)

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import numpy as np


def generate_configs():
    base_configs = [
        {
            "M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "DTYPE_A": "float4", "DTYPE_B": "float4", "SCALE_BLOCK": 32
        },
        # TODO: Add more shapes
    ]
    configs = []
    for config in base_configs:
        new_config = config.copy()
        configs.append(new_config)
    return configs


@gluon.jit
def mxgemm_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_scale: gl.constexpr, fpflag_a: gl.constexpr, fpflag_b: gl.constexpr,
                  SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                  GROUP_SIZE_M: gl.constexpr):

    BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
    A_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    B_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [2, 2])
    A_SCALE_LINEAR_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[0, 0], [16, 0]],
        block_bases=[], shape=[32, 4])
    B_SCALE_LINEAR_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[16, 0], [0, 0]],
        block_bases=[], shape=[32, 4])
    DIV_FACTOR_A: gl.constexpr = 2 if fpflag_a == 4 else 1
    DIV_FACTOR_B: gl.constexpr = 2 if fpflag_b == 4 else 1

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, A_BLOCKED_LAYOUT))) % M
    offs_ak = gl.arange(0, BLOCK_K // DIV_FACTOR_A, layout=gl.SliceLayout(0, A_BLOCKED_LAYOUT))
    offs_bk = gl.arange(0, BLOCK_K // DIV_FACTOR_B, layout=gl.SliceLayout(1, B_BLOCKED_LAYOUT))
    offs_bn = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, B_BLOCKED_LAYOUT))) % N

    offs_scale_am = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))) % M
    offs_scale_ak = gl.arange(0, BLOCK_K // SCALE_BLOCK, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_scale_bn = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))) % N
    offs_scale_bk = gl.arange(0, BLOCK_K // SCALE_BLOCK, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))

    a_scale_ptr = a_scale + offs_scale_am[:, None] * stride_scale + offs_scale_ak[None, :]
    b_scale_ptr = b_scale + offs_scale_bn[:, None] * stride_scale + offs_scale_bk[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for k in range(0, gl.cdiv(K, BLOCK_K)):
        k_remaining_a = K - k * (BLOCK_K // DIV_FACTOR_A)
        k_remaining_b = K - k * (BLOCK_K // DIV_FACTOR_B)
        valid_k_a = offs_ak < k_remaining_a
        valid_k_b = offs_bk < k_remaining_b

        scale_a = gl.load(a_scale_ptr)
        scale_b = gl.load(b_scale_ptr)
        scale_a = gl.convert_layout(scale_a, A_SCALE_LINEAR_LAYOUT)
        scale_b = gl.convert_layout(scale_b, B_SCALE_LINEAR_LAYOUT)

        a = gl.load(a_ptrs, mask=valid_k_a[None, :], other=0.0)
        b = gl.load(b_ptrs, mask=valid_k_b[:, None], other=0.0)
        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B) * stride_bk

        a = gl.convert_layout(a, gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT, k_width=64))
        b = gl.convert_layout(b, gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT, k_width=64))

        if fpflag_a == 4 and fpflag_b == 4:
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)

        a_scale_ptr += BLOCK_K // SCALE_BLOCK
        b_scale_ptr += BLOCK_K // SCALE_BLOCK

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.store(c_ptrs, accumulator, mask=c_mask)


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)

    a_scale_f32 = a_scale_f32.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # b_scales are always col major
    # b_scale_f32 = b_scale_f32.T.contiguous()

    a = a_f32 * a_scale_f32
    b = b_f32 * b_scale_f32

    ref_out = torch.matmul(a, b).to(torch.float32)

    return ref_out


def init_data(dtype, d0, d1, allones):
    ub = 2 if allones else 5
    if dtype == 'float4':
        dataa = torch.randint(1, ub, (d0, d1))
        return MXFP4Tensor(data=dataa)
    else:
        torch_type = getattr(torch, dtype)
        return (torch.randint(1, ub, (d0, d1))).to(torch_type)


def getfpflag(dtype):
    fpflag = 8
    if dtype == 'float4':
        fpflag = 4
    elif dtype == 'float6_e2m3':
        fpflag = 62
    elif dtype == 'float6_e3m2':
        fpflag = 63
    return fpflag


def testGemm(config):
    M = config["M"]
    N = config["N"]
    K = config["K"]
    blockSizeM = config["BLOCK_M"]
    blockSizeN = config["BLOCK_N"]
    blockSizeK = config["BLOCK_K"]
    numCtas = config['NUM_CTAS']
    numWarps = config['NUM_WARPS']
    dtype_a = config['DTYPE_A']
    dtype_b = config['DTYPE_B']
    scale_block = config['SCALE_BLOCK']

    fpflag_a = getfpflag(dtype_a)
    fpflag_b = getfpflag(dtype_b)

    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K, False)
    b = init_data(dtype_b, K, N, False)
    a_size = (M, triton.cdiv(K, scale_block))
    b_size = (N, triton.cdiv(K, scale_block))
    a_scale = MXScaleTensor(size=a_size).random(high=32.0).data
    b_scale = MXScaleTensor(size=b_size).random(high=32.0).data

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype_a in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if dtype_b in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.empty(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = a_scale_d.stride(0)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    mxgemm_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                        stride_cm, stride_cn, stride_scale, fpflag_a, fpflag_b, scale_block, blockSizeM, blockSizeN,
                        blockSizeK, group_size_m, num_warps=numWarps, num_ctas=numCtas)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-05, atol=1e-2)


if __name__ == "__main__":
    for config in generate_configs():
        testGemm(config)
