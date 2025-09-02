import hip

hip.hip.hipInit(0)

import triton
import triton.language as tl
import torch
import triton
import triton.language as tl
import numpy as np
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from kernels.test_common import allclose_numpy
from kernels.mxgemm_kernel import mxgemm_kernel

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
        dataa = torch.randint(1,ub, (d0,d1))
        return MXFP4Tensor(data = dataa)
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
    print(config)
    M = config['M']
    N = config['N']
    K = config['K']
    blockSizeM = config['BLOCK_M']
    blockSizeN = config['BLOCK_N']
    blockSizeK = config['BLOCK_K']
    numCtas = config['NUM_CTAS']
    numWarps = config['NUM_WARPS']
    dtype_a = config['DTYPE_A']
    dtype_b = config['DTYPE_B']
    scale_block = config['SCALE_BLOCK']

    # num_stages = 3

    fpflag_a = getfpflag(dtype_a)
    fpflag_b = getfpflag(dtype_b)

    ## For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    np.set_printoptions(threshold=np.inf)

    a = init_data(dtype_a, M, K, False)
    b = init_data(dtype_b, K, N, False)
    a_size = (M, (K + scale_block - 1) // scale_block)
    b_size = (N, (K + scale_block - 1) // scale_block)
    a_scale = MXScaleTensor(size = a_size).random(high = 32.0).data
    b_scale = MXScaleTensor(size = b_size).random(high = 32.0).data

    # a_scale = torch.randint(200, 201, (M, K // scale_block), dtype=torch.uint8)
    # b_scale = torch.randint(200, 201, (N, K // scale_block), dtype=torch.uint8)
    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype_a in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if dtype_b in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)


    c_triton = torch.empty(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    group_size_m = 1
    USE_TDM = 1

    stride_scale = a_scale_d.stride(0)

    mxgemm_kernel[grid](a_d, b_d, c_triton, a_scale_d, b_scale_d, M, N, K, stride_scale,
     a_d.stride(0), a_d.stride(1), b_d.stride(0), b_d.stride(1), c_triton.stride(0), c_triton.stride(1), 
    fpflag_a, fpflag_b, scale_block, blockSizeM, blockSizeN, blockSizeK, group_size_m, USE_TDM,
    num_warps=numWarps, num_ctas = numCtas)

    c_ref_numpy = c_ref.cpu().numpy()
    c_triton_numpy = c_triton.cpu().numpy()

    if not allclose_numpy(c_triton_numpy, c_ref_numpy, atol=1e-2):
        print("FAIL")
    else:
        print("OK")

def generate_configs():
    # for dtype in ['float8_e5m2', 'float4']:
    base_configs = [
        {"M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":128, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 256, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 256, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 256, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 256, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 32, "N": 32, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 64, "N": 64, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 64, "N": 64, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 64, "N": 64, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 64, "N": 64, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 128, "N": 128, "K": 512, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 128, "N": 128, "K": 512, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 128, "N": 128, "K": 512, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float4", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 128, "N": 128, "K": 512, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},

        {"M": 1, "N": 8192, "K": 128, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 8192, "N": 8192, "K": 128, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 1, "N": 8192, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 8192, "N": 8192, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float4", "SCALE_BLOCK":32},
        {"M": 1, "N": 8192, "K": 128, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 8192, "N": 8192, "K": 128, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 1, "N": 8192, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
        {"M": 8192, "N": 8192, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1, "DTYPE_A":"float8_e5m2", "DTYPE_B":"float8_e5m2", "SCALE_BLOCK":32},
    ]
    configs = base_configs

    return configs

if __name__ == "__main__":
    for config in generate_configs():
        testGemm(config)
