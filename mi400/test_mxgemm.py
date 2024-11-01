from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
from triton.tools.env import getTritonBasePath
from triton.tools.mxfp import MXFP4Tensor, MXFP6Tensor, MXScaleTensor
import triton
from sim.utils import *
import warnings

import os
import torch


def shouldFilter(dtype, config):
    if dtype in ["float8_e4m3fn", "float8_e5m2"]:
        return config["BLOCK_K"] < 64

    if dtype in ['float4', 'float6_e2m3', 'float6_e3m2']:
        return config['BLOCK_K'] < 128

    if config['K'] > 8192:
        return not (dtype in ['float4', 'float6_e2m3', 'float6_e3m2'])

    return False


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, dtype_src_str_a, dtype_src_str_b):
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)

    a_scale_f32 = a_scale_f32.repeat_interleave(scale_block, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(scale_block, dim=1)

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = a_f32 * a_scale_f32
    b = b_f32 * b_scale_f32

    ref_out = torch.matmul(a, b).to(torch.float32)

    return ref_out


# Default to float16. Change this to bfloat16 to use bf16 datatypes
def get_ptype(dtype):
    ptype = "fp16"
    if dtype == "bfloat16":
        ptype = "bf16"
    elif dtype == "float8_e4m3fn":
        ptype = "fp8e4nv"
    elif dtype == "float8_e5m2":
        ptype = "fp8e5"
    elif dtype in ['float6_e2m3', 'float6_e3m2', 'float4']:
        ptype = 'u8'
    else:
        raise ValueError(f"Type {dtype} not supported!")

    return ptype


def init_data(dtype, d0, d1, allones):
    ub = 2 if allones else 5
    if dtype == 'float4':
        dataa = torch.randint(1, ub, (d0, d1))
        return MXFP4Tensor(data=dataa)
        # print(a.to(torch.float32))
        # print(a.data)
    elif dtype == "float6_e2m3":
        return MXFP6Tensor(data=torch.randint(1, ub, (d0, d1)), e=2)
        # b = MXFP6Tensor(data = torch.randint(1,5, (K,N)), e=2)
    elif dtype == "float6_e3m2":
        return MXFP6Tensor(data=torch.randint(1, ub, (d0, d1)), e=3)
        # b = MXFP6Tensor(data = torch.randint(1,5, (K,N)), e=3)
    else:
        torch_type = getattr(torch, dtype)
        return (torch.randint(1, ub, (d0, d1))).to(torch_type)
        # b = (torch.randint(1, 6, (K, N))).to(torch_type)


def generate_configs():
    # for dtype in ['float8_e5m2', 'float4']:
    base_configs = [
        {
            "M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "DTYPE_A": "float4", "DTYPE_B": "float8_e5m2", "SCALE_BLOCK": 32
        },
        # {"M": 32, "N": 32, "K": 256, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1},
        # {"M": 32, "N": 32, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1},
        # {"M": 64, "N": 64, "K": 512, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1},
        # {"M": 128, "N": 128, "K": 512, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":256, "NUM_WARPS": 4, "NUM_CTAS": 1},
    ]
    configs = base_configs
    # for config in base_configs:
    #     for dtype in ['float4']:
    #         for sbs in [32]:
    #             new_config = config.copy()
    #             new_config['DTYPE_A'] = dtype
    #             new_config['SCALE_BLOCK'] = sbs
    #             if shouldFilter(dtype, config):
    #                 continue
    #             configs.append(new_config)

    return configs


def getfpflag(dtype):
    fpflag = 8
    if dtype == 'float4':
        fpflag = 4
    elif dtype == 'float6_e2m3':
        fpflag = 62
    elif dtype == 'float6_e3m2':
        fpflag = 63
    return fpflag


def triton_gemm_mxfp(config):
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

    kernel_file = "mxgemm_kernel"
    outdir = "mxgemm_kernel"
    num_stages = 3

    fpflag_a = getfpflag(dtype_a)
    fpflag_b = getfpflag(dtype_b)
    ptype_a = get_ptype(dtype=dtype_a)
    ptype_b = get_ptype(dtype=dtype_b)

    args = Arguments()
    args.kernel_name = "mxgemm_kernel"
    args.path = os.path.join(getTritonBasePath(), f"mi400/kernels/{kernel_file}.py")
    group_size_m = 1
    USE_TDM = 1
    args.signature = f"*{ptype_a}:16,*{ptype_b}:16,*fp32:16,*u8:16,*u8:16,i32:16, i32:16, i32:16," + \
                       f"{K//scale_block}, i32:16,1, i32:16, 1, i32:16, 1, {fpflag_a},{fpflag_b},{scale_block},{blockSizeM},{blockSizeN},{blockSizeK},{group_size_m},{USE_TDM}"
    args.out_path = os.path.join(getTritonBasePath(), create_output_dir(config, prefix=outdir))
    args.num_warps = numWarps
    args.num_stages = num_stages
    args.num_cta = numCtas
    args.arch = 'gfx1251'
    shaderInfo = aot_compile(args)
    if shaderInfo.use_scratch:
        warnings.warn("Skipping this config because it uses scratch size, which is not supported.")
        return None
    print(f"shaderInfo = {shaderInfo}")

    ## For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)
    sim = MI400Simulator(args.out_path)
    a = init_data(dtype_a, M, K, False)
    b = init_data(dtype_b, K, N, False)

    a_scale = torch.randint(127, 130, (M, K // scale_block), dtype=torch.uint8)
    b_scale = torch.randint(127, 130, (N, K // scale_block), dtype=torch.uint8)
    c = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, dtype_a, dtype_b)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if dtype_a in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if dtype_b in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    addressA = sim.createInputSurface(a)
    addressScaleA = sim.createInputSurface(a_scale)
    addressB = sim.createInputSurface(b)
    addressScaleB = sim.createInputSurface(b_scale)
    addressC = sim.createOutputSurface(c)

    numBlocks = triton.cdiv(M, blockSizeM) * triton.cdiv(N, blockSizeN)
    grid = [numBlocks, 1, 1]
    sim.createArgs([
        addressA, addressB, addressC, addressScaleA, addressScaleB, a.shape[0], b.shape[1], b.shape[0],
        a.stride(0),
        b.stride(0),
        c.stride(0), fpflag_a, fpflag_b, scale_block
    ], grid)
    surfaceIniFile = sim.launch(args.num_warps, args.num_cta, grid, shaderInfo)
    regIniFile = sim.done()

    return FFMConfig(name="gemm_mxfp", id=config, variance=0.01, sp3=shaderInfo.sp3filename, regIni=regIniFile,
                     surfaceIni=surfaceIniFile)


if __name__ == "__main__":
    ffmConfigs = []
    configs = generate_configs()
    for config in configs:
        ffgConfig = triton_gemm_mxfp(config)
        if ffgConfig:
            ffmConfigs.append(ffgConfig)
    cfgstr = (generateFFMConfigs(ffmConfigs=ffmConfigs))
    print(cfgstr)
