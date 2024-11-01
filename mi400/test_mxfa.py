from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
from triton.tools.mxfp import MXFP4Tensor, MXFP6Tensor, MXScaleTensor
from sim.utils import *

import os
import torch
import re

isllamacpp = 0


def shouldFilter(dtype, config):
    if dtype in ["float8_e4m3fn", "float8_e5m2"]:
        return config["HEAD_DIM"] < 64 or config["BLOCK_N"] < 64
    return False


def generate_configs():
    base_configs = [
        {
            "HQ": 1, "N_CTX_Q": 42, "HK": 1, "N_CTX_KV": 128, "BLOCK_M": 32, "BLOCK_N": 128, "HEAD_DIM": 128,
            "NUM_WARPS": 4, "NUM_CTAS": 1
        },
        # {"HQ": 1, "N_CTX_Q": 42, "HK": 1, "N_CTX_KV": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 1},
        # {"HQ": 4, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 1},
        # {"HQ": 4, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 256, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 1},
        # {"HQ": 16, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 256, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS": 1, "NUM_CTAS": 1},
    ]
    # configs = []
    # for dtype in ["bfloat16", "float8_e5m2"]:
    #     for config in base_configs:
    #         if shouldFilter(dtype, config):
    #             continue
    #         new_config = config.copy()
    #         new_config["DTYPE"] = dtype
    #         configs.append(new_config)
    return base_configs


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


def attn(q, q_scale, k, k_scale, v, v_scale, scale_block):
    # q_scale = fp8e8m0_to_float32(q_scale).repeat_interleave(scale_block, dim=-1)
    k_scale = fp8e8m0_to_float32(k_scale).repeat_interleave(scale_block, dim=-1)
    v_scale = fp8e8m0_to_float32(v_scale).repeat_interleave(scale_block, dim=-2)
    # print(q_scale.size())
    print(k_scale.size())
    print(v_scale.size())
    # q = q.to(torch.float32) * q_scale
    q = q.to(torch.float32)
    print(q.size())
    k = k.to(torch.float32) * k_scale
    print(k.size())
    v = v.to(torch.float32) * v_scale
    print(v.size())

    q_shape = q.shape
    k_shape = k.shape
    HQ = q_shape[1]
    HK = k_shape[1]

    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    sm_scale = 1.0
    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k.float()) * sm_scale

    # scores += b.float()
    p = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p, v.float())
    # O = ref_out.transpose(1, 2).clone()
    return ref_out


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


def generate_tensor(dtype, B, H, N, D):
    if dtype == 'float4':
        a = MXFP4Tensor(data=torch.randint(1, 3, (B, H, N, D)))
    elif dtype == "float6_e2m3":
        a = MXFP6Tensor(data=torch.randint(1, 2, (B, H, N, D)), e=2)
    elif dtype == "float6_e3m2":
        a = MXFP6Tensor(data=torch.randint(1, 2, (B, H, N, D)), e=3)
    else:
        torch_type = getattr(torch, dtype)
        a = (torch.randint(1, 2, (B, H, N, D))).to(torch_type)
    return a


def test_fa(config):
    print(f"Compiling {create_config_id(config)}")

    N_CTX_Q = config["N_CTX_Q"]
    N_CTX_KV = config["N_CTX_KV"]
    HQ = config["HQ"]
    HK = config["HK"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    HEAD_DIM = config["HEAD_DIM"]
    NUM_WAPRS = config["NUM_WARPS"]
    NUM_CTAS = config["NUM_CTAS"]

    signatureQ = "i32:16,i32:16,i32:16,1"
    signatureK = "i32:16,i32:16,i32:16,1"
    signatureV = "i32:16,i32:16,i32:16,1"
    signatureO = "i32:16,i32:16,i32:16,1"

    SM_SCALE = 1.0
    scale_block = 32
    scale_stride = HEAD_DIM // scale_block

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (HEAD_DIM - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    BLOCK_DMODEL = max(padded_d_model, 16)
    ACTUAL_BLOCK_DMODEL = HEAD_DIM

    args = Arguments()
    dtypeq = "float8_e5m2"
    dtypekv = "float4"
    ptypeq = get_ptype(dtypeq)
    ptypekv = get_ptype(dtypekv)

    args.arch = "gfx1251"
    args.num_cta = NUM_CTAS
    args.kernel_name = "mxfa"
    args.path = os.path.join(getTritonBasePath(), "mi400/kernels/mxfa.py")
    args.signature = f"*{ptypeq}:16,*{ptypekv}:16,*{ptypekv}:16, *u8:16, *u8:16, *u8:16, {SM_SCALE},*fp32:16"
    args.signature += f",i32:16,i32:16"
    args.signature += f",{signatureQ},{signatureK},{signatureV},{signatureO}"
    args.signature += f",{HQ}, {HK}, {ACTUAL_BLOCK_DMODEL}, {BLOCK_M}, {BLOCK_DMODEL}, {BLOCK_N}, {scale_block}, {scale_stride}"
    # Store each config in a subfolder
    args.out_path = create_output_dir(config, prefix="fa")
    args.num_warps = NUM_WAPRS
    args.num_stages = 1
    shaderInfo = aot_compile(args)
    # Filter out configurations that use scratch memory
    if shaderInfo.use_scratch:
        print("use scratch")
        return None

    # # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    q = generate_tensor(dtypeq, 1, HQ, N_CTX_Q, HEAD_DIM)
    k = generate_tensor(dtypekv, 1, HK, N_CTX_KV, HEAD_DIM)
    v = generate_tensor(dtypekv, 1, HK, N_CTX_KV, HEAD_DIM)
    q_scale = torch.randint(127, 128, (1, HQ, N_CTX_Q, HEAD_DIM // scale_block), dtype=torch.uint8)
    k_scale = torch.randint(127, 128, (1, HK, N_CTX_KV, HEAD_DIM // scale_block), dtype=torch.uint8)
    v_scale = torch.randint(127, 128, (1, HK, N_CTX_KV // scale_block, HEAD_DIM), dtype=torch.uint8)
    o = attn(q, None, k, k_scale, v, v_scale, scale_block)
    print(o.shape)

    k = k.to_packed_tensor(dim=3)
    v = v.to_packed_tensor(dim=2)

    addressQ = sim.createInputSurface(q)
    # dimsQ = [HEAD_DIM * N_CTX_Q * HQ, HEAD_DIM * N_CTX_Q, HEAD_DIM]
    dimsQ = list(q.stride()[0:3])
    addressK = sim.createInputSurface(k)
    # dimsK = [HEAD_DIM * N_CTX_KV * HK, HEAD_DIM * N_CTX_KV, HEAD_DIM]
    dimsK = list(k.stride()[0:3])
    addressV = sim.createInputSurface(v)
    # dimsV = [HEAD_DIM * N_CTX_KV * HK, HEAD_DIM * N_CTX_KV, HEAD_DIM]
    dimsV = list(v.stride()[0:3])
    addressO = sim.createOutputSurface(o)
    dimsO = [N_CTX_Q * HEAD_DIM * HQ, HEAD_DIM * HQ, HEAD_DIM]

    scaleQ = sim.createInputSurface(q_scale)
    scaleK = sim.createInputSurface(k_scale)
    scaleV = sim.createInputSurface(v_scale)
    kargs = [addressQ, addressK, addressV, scaleQ, scaleK, scaleV, addressO] + [N_CTX_Q, N_CTX_KV
                                                                                ] + dimsQ + dimsK + dimsV + dimsO
    numBlocks = int((N_CTX_Q + BLOCK_M - 1) / BLOCK_M)

    grid = [numBlocks, HQ, 1]
    sim.createArgs(kargs, grid)
    surfaceIniFile = sim.launch(args.num_warps, args.num_cta, grid, shaderInfo)
    regIniFile = sim.done()
    faVariance = 0.03
    return FFMConfig(name="fa", id=config, variance=faVariance, sp3=shaderInfo.sp3filename, regIni=regIniFile,
                     surfaceIni=surfaceIniFile)


def testAllConfigs():
    ffmConfigs = []
    configs = generate_configs()
    for config in configs:
        ffmConfig = test_fa(config)
        if ffmConfig:
            ffmConfigs.append(ffmConfig)
    return ffmConfigs


if __name__ == "__main__":
    ffmConfigs = testAllConfigs()
    cfgstr = (generateFFMConfigs(ffmConfigs=ffmConfigs))
    print(cfgstr)
