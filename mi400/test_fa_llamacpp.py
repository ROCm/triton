from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
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
            "HQ": 1, "N_CTX_Q": 42, "HK": 1, "N_CTX_KV": 64, "BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS":
            1, "NUM_CTAS": 1
        },
        {
            "HQ": 1, "N_CTX_Q": 42, "HK": 1, "N_CTX_KV": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS":
            1, "NUM_CTAS": 1
        },
        {
            "HQ": 4, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS":
            1, "NUM_CTAS": 1
        },
        {
            "HQ": 4, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 256, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64, "NUM_WARPS":
            1, "NUM_CTAS": 1
        },
        {
            "HQ": 16, "N_CTX_Q": 42, "HK": 4, "N_CTX_KV": 256, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64,
            "NUM_WARPS": 1, "NUM_CTAS": 1
        },
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


def attn(q, k, v, b):
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

    scores += b.float()
    p = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p, v.float())
    O = ref_out.transpose(1, 2).clone()
    return O


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
    DTYPE = config["DTYPE"]

    signatureQ = "i32:16,i32:16,i32:16,1"
    signatureK = "i32:16,i32:16,i32:16,1"
    signatureV = "i32:16,i32:16,i32:16,1"
    signatureO = "i32:16,i32:16,i32:16,1"
    signatureBias = "i32:16,i32:16,i32:16,1"
    SM_SCALE = 1.0

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (HEAD_DIM - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    BLOCK_DMODEL = max(padded_d_model, 16)
    ACTUAL_BLOCK_DMODEL = HEAD_DIM

    args = Arguments()
    ptype = "fp16"
    if DTYPE == "bfloat16":
        ptype = "bf16"
    elif DTYPE == "float8_e4m3fn":
        ptype = "fp8e4nv"
    elif DTYPE == "float8_e5m2":
        ptype = "fp8e5"

    ptypeq = ptype
    if isllamacpp:
        ptypeq = "fp32"

    args.arch = "gfx1251"
    args.num_cta = NUM_CTAS
    args.kernel_name = "flash_attention_llamacpp"
    args.path = os.path.join(getTritonBasePath(), "mi400/kernels/flash_attention_llamacpp.py")
    args.signature = f"*{ptypeq}:16,*{ptype}:16,*{ptype}:16, *{ptype}:16, {SM_SCALE},*fp32:16"
    args.signature += f",i32:16,i32:16"
    args.signature += f",{signatureQ},{signatureK},{signatureV},{signatureO},{signatureBias}"
    args.signature += f",{HQ}, {HK}, {ACTUAL_BLOCK_DMODEL}, {BLOCK_M}, {BLOCK_DMODEL}, {BLOCK_N}, {isllamacpp}"
    # Store each config in a subfolder
    args.out_path = create_output_dir(config, prefix="fa")
    args.num_warps = NUM_WAPRS
    args.num_stages = 3
    shaderInfo = aot_compile(args)
    # Filter out configurations that use scratch memory
    if shaderInfo.use_scratch:
        print("use scratch")
        return None

    # # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    torch_type = getattr(torch, DTYPE)
    torch_typeq = torch_type
    if isllamacpp:
        torch_typeq = torch.float32

    q = (torch.rand((1, HQ, N_CTX_Q, HEAD_DIM))).to(torch_typeq)
    k = (torch.rand((1, HK, N_CTX_KV, HEAD_DIM))).to(torch_type)
    v = (torch.rand((1, HK, N_CTX_KV, HEAD_DIM))).to(torch_type)
    b = (torch.zeros((1, 1, N_CTX_Q, N_CTX_KV))).to(torch_type)

    o = attn(q, k, v, b)

    addressQ = sim.createInputSurface(q)
    dimsQ = [HEAD_DIM * N_CTX_Q * HQ, HEAD_DIM * N_CTX_Q, HEAD_DIM]
    addressK = sim.createInputSurface(k)
    dimsK = [HEAD_DIM * N_CTX_KV * HK, HEAD_DIM * N_CTX_KV, HEAD_DIM]
    addressV = sim.createInputSurface(v)
    dimsV = [HEAD_DIM * N_CTX_KV * HK, HEAD_DIM * N_CTX_KV, HEAD_DIM]
    addressBias = sim.createInputSurface(b)
    dimsBias = [HEAD_DIM * N_CTX_KV * N_CTX_Q, N_CTX_KV * N_CTX_Q, N_CTX_KV]
    addressO = sim.createOutputSurface(o)
    dimsO = [N_CTX_Q * HEAD_DIM * HQ, HEAD_DIM, HEAD_DIM * HQ]
    kargs = [addressQ, addressK, addressV, addressBias, addressO] + [N_CTX_Q, N_CTX_KV
                                                                     ] + dimsQ + dimsK + dimsV + dimsO + dimsBias
    numBlocks = int(N_CTX_Q / BLOCK_M) * args.num_cta
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
