from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
from sim.utils import *
from kernels.flash_attention_kernel import generate_configs

import os
import torch
import re


def getHeadersAndData(ffmResults):
    table_headers = ["N_CTX", "BLOCK_M", "BLCOK_N", "HEAD_DIM", "NUM_WARPS", "NUM_CTAS", "DTYPE", "Status"]
    rows = []
    for t, r in ffmResults.items():
        match = re.search(
            "fa_test_N_CTX_(.*)_BLOCK_M_(.*)_BLOCK_N_(.*)_HEAD_DIM_(.*)_NUM_WARPS_(.*)_NUM_CTAS_(.*)_DTYPE_(.*)", t)
        if match:
            N_CTX = match.group(1)
            BLOCK_M = match.group(2)
            BLOCK_N = match.group(3)
            HEAD_DIM = match.group(4)
            NUM_WARPS = match.group(5)
            NUM_CTAS = match.group(6)
            DTYPE = match.group(7)
            rows.append([N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS, NUM_CTAS, DTYPE, r])
    return table_headers, rows, "Flash Attention"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    m = torch.max(x, axis=1).values[:, None]
    # e_x = x-m
    e_x = torch.exp(x - m)
    return e_x / e_x.sum(axis=1)[:, None]  # only difference


def attn(Q, K, V):
    S = torch.matmul(Q.float(), torch.transpose(K.float(), 0, 1))
    # P = torch.softmax(S.float(), dim=-1)
    P = softmax(S)
    # P = torch.softmax(S.float(), dim=1)
    # P = S.float()
    O = torch.matmul(P, V.float())  #.to(Q.dtype)
    print(O)
    # O = S.half()
    return O


def test_fa(config):
    print(f"Compiling {create_config_id(config)}")

    N_CTX = config["N_CTX"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    HEAD_DIM = config["HEAD_DIM"]
    NUM_WAPRS = config["NUM_WARPS"]
    NUM_CTAS = config["NUM_CTAS"]
    DTYPE = config["DTYPE"]
    H = 1
    STAGE = 1

    signatureQ = "i32:16,i32:16,i32:16,1"
    signatureK = "i32:16,i32:16,i32:16,1"
    signatureV = "i32:16,i32:16,i32:16,1"
    signatureO = "i32:16,i32:16,i32:16,1"

    args = Arguments()
    ptype = "fp16"
    if DTYPE == "bfloat16":
        ptype = "bf16"
    elif DTYPE == "float8_e4m3fn":
        ptype = "fp8e4nv"
    elif DTYPE == "float8_e5m2":
        ptype = "fp8e5"
    args.arch = "gfx1251"
    args.num_cta = NUM_CTAS
    args.kernel_name = "flash_attention_kernel"
    args.path = os.path.join(getTritonBasePath(), "mi400/kernels/flash_attention_kernel.py")
    print(args.path)
    args.signature = f"*{ptype}:16,*{ptype}:16,*{ptype}:16,*fp32:16"
    args.signature += ",fp32:16"
    args.signature += f",{signatureQ},{signatureK},{signatureV},{signatureO}"
    args.signature += ",1,1,i32:16"
    args.signature += f",{BLOCK_M}, {BLOCK_N}, {HEAD_DIM}, {STAGE}"
    # Store each config in a subfolder
    args.out_path = create_output_dir(config, prefix="fa")
    args.num_warps = NUM_WAPRS
    args.num_stages = 3
    # args.global_prefetch = 1
    shaderInfo = aot_compile(args)
    # Filter out configurations that use scratch memory
    if shaderInfo.use_scratch:
        return None

    # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    torch_type = getattr(torch, DTYPE)

    q = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)
    k = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)
    v = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)

    o = attn(q, k, v)

    addressQ = sim.createInputSurface(q)
    dimsQ = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressK = sim.createInputSurface(k)
    dimsK = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressV = sim.createInputSurface(v)
    dimsV = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressO = sim.createOutputSurface(o)
    dimsO = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    kargs = [addressQ, addressK, addressV, addressO, 1.0] + dimsQ + dimsK + dimsV + dimsO + [N_CTX]
    numBlocks = int(N_CTX / BLOCK_M) * args.num_cta
    grid = [numBlocks, 1, 1]
    sim.createArgs(kargs, grid)
    surfaceIniFile = sim.launch(args.num_warps, args.num_cta, grid, shaderInfo)
    regIniFile = sim.done()
    faVariance = 0.01
    if DTYPE not in ("float16", "bfloat16"):
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
