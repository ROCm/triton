from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
from triton.tools.env import getTritonBasePath
from sim.utils import *
import os
import torch
import re
import warnings


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
        # {"M": 32, "N": 32, "K": 128, "BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":128, "NUM_WARPS": 1, "NUM_CTAS": 1, "USE_TDM":1},
        # {"M": 64, "N": 64, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":64, "NUM_WARPS": 4  , "NUM_CTAS": 1, "USE_TDM":1},
        # {"M": 128, "N": 128, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":64, "NUM_WARPS": 4, "NUM_CTAS": 1, "USE_TDM":1},
        # {"M": 256, "N": 256, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":32, "NUM_WARPS": 4, "NUM_CTAS": 1, "USE_TDM":1},
        {
            "M": 64, "N": 64, "K": 64, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "USE_TDM": 1
        },
        {
            "M": 128, "N": 128, "K": 128, "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_WARPS": 4, "NUM_CTAS": 4,
            "USE_TDM": 1
        },
        # {"M": 256, "N": 256, "K": 64, "BLOCK_M": 64, "BLOCK_N":64, "BLOCK_K":64, "NUM_WARPS": 4, "NUM_CTAS": 1, "USE_TDM":1},
        # {"M": 1, "N": 2*43, "K": 512, "BLOCK_M": 128, "BLOCK_N":128, "BLOCK_K":128, "NUM_WARPS": 8, "NUM_CTAS": 1, "USE_TDM":1},
    ]
    configs = []
    for dtype in ["bfloat16"]:  #, "float8_e5m2"]:
        for config in base_configs:
            new_config = config.copy()
            new_config["DTYPE"] = dtype
            if shouldFilter(dtype, config):
                continue
            configs.append(new_config)
    return configs


def getHeadersAndData(ffmResults):
    table_headers = [
        "M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K", "NUM_WARPS", "NUM_CTAS", "USE_TDM", "DTYPE", "Status"
    ]
    rows = []
    for t, r in ffmResults.items():
        match = re.search(
            "gemm_test_M_(.*)_N_(.*)_K_(.*)_BLOCK_M_(.*)_BLOCK_N_(.*)_BLOCK_K_(.*)_NUM_WARPS_(.*)_NUM_CTAS_(.*)_USE_TDM_(.*)_DTYPE_(.*)",
            t)
        if match:
            M = match.group(1)
            N = match.group(2)
            K = match.group(3)
            BLOCK_M = match.group(4)
            BLOCK_N = match.group(5)
            BLOCK_K = match.group(6)
            NUM_WARPS = match.group(7)
            NUM_CTAS = match.group(8)
            USE_TDM = match.group(9)
            DTYPE = match.group(10)
            rows.append([M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, USE_TDM, DTYPE, r])
    return table_headers, rows, "GEMM"


def testGemm(config):
    # Default to float16. Change this to bfloat16 to use bf16 datatypes
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
    cStrideM = 1
    kernel_file = "gemm_kernel_llamacpp"
    outdir = "gemm_kernel_llamacpp"
    num_stages = 3
    ptype = getPtype(DTYPE)

    args = Arguments()
    args.num_cta = NUM_CTAS
    args.kernel_name = "gemm_kernel_llamacpp"
    args.path = os.path.join(getTritonBasePath(), f"mi400/kernels/{kernel_file}.py")
    args.signature = f"*{ptype}:16,*{ptype}:16,*fp32:16,i32:16,i32:16,i32:16,i32:16,{cStrideM},{BLOCK_M},{BLOCK_N},{BLOCK_K},{groupSizeM},{USE_TDM}"
    args.out_path = os.path.join(getTritonBasePath(), create_output_dir(config, prefix="gemm"))
    args.num_warps = NUM_WARPS
    args.num_stages = num_stages
    args.arch = "gfx1250"
    shaderInfo = aot_compile(args)
    if shaderInfo.use_scratch:
        warnings.warn("Skipping this config because it uses scratch size, which is not supported.")
        return None

    ## For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    torch_type = getattr(torch, DTYPE)
    sim = MI400Simulator(args.out_path)
    a = (torch.randint(1, 6, (M, K))).to(torch_type)
    b = (torch.randint(1, 6, (N, K))).to(torch_type)
    c = torch.matmul(a.float(), b.t().float()).float()

    addressA = sim.createInputSurface(a)
    addressB = sim.createInputSurface(b)
    addressC = sim.createOutputSurface(c)

    numBlocks = int((M + BLOCK_M - 1) / BLOCK_M) * int((N + BLOCK_N - 1) / BLOCK_N) * args.num_cta
    grid = [numBlocks, 1, 1]
    sim.createArgs([addressA, addressB, addressC, M, N, K, N, 1], grid)
    surfaceIniFile = sim.launch(args.num_warps, args.num_cta, grid, shaderInfo)
    regIniFile = sim.done()

    return FFMConfig(name="gemm", id=config, variance=0.01, sp3=shaderInfo.sp3filename, regIni=regIniFile,
                     surfaceIni=surfaceIniFile)


def testAllConfigs():
    ffmConfigs = []
    configs = generate_configs()
    for config in configs:
        ffmConfig = testGemm(config)
        if ffmConfig:
            ffmConfigs.append(ffmConfig)
    return ffmConfigs


if __name__ == "__main__":
    ffmConfigs = testAllConfigs()
    cfgstr = (generateFFMConfigs(ffmConfigs=ffmConfigs))
    print(cfgstr)
