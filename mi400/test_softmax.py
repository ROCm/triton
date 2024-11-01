from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments
from triton.tools.env import getTritonBasePath
from sim.utils import *
import triton
import re
import os
import torch
import numpy as np
from kernels.softmax_kernel import generate_configs


def getHeadersAndData(ffmResults):
    table_headers = ["M", "N", "DTYPE", "Status"]
    rows = []
    for t, r in ffmResults.items():
        match = re.search("softmax_test_M_(.*)_N_(.*)_K_(.*)_DTYPE_(.*)", t)
        if match:
            M = match.group(1)
            N = match.group(2)
            DTYPE = match.group(10)
            rows.append([M, N, DTYPE, r])
    return table_headers, rows, "SOFTMAX"


def testSoftmax(config):
    args = Arguments()
    N = config["N"]
    M = config["M"]
    args.kernel_name = "softmax_kernel"
    BLOCK_SIZE = triton.next_power_of_2(N)
    args.path = os.path.join(getTritonBasePath(), "mi400/kernels/softmax_kernel.py")
    args.signature = f"*fp32,*fp32,i32,i32,i32,i32,{BLOCK_SIZE}"
    args.out_path = os.path.join(getTritonBasePath(), "softmax")
    args.num_warps = 4
    args.num_stages = 1
    args.flush_denorm = 1
    args.arch = "gfx1251"
    shaderInfo = aot_compile(args)

    # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    input = torch.randn(M, N, dtype=torch.float32)
    output = torch.softmax(input, dim=1)
    addressInput = sim.createInputSurface(input)
    addressOutput = sim.createOutputSurface(output)

    numBlocks = 2
    grid = [numBlocks, 1, 1]
    sim.createArgs([addressOutput, addressInput, N, N, M, N], grid)
    surfaceIniFile = sim.launch(args.num_warps, args.num_cta, grid, shaderInfo)
    regIniFile = sim.done()
    return FFMConfig(name="softmax", id=config, variance=0.01, sp3=shaderInfo.sp3filename, regIni=regIniFile,
                     surfaceIni=surfaceIniFile)


def testAllConfigs():
    ffmConfigs = []
    configs = generate_configs()
    for config in configs:
        ffmConfig = testSoftmax(config)
        if ffmConfig:
            ffmConfigs.append(ffmConfig)
    return ffmConfigs


if __name__ == "__main__":
    ffmConfigs = testAllConfigs()
    cfgstr = (generateFFMConfigs(ffmConfigs=ffmConfigs))
    print(cfgstr)
