from triton.tools.mi400.aot import aot_compile
from triton.tools.mi400.mi400Simulator import MI400Simulator
from triton.tools.mi400.sim_arguments import Arguments
from triton.tools.env import getTritonBasePath

import os

import torch

configs = [
    # {"M": 32, "NUM_WARPS": 1},
    # {"M": 64, "NUM_WARPS": 1},
    # {"M": 96, "NUM_WARPS": 1},
    {"M": 128, "NUM_WARPS": 1},
    {"M": 128, "NUM_WARPS": 2},
    {"M": 128, "NUM_WARPS": 4},
    {"M": 128, "NUM_WARPS": 8},
    # {"M": 64, "NUM_WARPS": 2},
]


def create_config_id(config):
    config_key_value_pairs = [f"{key}_{value}" for key, value in config.items()]
    return "_".join(config_key_value_pairs)


def create_output_dir(config):
    return os.path.join(getTritonBasePath(), "simple_async_copy", create_config_id(config))


def test_fa(config):
    print(f"Compiling {create_config_id(config)}")

    M = config["M"]
    NUM_WAPRS = config["NUM_WARPS"]
    args = Arguments()
    args.kernel_name = "simple_async_copy"
    args.path = os.path.join(getTritonBasePath(), "python/test/mi400_kenrels/simple_async_load.py")
    args.signature = f"*fp32:16,*fp32:16,{M}"

    # Store each config in a subfolder
    args.out_path = create_output_dir(config)
    args.num_warps = NUM_WAPRS
    args.num_stages = 1
    shaderInfo = aot_compile(args)

    # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    input = torch.zeros(M, dtype=torch.float32) + 1
    # input = torch.rand_like(input)
    ref = input.clone()

    addressInput = sim.createInputSurface(input)
    addressOutput = sim.createOutputSurface(ref)
    kargs = [addressInput, addressOutput]
    numBlocks = 1  # int(M / blockSizeM) * int(N / blockSizeN)
    grid = [numBlocks, 1, 1]
    print(shaderInfo)
    sim.createArgs(kargs, grid)
    sim.launch(args.num_warps, grid, shaderInfo)
    sim.done()


def generate_test_config(configs, variance):
    config_str = ""
    for config in configs:

        config_str += f"""
        {{
                "mi400_simple_async_load_test_{create_config_id(config)}",
                "{create_output_dir(config)}/simple_async_copy.sp3",
                "{create_output_dir(config)}/memory_surface.ini",
                "{create_output_dir(config)}/reg_seq.ini",
                {variance}f, /*Variance*/
        }},
        """
    print(config_str)


if __name__ == "__main__":
    for config in configs:
        test_fa(config)
    generate_test_config(configs, 0.01)
