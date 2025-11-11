import hip

hip.hip.hipInit(0)

import os
import triton
import torch
from kernels.softmax_kernel import generate_configs, softmax_kernel
import argparse


def testSoftmax(config, args):
    N = config["N"]
    M = config["M"]
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    num_stages = 1

    # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    input_h = torch.randn(M, N, dtype=torch.float32)
    output_h = torch.softmax(input_h, dim=1)

    input_d = input_h.cuda()
    output_d = torch.zeros_like(output_h, dtype=torch.float32).cuda()

    numBlocks = 2
    grid = [numBlocks, 1, 1]
    handle = softmax_kernel[grid](
        output_d, input_d,  #
        N, N, M, N,  #
        BLOCK_SIZE=BLOCK_SIZE,  #
        num_warps=num_warps,  #
        num_stages=num_stages)

    if args.dump_ir != 'none':
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'softmax.{args.dump_ir}'
        with open(os.path.join(curr_dir, filename), "w") as file:
            file.write(handle.asm[args.dump_ir])

    output_d = output_d.cpu()
    try:
        torch.testing.assert_close(output_h, output_d, rtol=1e-05, atol=1e-08)
        print("✅ Triton within tolerances.")
    except Exception as err:
        print("❌ Triton and Torch differ")
        print(err)
        if args.verbose:
            print(f"{output_h=}")
            print(f"{output_d=}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    parser.add_argument("--dump-ir", choices=['none', 'ttir', 'ttgir', 'llir', 'amdgcn'], default="none",
                        help="dump IR format")
    args = parser.parse_args()

    configs = generate_configs()
    for config in configs:
        testSoftmax(config, args)
