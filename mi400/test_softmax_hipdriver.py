import hip

hip.hip.hipInit(0)

import os
import torch
import triton
import triton.language as tl
import pytest
import argparse


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)  # -> workgroup
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def run_softmax(config, args):
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


@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("N", [32, 64, 4096])
@pytest.mark.parametrize("dtype", ['float32'])
def test_softmax(M, N, dtype):
    config = {
        "M": M,  #
        "N": N,  #
        "DTYPE": dtype
    }

    class Args:

        def __init__(self):
            self.verbose = False
            self.dump_ir = 'none'

    args = Args()

    run_softmax(config, args)


def generate_configs():
    base_configs = [
        {"M": 32, "N": 32},
        {"M": 32, "N": 64},
        {"M": 128, "N": 4096},
    ]
    configs = []
    for dtype in ["float32"]:
        for config in base_configs:
            new_config = config.copy()
            new_config["DTYPE"] = dtype
            configs.append(new_config)
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    parser.add_argument("--dump-ir", choices=['none', 'ttir', 'ttgir', 'llir', 'amdgcn'], default="none",
                        help="dump IR format")
    args = parser.parse_args()

    configs = generate_configs()
    for config in configs:
        print(f'{config=}')
        run_softmax(config, args)
