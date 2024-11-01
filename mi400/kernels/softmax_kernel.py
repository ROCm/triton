import torch
import triton
import triton.language as tl
import numpy as np
from kernels.test_common import allclose_numpy


def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


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


def testSoftmax(config):
    M = config["M"]
    N = config["N"]
    BLOCK_SIZE = triton.next_power_of_2(N)

    torch.manual_seed(42)
    input_h = torch.randn(M, N, dtype=torch.float32)
    input_d = input_h.cuda()
    output_d = torch.empty_like(input_d, device=0)
    numBlocks = 2
    grid = [numBlocks, 1, 1]
    softmax_kernel[grid](output_d, input_d, N, N, M, N, BLOCK_SIZE=BLOCK_SIZE)
    y_triton = output_d.cpu().numpy()
    input_h = input_h.numpy()
    y_numpy = softmax(input_h)
    if not allclose_numpy(y_triton, y_numpy):
        print("FAIL")
    else:
        print("OK")


if __name__ == "__main__":
    for config in generate_configs():
        testSoftmax(config)
