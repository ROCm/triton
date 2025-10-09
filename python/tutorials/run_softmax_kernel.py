import argparse
import torch
import triton
import triton.language as tl

# EDIT THIS to import your kernel symbol from your file:
# e.g. if your kernel is in softmax_kernel.py, use:
# from softmax_kernel import _softmax_kernel_online
from better_softmax import _softmax_kernel_online  # <-- change this


def triton_softmax_online(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Launch wrapper for your Triton kernel."""
    assert x.is_cuda, "Move the tensor to GPU first (x = x.cuda())."
    if not x.is_contiguous():
        x = x.contiguous()

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    grid = (n_rows,)  # one program per row
    num_warps = 4 if block_size <= 128 else 8

    _softmax_kernel_online[grid](
        y,                      # output_ptr
        x,                      # input_ptr
        x.stride(0),            # input_row_stride (elements)
        y.stride(0),            # output_row_stride (elements)
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,  # tl.constexpr
        num_warps=num_warps,    # launch meta
    )
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Triton softmax kernel with optional shape/seed overrides.")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows to use (defaults to random in [1, 2048)).")
    parser.add_argument("--cols", type=int, default=None, help="Number of cols to use (defaults to random in [1, 4096)).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for torch RNG. If omitted, a random seed is drawn.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        torch.seed()  # draw a fresh, random seed for this run
    device = "cuda" if torch.cuda.is_available() else "hip"

    # Random shape
    n_rows = args.rows if args.rows is not None else int(torch.randint(1, 2048, ()).item())
    n_cols = args.cols if args.cols is not None else int(torch.randint(1, 4096, ()).item())

    dtype = torch.float16  # try bfloat16/float32 too
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)

    block_size = 256
    y_triton = triton_softmax_online(x, block_size=block_size)

    # Reference (PyTorch)
    y_ref = torch.softmax(x.float(), dim=-1).to(dtype)

    max_abs_err = (y_triton - y_ref).abs().max().item()
    print(f"Shape: [{n_rows}, {n_cols}], BLOCK_SIZE={block_size}, seed={args.seed if args.seed is not None else 'random'}")
    print(f"max abs err vs torch: {max_abs_err:.3e}")
