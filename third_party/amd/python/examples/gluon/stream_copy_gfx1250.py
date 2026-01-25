"""
Simple Gluon stream copy kernel for GFX1250 (Babel style).
- FP16 data input/output
- 4 warps per CTA (4 wave workgroup)
- 128-bit read/write (8 x fp16 elements per lane)
"""

# ruff: noqa: E402
import hip

# Initialize HIP
hip.hip.hipInit(0)

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import pytest

# Constants for GFX1250
THREADS_PER_WARP = 32  # GFX1250 warp size
ELEMENTS_PER_THREAD = 8  # 128 bits / 16 bits per fp16 = 8 elements
WARPS_PER_CTA = 4  # 4 warps per CTA.
BLOCK_SIZE = ELEMENTS_PER_THREAD * THREADS_PER_WARP * WARPS_PER_CTA  # 1024 elements per workgroup (4 warps)


@gluon.jit
def stream_copy_kernel(
    src_ptr,
    dst_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """
    Simple stream copy kernel.
    4 warps (128 threads) process BLOCK_SIZE elements (1024 fp16 values).
    Each thread loads/stores 8 fp16 elements (128 bits).
    Uses BlockedLayout for explicit thread mapping.
    """
    # Get workgroup ID
    wg_id = gl.program_id(0)

    # Base offset for this workgroup
    base_offset = wg_id * BLOCK_SIZE

    # Create a 1D blocked layout for the tensor
    # BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)
    # 8 elements per thread, 32 threads per warp, 4 warps per CTA
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])

    # Generate row/col offsets for a [1, BLOCK_SIZE] 2D view
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    # Combine with base offset
    offsets = base_offset + offs_col

    # Create mask for boundary checking
    mask = offsets < N

    # Load from source using buffer_load (128-bit coalesced access)
    data = gl.amd.gfx1250.buffer_load(src_ptr, offsets, mask=mask)

    # Store to destination using buffer_store (128-bit coalesced access)
    gl.amd.gfx1250.buffer_store(data, dst_ptr, offsets, mask=mask)


@gluon.jit
def stream_copy_kernel_looped(
    src_ptr,
    dst_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """
    Looped stream copy kernel for handling larger shapes.

    Uses a grid-stride loop pattern where each workgroup processes multiple
    chunks of BLOCK_SIZE elements. Loop bounds and iterations are computed
    dynamically inside the kernel based on N and grid dimensions.

    4 warps (128 threads) process BLOCK_SIZE elements (1024 fp16 values) per iteration.
    Each thread loads/stores 8 fp16 elements (128 bits).
    Uses BlockedLayout for explicit thread mapping.
    """
    # Get workgroup ID
    wg_id = gl.program_id(0)

    # Compute loop stride (total elements processed per iteration across all workgroups)
    grid_stride = NUM_WGS * BLOCK_SIZE

    # Compute number of loop iterations needed for this workgroup
    # Each workgroup starts at wg_id * BLOCK_SIZE and strides by grid_stride
    # We need to iterate until we've covered all elements assigned to this workgroup
    # num_iters = ceil((N - wg_id * BLOCK_SIZE) / grid_stride) for positive values
    start_offset = wg_id * BLOCK_SIZE
    # Compute iterations: ceil((N - start_offset) / grid_stride) when start_offset < N
    # Using integer arithmetic: (N - start_offset + grid_stride - 1) // grid_stride
    remaining = N - start_offset
    # Clamp to 0 if negative (workgroup has no work)
    remaining = gl.maximum(remaining, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    # Create a 1D blocked layout for the tensor
    # BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)
    # 8 elements per thread, 32 threads per warp, 4 warps per CTA
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])

    # Generate column offsets for a [1, BLOCK_SIZE] 2D view (constant across iterations)
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    # Grid-stride loop: iterate over chunks assigned to this workgroup
    for i in range(0, num_iters):
        # Compute base offset for this iteration
        # Each iteration advances by grid_stride
        base_offset = start_offset + i * grid_stride

        # Combine with column offsets
        offsets = base_offset + offs_col

        # Create mask for boundary checking
        mask = offsets < N

        # Load from source using buffer_load (128-bit coalesced access)
        data = gl.amd.gfx1250.buffer_load(src_ptr, offsets, mask=mask)

        # Store to destination using buffer_store (128-bit coalesced access)
        gl.amd.gfx1250.buffer_store(data, dst_ptr, offsets, mask=mask)


@gluon.jit
def stream_copy_kernel_looped_pipelined(
    src_ptr,
    dst_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """
    Software pipelined stream copy kernel with prefetch from global memory.

    Uses a double-buffering approach where we prefetch the next iteration's data
    from global memory while storing the current iteration's data. This overlaps
    global memory latency with compute/store operations.

    Pipeline structure:
      - Prologue: Issue load for iteration 0 before entering the loop
      - Main loop: For iteration i, store data[i] while prefetching data[i+1]
      - The prefetch on the last iteration is masked out (all-false mask)

    4 warps (128 threads) process BLOCK_SIZE elements (1024 fp16 values) per iteration.
    Each thread loads/stores 8 fp16 elements (128 bits).
    Uses BlockedLayout for explicit thread mapping.
    """
    # Get workgroup ID
    wg_id = gl.program_id(0)

    # Compute loop stride (total elements processed per iteration across all workgroups)
    grid_stride = NUM_WGS * BLOCK_SIZE

    # Compute number of loop iterations needed for this workgroup
    start_offset = wg_id * BLOCK_SIZE
    remaining = N - start_offset
    remaining = gl.maximum(remaining, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    # Create a 1D blocked layout for the tensor
    # BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)
    # 8 elements per thread, 32 threads per warp, 4 warps per CTA
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])

    # Generate column offsets for a [1, BLOCK_SIZE] 2D view (constant across iterations)
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    # ==================== PROLOGUE ====================
    # Prefetch first iteration's data before entering the main loop.
    # This issues the global memory load early so data is in-flight.
    offsets_0 = start_offset + offs_col
    mask_0 = offsets_0 < N
    data = gl.amd.gfx1250.buffer_load(src_ptr, offsets_0, mask=mask_0)

    # ==================== MAIN LOOP (Software Pipelined) ====================
    # Each iteration:
    #   1. Compute offsets for current iteration (i)
    #   2. Issue prefetch load for next iteration (i+1) - overlaps with store
    #   3. Store current iteration's data (loaded in previous iter or prologue)
    #   4. Rotate: data = data_next
    for i in range(0, num_iters):
        # Current iteration's offsets and mask
        offsets_curr = start_offset + i * grid_stride + offs_col
        mask_curr = offsets_curr < N

        # Prefetch next iteration's data while we process current iteration.
        # On the last iteration (i == num_iters - 1), offsets_next will be
        # out of bounds, so mask_next will be all-false, making this a no-op.
        offsets_next = start_offset + (i + 1) * grid_stride + offs_col
        mask_next = offsets_next < N
        data_next = gl.amd.gfx1250.buffer_load(src_ptr, offsets_next, mask=mask_next)

        # Store current data (from prologue or previous prefetch)
        # This store overlaps with the prefetch load issued above
        gl.amd.gfx1250.buffer_store(data, dst_ptr, offsets_curr, mask=mask_curr)

        # Rotate buffer: next iteration will use the prefetched data
        data = data_next


LOOPED_KERNEL_WORKGROUPS = 32  # Fixed workgroup count for looped kernels


def run_stream_copy(kernel_fn, N: int, check: bool = True):
    """
    Run the stream copy kernel.

    Args:
        kernel_fn: The kernel function to run
        N: Number of elements to copy
        check: Whether to verify correctness

    Returns:
        The compiled kernel for profiling
    """
    # Use FP16 dtype
    dtype = torch.float16

    # Create random source tensor
    torch.random.manual_seed(42)
    src = torch.randn(N, dtype=dtype)

    # Create destination tensor (zeros)
    dst = torch.zeros(N, dtype=dtype)

    # Keep reference on CPU if checking
    if check:
        ref = src.clone()

    # Move to GPU
    src = src.cuda()
    dst = dst.cuda()

    # Calculate grid size
    # Looped kernels use a fixed workgroup count (32) regardless of problem size.
    # The kernel internally loops with grid-stride to cover all elements.
    # Non-looped (base) kernel uses ceil(N / BLOCK_SIZE) workgroups.
    is_looped_kernel = kernel_fn in (stream_copy_kernel_looped, stream_copy_kernel_looped_pipelined)
    if is_looped_kernel:
        num_workgroups = LOOPED_KERNEL_WORKGROUPS
    else:
        num_workgroups = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch kernel with 4 warps (4 waves)
    kernel = kernel_fn[(num_workgroups, )](
        src,
        dst,
        N,
        BLOCK_SIZE,
        num_workgroups,
        num_warps=4,  # 4 waves per workgroup
        waves_per_eu=1,
    )

    # Synchronize
    torch.cuda.synchronize()

    # Move result back to CPU for verification
    dst_cpu = dst.cpu()

    if check:
        # Verify correctness
        torch.testing.assert_close(dst_cpu, ref, rtol=0, atol=0)
        print(f"PASSED: Stream copy of {N} fp16 elements")

    return kernel


# Test configurations
def generate_test_configs():
    return [
        # Original kernel tests
        pytest.param(stream_copy_kernel, 32768, id="base_N=32768"),
        pytest.param(stream_copy_kernel_looped, 327680, id="looped_N=327680"),
        pytest.param(stream_copy_kernel_looped_pipelined, 327680, id="pipelined_N=327680"),
        # Small and unalgined kernel tests
        pytest.param(stream_copy_kernel, 500, id="base_N=500"),
        pytest.param(stream_copy_kernel_looped, 500, id="looped_N=500"),
        pytest.param(stream_copy_kernel_looped_pipelined, 500, id="pipelined_N=500"),
    ]


@pytest.mark.parametrize("kernel_fn, N", generate_test_configs())
def test_stream_copy(kernel_fn, N):
    """Test stream copy kernel correctness."""
    run_stream_copy(kernel_fn, N, check=True)


def select_kernel(arg_kernel_type):
    if arg_kernel_type == "pipelined":
        kernel_fn = stream_copy_kernel_looped_pipelined
        kernel_name = "stream_copy_kernel_looped_pipelined"
    elif arg_kernel_type == "looped":
        kernel_fn = stream_copy_kernel_looped
        kernel_name = "stream_copy_kernel_looped"
    else:
        kernel_fn = stream_copy_kernel
        kernel_name = "stream_copy_kernel"
    return kernel_fn, kernel_name


if __name__ == "__main__":
    import argparse

    # Handle imports for both pytest (module context) and direct execution
    try:
        from .gfx1250_utils import static_profile
    except ImportError:
        from gfx1250_utils import static_profile

    parser = argparse.ArgumentParser(description="Stream Copy Kernel for GFX1250")
    parser.add_argument("-n", type=int, default=65536, help="Number of elements to copy")
    parser.add_argument("--kernel-type", type=str, choices=["default", "looped", "pipelined"], default="default",
                        help="Kernel type to use")
    args = parser.parse_args()

    kernel_fn, kernel_name = select_kernel(args.kernel_type)
    is_looped = args.kernel_type in ("looped", "pipelined")
    if is_looped:
        num_workgroups = LOOPED_KERNEL_WORKGROUPS
    else:
        num_workgroups = (args.n + BLOCK_SIZE - 1) // BLOCK_SIZE

    print(f"Running {kernel_name} with N={args.n} fp16 elements")
    print("Configuration: 4 warps/CTA, 128-bit (8 fp16) read/write per thread")
    print(f"Block size: {BLOCK_SIZE} elements per workgroup")
    print(f"Grid size: {num_workgroups} workgroups")
    if args.kernel_type == "looped":
        print(
            f"Looped kernel: each workgroup handles multiple iterations (fixed {LOOPED_KERNEL_WORKGROUPS} workgroups)")
    if args.kernel_type == "pipelined":
        print(
            f"Pipelined kernel: software pipelined with global memory prefetch (fixed {LOOPED_KERNEL_WORKGROUPS} workgroups)"
        )
    print()

    kernel = run_stream_copy(kernel_fn, args.n, check=True)

    print("\nStatic Profile:")
    static_profile(kernel)
