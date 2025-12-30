import pytest
import torch
import triton

## Version 0 matmul
## No pipeline, no explicit LDS, just buffer_load
## vgpr: 244
## perf: 680

#from matmul_kernels.matmul_kernel import v0 as matmul_kernel

## Version 1 matmul
## Replace buffer_load with AsyncCopy
## no LDS swizzling
## vgpr: 216
## perf: 685
## LDS swizzling (vec=8, perPhase=2, maxPhase=8)
## vgpr: 178
## perf: 800
## However, the swizzled version has numerical issues

#from matmul_kernels.matmul_kernel import v1 as matmul_kernel

## Version 2 matmul
#from matmul_kernels.matmul_kernel import v2 as matmul_kernel

## Version 3 matmul 4 waves
#from matmul_kernels.matmul_kernel import v3 as matmul_kernel

## Version 4 matmul
## 4 waves with 2-stage pipelining
#from matmul_kernels.matmul_kernel import v4 as matmul_kernel

## Version 5 matmul
## 4 waves with 3-stage pipelining
#from matmul_kernels.matmul_kernel import v5 as matmul_kernel

from matmul_kernels.matmul_kernel import v6 as matmul_kernel

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def matmul(a, b, num_warps):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return c


M, N, K = 4096, 4096, 4096
#M, N, K = 256, 256, 256
num_warps = 4

## Check correctness
a = torch.rand((M, K), device=DEVICE, dtype=torch.float16) - .5
b = torch.rand((N, K), device=DEVICE, dtype=torch.float16).T - .5
triton_output = matmul(a, b, num_warps)
torch_output = torch.matmul(a, b)

part_torch = torch_output[0:256:, 0:256]
part_triton = triton_output[0:256:, 0:256]

#for row in range(0, 256):
#    #for col in range(0, N // 256):
#    part_torch = torch_output[row, 0:256]
#    part_triton = triton_output[row, 0:256]
#    if torch.allclose(part_triton, part_torch, atol=1e-1, rtol=0):
#        print(f"{(row)}: ✅")
#    else:
#        print(f"{(row)}: ❌")

if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

print(f"max diff = {torch.max(triton_output-torch_output)}")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[(M, N, K)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["triton"],  # if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
        line_names=["Triton"],  # if fp8_inputs else [ref_lib, "Triton"],  # Line styles
        styles=[("green", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" + ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
    num_warps = 4
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, num_warps), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=True)
