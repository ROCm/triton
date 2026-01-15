"""
测试 gemm.py 中的 matmul_kernel
这个测试文件提供了简单易用的测试用例来验证 matmul_kernel 的正确性
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# 确保使用 huizzhan 目录下的 triton
triton_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(triton_path))
print(f"使用 Triton 路径: {triton_path}")

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 先设置环境变量，避免 triton 导入问题
import os
os.environ.setdefault('TRITON_INTERPRET', '0')

try:
    from gemm import matmul_triton_main_perf, name_to_torch_types, gen_input, dtype_is_8_bit, SCALE_BLOCK_SIZE
    print("✓ 成功导入 gemm 模块")
except Exception as e:
    print(f"✗ 导入 gemm 模块时出错: {e}")
    print("尝试直接使用简化的测试...")
    import traceback
    traceback.print_exc()
    matmul_triton_main_perf = None

def test_matmul(M, N, K, num_warmup=5, num_iters=100):
    dtype=torch.bfloat16
    device="cuda"

    print(f"\n{'='*60}")
    print(f"Testing GEMM: M={M}, N={N}, K={K}, dtype={dtype}")
    print(f"{'='*60}")

    a = torch.randn((M,K), dtype=dtype, device=device)
    w = torch.randn((N,K), dtype=dtype, device=device)
    w_t = w.t().contiguous()  # 提前转置 w 并确保内存连续，shape: (K, N)

    print(f"Input shapes: a={a.shape}, w={w.shape}, w_t={w_t.shape}")
    print(f"a: min={a.min().item():.4f}, max={a.max().item():.4f}, mean={a.mean().item():.4f}")
    print(f"w: min={w.min().item():.4f}, max={w.max().item():.4f}, mean={w.mean().item():.4f}")

    print(f"\n{'='*60}")
    print("Running Triton matmul_triton_main_perf...")
    print(f"{'='*60}")
    
    # 检查 matmul_triton_main_perf 是否可用
    if matmul_triton_main_perf is None:
        print("错误: matmul_triton_main_perf 函数不可用，无法运行测试")
        print("请确保 triton 已正确安装并且环境配置正确")
        return
    
    # 使用 gemm.py 的 matmul，需要传入输出张量
    new_out = torch.empty((M, N), dtype=dtype, device=device)
    matmul_triton_main_perf(a, w_t, new_out, a_scale=None, b_scale=None, scale_a8_b8=None, activation="")
    print(f"Triton output shape: {new_out.shape}")
    print(f"Triton output: min={new_out.min().item():.4f}, max={new_out.max().item():.4f}, mean={new_out.mean().item():.4f}")

    # ori_out = triton_matmul(a, w)
    print(f"\n{'='*60}")
    print("Running PyTorch linear...")
    print(f"{'='*60}")
    torch_out = F.linear(a, w, bias=None)
    print(f"PyTorch output shape: {torch_out.shape}")
    print(f"PyTorch output: min={torch_out.min().item():.4f}, max={torch_out.max().item():.4f}, mean={torch_out.mean().item():.4f}")

    print(f"\n{'='*60}")
    print("Comparing results...")
    print(f"{'='*60}")
    diff = (new_out - torch_out).abs()
    print(f"Absolute difference: min={diff.min().item():.6f}, max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
    
    rel_diff = diff / (torch_out.abs() + 1e-8)
    print(f"Relative difference: min={rel_diff.min().item():.6f}, max={rel_diff.max().item():.6f}, mean={rel_diff.mean().item():.6f}")
    
    # 根据数据类型调整容差：BF16 精度较低，需要更宽松的容差
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2  # BF16: 1% 相对误差
        print(f"\nRunning torch.testing.assert_close with rtol={rtol}, atol={atol} (BF16 mode)...")
    else:
        rtol, atol = 1e-3, 1e-3  # FP16/FP32: 0.1% 相对误差
        print(f"\nRunning torch.testing.assert_close with rtol={rtol}, atol={atol}...")
    
    try:
        torch.testing.assert_close(new_out, torch_out, rtol=rtol, atol=atol)
        print(f"✅ Test PASSED!")
    except AssertionError as e:
        print(f"❌ Test FAILED!")
        print(f"Error: {e}")
        raise
    print(f"{'='*60}\n")

    # Performance comparison
    print(f"\n{'='*60}")
    print(f"Performance Comparison (warmup={num_warmup}, iterations={num_iters})")
    print(f"{'='*60}")
    
    # Warmup and benchmark Triton
    print(f"Warming up Triton kernel...")
    for _ in range(num_warmup):
        matmul_triton_main_perf(a, w_t, new_out, a_scale=None, b_scale=None, scale_a8_b8=None, activation="")
    torch.cuda.synchronize()
    
    print(f"Benchmarking Triton kernel...")
    gluon_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        matmul_triton_main_perf(a, w_t, new_out, a_scale=None, b_scale=None, scale_a8_b8=None, activation="")
        end.record()
        torch.cuda.synchronize()
        gluon_times.append(start.elapsed_time(end))  # milliseconds
    
    # Warmup and benchmark PyTorch
    print(f"Warming up PyTorch linear...")
    for _ in range(num_warmup):
        _ = F.linear(a, w, bias=None)
    torch.cuda.synchronize()
    
    print(f"Benchmarking PyTorch linear...")
    torch_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = F.linear(a, w, bias=None)
        end.record()
        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))  # milliseconds
    
    # Calculate statistics
    gluon_times = np.array(gluon_times)
    torch_times = np.array(torch_times)
    
    gluon_mean = gluon_times.mean()
    gluon_std = gluon_times.std()
    gluon_min = gluon_times.min()
    gluon_max = gluon_times.max()
    gluon_median = np.median(gluon_times)
    
    torch_mean = torch_times.mean()
    torch_std = torch_times.std()
    torch_min = torch_times.min()
    torch_max = torch_times.max()
    torch_median = np.median(torch_times)
    
    # Calculate TFLOPS
    flops = 2 * M * N * K  # multiply-add counts as 2 ops
    gluon_tflops = flops / (gluon_mean * 1e-3) / 1e12
    torch_tflops = flops / (torch_mean * 1e-3) / 1e12
    
    print(f"\n{'='*60}")
    print(f"Triton Kernel Performance:")
    print(f"  Mean:   {gluon_mean:.3f} ms (± {gluon_std:.3f} ms)")
    print(f"  Median: {gluon_median:.3f} ms")
    print(f"  Min:    {gluon_min:.3f} ms")
    print(f"  Max:    {gluon_max:.3f} ms")
    print(f"  TFLOPS: {gluon_tflops:.2f}")
    
    print(f"\nPyTorch Linear Performance:")
    print(f"  Mean:   {torch_mean:.3f} ms (± {torch_std:.3f} ms)")
    print(f"  Median: {torch_median:.3f} ms")
    print(f"  Min:    {torch_min:.3f} ms")
    print(f"  Max:    {torch_max:.3f} ms")
    print(f"  TFLOPS: {torch_tflops:.2f}")
    
    speedup = torch_mean / gluon_mean
    print(f"\n{'='*60}")
    if speedup > 1.0:
        print(f"🚀 Triton is {speedup:.2f}x FASTER than PyTorch!")
    elif speedup < 1.0:
        print(f"⚠️  Triton is {1/speedup:.2f}x SLOWER than PyTorch")
    else:
        print(f"⚖️  Triton and PyTorch have similar performance")
    print(f"{'='*60}\n")

def main():
    # test_matmul(64, 3072, 2048)
    test_matmul(128, 3072, 2048)
    # """主函数：运行所有测试"""
    # print("\n" + "=" * 60)
    # print("开始测试 gemm.py 中的 matmul_kernel")
    # print("=" * 60 + "\n")
    
    # try:
    #     # 检查 CUDA 是否可用
    #     if not torch.cuda.is_available():
    #         print("错误: CUDA 不可用，无法运行测试")
    #         return 1
        
    #     print(f"使用设备: {torch.cuda.get_device_name(0)}")
    #     print(f"CUDA 版本: {torch.version.cuda}")
    #     print(f"计算能力: {torch.cuda.get_device_capability()}")
    #     print()
        
    #     # 运行测试
    #     test_basic_fp16()
    #     test_basic_bf16()
    #     test_non_square()
    #     test_large_matrix()
    #     test_fp8_tensor_scaling()
    #     test_mixed_precision()
        
    #     # 性能测试
    #     benchmark_performance()
        
    #     print("=" * 60)
    #     print("所有测试完成!")
    #     print("=" * 60)
        
    #     return 0
        
    # except Exception as e:
    #     print(f"\n错误: 测试过程中发生异常: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return 1


if __name__ == "__main__":
    sys.exit(main())

