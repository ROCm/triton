import subprocess
import os
import pandas as pd
from prettytable import PrettyTable

def run_profiling(triton_dir, M, N, K, output_file):
    command = [
        "rocprof", "--stats", "-o", output_file,
        "python", f"{triton_dir}/python/perf-kernels/fused_moe/twolayerMLP.py", 
        "-M", str(M), "-N", str(N), "-K", str(K),
    ]
    subprocess.run(command, check=True)

def parse_profiling_output(output_file, kernel_names):
    df = pd.read_csv(output_file)
    results = {}
    for kernel in kernel_names:
        kernel_data = df[df['Name'].str.strip('"') == kernel]
        if not kernel_data.empty:
            results[kernel] = kernel_data['AverageNs'].iloc[0] / 1000.0
        else:
            results[kernel] = None
    
    # Calculate sum of other kernels
    other_kernels = df[~df['Name'].str.strip('"').isin(kernel_names)]
    other_kernels_sum = other_kernels['AverageNs'].sum() / 1000.0
    results['other_kernels_sum'] = other_kernels_sum
    
    return results

def main():
    triton_dir = os.environ.get("TRITONDIR", "~/triton")  # Default to ~/triton if not set
    output_file = os.path.expanduser("~/profiling.csv")
    kernel_names = ["gemm2gemm.kd", "gemm2gemm_persistent.kd", "gemm2gemm_persistent_buffered.kd", "reduce_buffers.kd", "matmul_kernel.kd"]
    M = [4096]
    N = [4096]
    K = [4096]
    
    table = PrettyTable()
    table.field_names = ["M", "N", "K"] + kernel_names + ["Other Kernels Sum (µs)"]
    
    for m, n, k in zip(M, N, K):
        print(f"Running profiling for M={m}, N={n}, K={k}")
        run_profiling(triton_dir, m, n, k, output_file)
        output_stats_file = os.path.expanduser("~/profiling.stats.csv")
        kernel_results = parse_profiling_output(output_stats_file, kernel_names)
        row = [m, n, k] + [kernel_results.get(kernel, "N/A") for kernel in kernel_names] + [kernel_results.get('other_kernels_sum', "N/A")]
        table.add_row(row)
    
    print("\nProfiling Summary (in microseconds):")
    print(table)

if __name__ == "__main__":
    main()
