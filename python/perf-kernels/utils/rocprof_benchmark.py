import subprocess
import os
import pandas as pd
from prettytable import PrettyTable

def run_profiling(triton_dir, batch_size, prefix_len, extend_len, output_file):
    command = [
        "rocprof", "--stats", "-o", output_file,
        "python", f"{triton_dir}/python/perf-kernels/MLA_extend.py",
        "-B", str(batch_size), "-prefix_len", str(prefix_len), "-extend_len", str(extend_len), "-dtype", "fp16",
        "-attn_impl", "normal" if prefix_len==0 else "absorb", "-fuse_wkc", "fuse_wvc"
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
    kernel_names = ["_fwd_fused_kernel.kd", "_fwd_kernel.kd"]
    batch_sizes = [16, 1]
    prefix_len = [0, 4096]
    extend_len = [4096, 2048]
    
    results = {B: {} for B in batch_sizes}
    for B, p, e in zip(batch_sizes, prefix_len, extend_len):
        print(f"Running profiling for B={B}...")
        run_profiling(triton_dir, B, p, e, output_file)
        output_stats_file = os.path.expanduser("~/profiling.stats.csv")
        kernel_results = parse_profiling_output(output_stats_file, kernel_names)
        results[B] = kernel_results
    
    table = PrettyTable()
    table.field_names = ["B", "prefix_len", "extend_len"] + kernel_names + ["Other Kernels Sum (µs)"]
    for B, p, e in zip(batch_sizes, prefix_len, extend_len):
        row = [B, p, e] + [results[B].get(kernel, "N/A") for kernel in kernel_names] + [results[B].get('other_kernels_sum', "N/A")]
        table.add_row(row)
    
    print("\nProfiling Summary (in microseconds):")
    print(table)

if __name__ == "__main__":
    main()
