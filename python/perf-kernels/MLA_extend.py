from utils.extend_attention import extend_attention_fwd

# from MLA_flash import extend_fused_attention_fwd
from utils.extend_attention import extend_fused_attention_fwd

import logging
import time

import triton
import triton.language as tl

import sys
import torch
import pytest

import argparse

from utils.rotary_embedding import DeepseekScalingRotaryEmbedding

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_hip_ = is_hip()

def input_helper_fused(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device):
    torch.manual_seed(0)
    q_extend = torch.randn(B * extend_length, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]

    # extend indexing
    qo_indptr = torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B*(prefix_length), device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    w_kc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)


    return q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc

@pytest.mark.parametrize("B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim", [
    (2, 16, 1024, 1024, 512, 64, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('fuse_wkc', [False, True])
@pytest.mark.parametrize('fuse_wvc', [False, True])
@pytest.mark.parametrize('absorb_wkc', [False, True])
@pytest.mark.parametrize('absorb_wvc', [False, True])
def test_op_fwd(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, fuse_wkc, fuse_wvc, absorb_wkc, absorb_wvc, sm_scale=1.0, logit_cap=0.0, device="cuda"):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    
    q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper_fused(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
   
    # Test with fused=False
    output_ref = forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                 fuse_wkc, fuse_wvc, w_kc, w_vc, absorb_wkc, absorb_wvc, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=False)

    # Test with fused=True
    output_fused = forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                   fuse_wkc, fuse_wvc, w_kc, w_vc, absorb_wkc, absorb_wvc, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=True)

    # Compare the outputs
    print(output_fused.sum())
    print("First 10 elements of output_ref:", output_ref.flatten()[:10])
    print("First 10 elements of output_fused:", output_fused.flatten()[:10])
    torch.testing.assert_close(output_ref, output_fused, rtol=1e-2, atol=1e-2)


def forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                    fuse_wkc, fuse_wvc, w_kc, w_vc, absorb_wkc, absorb_wvc, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused):
    
    if not fused: # aka reference
        fuse_wkc = False
        fuse_wvc = False

    if not fuse_wkc: # 1st gemm
        q_input = torch.empty((*q_extend.shape[:-1], kv_lora_rank + qk_rope_head_dim), dtype=q_extend.dtype, device=q_extend.device)
        q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
        q_nope = q_extend[..., :v_head_dim]
        q_nope = torch.bmm(q_nope.transpose(0, 1), w_kc.transpose(1, 2))
        q_input[..., :kv_lora_rank] = q_nope.transpose(0, 1)
    else:
        q_input = q_extend

    if fuse_wvc: # Depending on if we fuse the 2nd gemm, we output different head dim
        out = torch.empty( (*q_extend.shape[:-1], v_head_dim), dtype=q_extend.dtype, device=q_extend.device)
    else:
        out = torch.empty( (*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)

    if not fused:
        extend_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    else:
        extend_fused_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap,
                                   fuse_w_kc=fuse_wkc, fuse_w_vc=fuse_wvc, w_kc=w_kc, w_vc=w_vc, absorb_w_kc=absorb_wkc, absorb_w_vc=absorb_wvc)
    
    if not fuse_wvc: # 2nd gemm
        bmm_output = torch.bmm(out.transpose(0, 1), w_vc)
        out = bmm_output.transpose(0, 1)

    return out

def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
                    (2, 16, 1024, 1024, 512, 64, 128),
                    ]
    
    if args.B:
        x_vals_list = [
                    (args.B, 16, 1024, 1024, 512, 64, 128),
                    ]

    x_names = ["B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim"]

    line_vals = ["ref", "fused"]

    if args.ref:
        line_vals = ["ref"]

    if args.fused:
        line_vals = ["fused"]

    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, sm_scale, logit_cap, device,
                  provider):
        warmup = 2
        rep = 10


        q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper_fused(
            B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
      
        
        if "fused" in provider:
            fn = lambda: forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                         True, False, w_kc, w_vc, True, False, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=True)

        if "ref" in provider:
            fn = lambda: forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                         False, False, w_kc, w_vc, False, False, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=False)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=None, print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals



arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-fused", action="store_true", default=False)
    parser.add_argument("-ref", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-do_gemms", type=bool, default=True)
    parser.add_argument("-absorb_wkc", type=bool, default=True)
    parser.add_argument("-absorb_wvc", type=bool, default=True)
    parser.add_argument("-B", type=int, default=1)
    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

import re
from prettytable import PrettyTable

def parse_vgpr_usage(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract VGPR-related information
    vgpr_info = []
    table_lines = []
    in_table = False

    for line in lines:
        # Parse autotuning outputs
        if re.search(r"Autotuning kernel", line):
            vgpr_info.append(line.strip())
        if re.search(r"Triton autotuning for function", line):
            vgpr_info.append(line.strip())

        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(r"^\s*MLA-decode:", line):
            in_table = True
            # table_lines.append(line.strip())
        elif in_table:
            table_lines.append(line.strip())

    # Print extracted information
    print("\n".join(vgpr_info))

    table = PrettyTable()
    table.field_names = table_lines[0].split()
    [table.add_row(line.split()[1:]) for line in table_lines[1:]]

    print(table)


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)

import sys
import time
import re
import os
import tempfile

def print_vgpr(args):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        output_file = temp_file.name

        # Redirect stdout and stderr to the temporary file
        sys.stdout = temp_file
        sys.stderr = temp_file
        
        os.environ["AMDGCN_ENABLE_DUMP"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        run_bench(args)  # Run the benchmark
        
        sys.stdout.flush()
        sys.stderr.flush()

    # Restore stdout and stderr to normal
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    time.sleep(0.5)  # Ensure everything is written before reading

    # Parse and print relevant output
    parse_vgpr_usage(output_file)

    # Remove the temporary file
    os.unlink(output_file)

def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(args)
        return 0
    run_bench(args)
    # test_op_fwd(args.B, 16, 128, 256, 256, 64, 64, torch.float16, True, False, False, False, 1.0, 0.0, "cuda")

if __name__ == "__main__":
    main()


