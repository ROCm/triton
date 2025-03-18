from utils.extend_attention import extend_attention_fwd, extend_fused_attention_fwd, extend_persistent_attention_fwd

import logging
import time

import triton
import triton.language as tl

import sys
import torch
import pytest

import argparse
from typing import Tuple

def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    fp8_max = finfo.max
    if is_hip_:
        fp8_max = 224.0
    scale = fp8_max / amax
    x_scl_sat = (x * scale).clamp(min=-fp8_max, max=fp8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_hip_ = is_hip()

def input_helper_fused(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device):
    torch.manual_seed(0)
    q_extend = torch.randn(B * extend_length, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]

    # extend indexing
    qo_indptr = torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(B * (prefix_length), 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B*(prefix_length), device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    w_kc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)

    # FP8 quantization for gemms
    w_q, w_descale = input_to_float8(torch.concatenate((w_kc, w_vc), dim=-1), torch.float8_e4m3fnuz)
    w_descale = w_descale.item()
    w_kc = w_q[..., :v_head_dim]
    w_vc = w_q[..., v_head_dim:]

    return q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, w_descale

@pytest.mark.parametrize("B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim", [
    (2, 16, 2048, 512, 512, 64, 128),
    (2, 16, 0, 2048, 512, 64, 128),
])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('ref_attn_impl', ["absorb", "normal"])
@pytest.mark.parametrize('fuse_wkc', [False, True])
@pytest.mark.parametrize('fuse_wvc', [False, True])
@pytest.mark.parametrize('fp8', [False, True])
def test_op_fwd(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, ref_attn_impl, fuse_wkc, fuse_wvc, fp8, sm_scale=1.0, logit_cap=0.0, device="cuda"):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    if ref_attn_impl == "normal":
        forward = forward_normal
    else:
        forward = forward_absorb
    
    q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, w_descale = input_helper_fused(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
   
    # Reference
    output_ref = forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap, kv_lora_rank, qk_rope_head_dim, v_head_dim,
                               w_kc, w_vc, w_descale, ref=True, fuse_wkc=False, fuse_wvc=False, fp8=fp8)

    # Fused
    output_fused = forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap, kv_lora_rank, qk_rope_head_dim, v_head_dim,
                               w_kc, w_vc, w_descale, ref=False, fuse_wkc=fuse_wkc, fuse_wvc=fuse_wvc, fp8=fp8)

    # Compare the outputs
    # Print debug information for mismatches
    max_mismatches = 10
    diff = (output_ref - output_fused).abs()
    mismatches = diff > 1e-2
    if mismatches.any():
        mismatch_indices = torch.nonzero(mismatches)[:max_mismatches]
        print(f"\nFound {mismatches.sum().item()} mismatches, showing first {len(mismatch_indices)}:")
        for idx in mismatch_indices:
            i, h, d = idx.tolist()
            print(f"Position [{i}, {h}, {d}]: ref={output_ref[i, h, d].item():.6f}, fused={output_fused[i, h, d].item():.6f}, diff={diff[i, h, d].item():.6f}")
    
    torch.testing.assert_close(output_ref, output_fused, rtol=1e-2, atol=1e-2)
    print("Unit test passes, output_ref and output_fused are close!")


def forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap, kv_lora_rank, qk_rope_head_dim, v_head_dim,
                    w_kc, w_vc, w_descale, ref=True, persistent=False, fuse_wkc=False, fuse_wvc=False, fp8=False):
    
    dtype = v_extend.dtype
    device = v_extend.device
    
    if ref:
        fuse_wkc = False
        fuse_wvc = False

    if not fuse_wkc: # 1st gemm
        q_input = torch.empty((*q_extend.shape[:-1], kv_lora_rank + qk_rope_head_dim), dtype=dtype, device=device)
        q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
        q_nope = q_extend[..., :v_head_dim]
        # if fp8: # to check numerical stability
        #     q_nope, q_descale = input_to_float8(q_nope, w_kc.dtype)
        #     q_descale = q_descale.item()
        #     q_nope_out = (torch.bmm(q_nope.to(torch.bfloat16).transpose(0, 1), w_kc.transpose(1,2).to(torch.bfloat16)) * q_descale * w_descale).to(dtype)
        # else:
        q_nope_out = torch.bmm(q_nope.to(dtype).transpose(0, 1), w_kc.transpose(1,2).to(dtype) * w_descale).to(dtype)
        q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)
    else:
        q_input = q_extend

    if fuse_wvc: # Depending on if we fuse the 2nd gemm, we output different head dim
        out = torch.empty( (*q_extend.shape[:-1], v_head_dim), dtype=dtype, device=device)
    else:
        out = torch.empty( (*q_extend.shape[:-1], kv_lora_rank), dtype=dtype, device=device)

    if ref:
        extend_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    else:
        if persistent:
            extend_persistent_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
        else:            
            extend_fused_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap,
                                    fuse_w_kc=fuse_wkc, fuse_w_vc=fuse_wvc, w_kc=w_kc, w_vc=w_vc, w_descale=w_descale, fp8=fp8, qk_rope_head_dim=qk_rope_head_dim, qk_nope_head_dim=v_head_dim, kv_lora_rank=kv_lora_rank)
    
    if not fuse_wvc: # 2nd gemm
        attn_bmm_output = torch.bmm(out.to(dtype).transpose(0, 1), w_vc.to(dtype) * w_descale).to(dtype)
        out = attn_bmm_output.transpose(0, 1)

    return out

def forward_normal(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap, kv_lora_rank, qk_rope_head_dim, v_head_dim,
                    w_kc, w_vc, w_descale, ref=True, persistent=False, fuse_wkc=False, fuse_wvc=False, fp8=False):
    
    dtype = v_extend.dtype
    device = v_extend.device
    
    out = torch.empty((*q_extend.shape[:-1], v_head_dim), dtype=dtype, device=device)

    if ref:
        fuse_wkc = False
        fuse_wvc = False

    q_input = q_extend
    H = q_input.shape[1]
    
    if not fuse_wkc:  # 1st gemm        
        k_extend_c = torch.einsum('zc,hcd->zhd', k_extend[..., :kv_lora_rank].squeeze().to(dtype), w_kc.to(dtype) * w_descale) 
        k_extend_r = k_extend[..., kv_lora_rank:].repeat(1, H, 1)
        
        k_buffer_c = torch.einsum('zc,hcd->zhd', k_buffer[..., :kv_lora_rank].squeeze().to(dtype), w_kc.to(dtype) * w_descale)
        k_buffer_r = k_buffer[..., kv_lora_rank:].repeat(1, H, 1)

        k_extend = torch.cat((k_extend_c, k_extend_r), dim=-1).to(q_input.dtype)
        k_buffer = torch.cat((k_buffer_c, k_buffer_r), dim=-1).to(q_input.dtype)

    if not fuse_wvc:  # 2nd gemm
        v_extend = torch.einsum('zc,hcd->zhd', v_extend.squeeze().to(dtype), w_vc.to(dtype) * w_descale)
        v_buffer = torch.einsum('zc,hcd->zhd', v_buffer.squeeze().to(dtype), w_vc.to(dtype) * w_descale)

    if ref:
        extend_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    else:
        if persistent:
            extend_persistent_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
        else:
            extend_fused_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap,
                                    fuse_w_kc=fuse_wkc, fuse_w_vc=fuse_wvc, w_kc=w_kc, w_vc=w_vc, w_descale=w_descale, fp8=fp8, qk_rope_head_dim=qk_rope_head_dim, qk_nope_head_dim=v_head_dim, kv_lora_rank=kv_lora_rank)

    return out
    
def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
        (16, 16, 0, 8192, 512, 64, 128, "normal"),
        (16, 16, 16324, 1024, 512, 64, 128, "absorb"),
    ]
    
    if args.B or args.attn_impl != "":
        x_vals_list = [
            (args.B, 16, args.prefix_len, args.extend_len, 512, 64, 128, args.attn_impl),
        ]

    x_names = ["B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "attn_impl"]
    line_vals = ["ref", "fused", "persistent"]

    if args.ref:
        line_vals = ["ref"]
    if args.fused:
        line_vals = ["fused"]

    plot_name = "MLA-decode-fuse_wkc-{}-fuse_wvc-{}".format(args.fuse_wkc, args.fuse_wvc)

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-'), ('blue', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, attn_impl, sm_scale, logit_cap, device, provider):
        warmup = 25
        rep = 100

        # Prepare inputs
        q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, w_descale = input_helper_fused(
            B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)

        if attn_impl == "normal":
            forward = forward_normal
        else:
            forward = forward_absorb

        # Define the function to benchmark based on provider
        if "fused" in provider:
            def fn():
                return forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, 
                                     custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                     kv_lora_rank, qk_rope_head_dim, v_head_dim, w_kc, w_vc, w_descale, ref=False, fuse_wkc=args.fuse_wkc, fuse_wvc=args.fuse_wvc, fp8=args.fp8)
        
        if "persistent" in provider:
            def fn():
                return forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, 
                                     custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                     kv_lora_rank, qk_rope_head_dim, v_head_dim, w_kc, w_vc, w_descale, ref=False, persistent=True, fuse_wkc=False, fuse_wvc=False, fp8=False)
        
        
        elif "ref" in provider:
            def fn():
                return forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, 
                                     custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                     kv_lora_rank, qk_rope_head_dim, v_head_dim, w_kc, w_vc, w_descale, ref=True, fp8=False)
        
        # warmup
        if args.cuda_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            
            # Use CUDA graph for benchmarking
            torch.cuda.synchronize()  # Synchronize before capturing
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()  # Capture the function execution
            torch.cuda.synchronize()  # Synchronize after capturing
            func = g.replay
        else:
            func = fn

        # Replay the graph for benchmarking
        ms = triton.testing.do_bench(func, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=None, print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='bf16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-fused", action="store_true", default=False)
    parser.add_argument("-ref", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-fuse_wkc", action="store_true", default=False)
    parser.add_argument("-fuse_wvc", action="store_true", default=False)
    parser.add_argument("-attn_impl", type=str, default="")
    parser.add_argument("-cuda_graph", action="store_true", default=False)
    parser.add_argument("-B", type=int, default=0)
    parser.add_argument("-prefix_len", type=int, default=0)
    parser.add_argument("-extend_len", type=int, default=4096)
    parser.add_argument("-fp8", action="store_true", default=False)

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
        if re.match(r"^\s*MLA-decode", line):
            vgpr_info.append(line.strip())
            in_table = True
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
    # run_bench(args)
    test_op_fwd(16, 16, 0, 8192, 512, 64, 128, torch.bfloat16, "absorb", True, False, False, 1.0, 0.0, "cuda") # sanity check for function body correctness

if __name__ == "__main__":
    main()


