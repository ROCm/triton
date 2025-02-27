from utils.extend_attention import extend_attention_fwd, extend_fused_attention_fwd


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

def input_helper(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, dtype, device):
    torch.manual_seed(0)
    q_extend = torch.randn(B * extend_length, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # kv_cache = torch.randn(B * (prefix_length + extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]
    o_extend = torch.empty(B*extend_length, H, kv_lora_rank, dtype=dtype, device=device)

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

    return q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend


def input_helper_fused(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device):
    torch.manual_seed(0)
    q_extend = torch.randn(B * extend_length, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]
    o_extend = torch.empty(B*extend_length, H, v_head_dim, dtype=dtype, device=device)

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


    return q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc

@pytest.mark.parametrize("B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim", [
    (2, 16, 1024, 1024, 512, 64, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('attn_impl', ["absorbed"])
@pytest.mark.parametrize('fuse_gemms', [True])
def test_op_fwd(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, attn_impl, fuse_gemms, device="cuda"):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    
    if fuse_gemms:
        q_extend, k_extend, v_extend, tri_out, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper_fused(
                        B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
    else:
        q_extend, k_extend, v_extend, tri_out, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend = input_helper(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, dtype, device)
        w_kc, w_vc = None, None

    extend_fused_attention_fwd(q_extend, k_extend, v_extend, tri_out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=1.0,
                                fuse_gemms=fuse_gemms, w_kc=w_kc, w_vc=w_vc, absorb_w_kc=True)
    
    # reference implementation
    if fuse_gemms:
        if attn_impl == "absorbed":
            q_input = torch.empty((*q_extend.shape[:-1],kv_lora_rank+qk_rope_head_dim), dtype=q_extend.dtype, device=q_extend.device)
            q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
            q_nope = q_extend[..., :v_head_dim]
            q_nope = torch.einsum("hzd,hdc->hzc", q_nope.transpose(0, 1), w_kc.transpose(1,2))
            q_input[..., :kv_lora_rank] = q_nope.transpose(0, 1)
            
            tmp_out = torch.empty((*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)
        else:
            q_input = q_extend
        
            k_extend_c = torch.einsum('zc,hcd->zhd', k_extend[..., :kv_lora_rank], w_kc)
            k_extend_r = k_extend[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
            k_extend = torch.cat((k_extend_c, k_extend_r), dim=-1)
            
            k_buffer_c = torch.einsum('zc,hcd->zhd', k_buffer[..., :kv_lora_rank], w_kc)
            k_buffer_r = k_buffer[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
            k_buffer = torch.cat((k_buffer_c, k_buffer_r), dim=-1)
            
            v_extend = torch.einsum('zc,hcd->zhd', v_extend, w_vc)
            v_buffer = torch.einsum('zc,hcd->zhd', v_buffer, w_vc)
            
            tmp_out = torch.empty((*q_extend.shape[:-1], v_head_dim), dtype=q_extend.dtype, device=q_extend.device)
    else:
        q_input = q_extend
        tmp_out = torch.empty((*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)

    
    extend_attention_fwd(q_input, k_extend, v_extend, tmp_out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=1.0)
    
    if not fuse_gemms: # sanity check for the function body correctness without gemm fusion
        torch.testing.assert_close(tmp_out, tri_out, atol=1e-2, rtol=1e-2)
        print("Function body matches!")
    else:
        if attn_impl == "absorbed":
            attn_bmm_output = torch.einsum("hzc,hcd->hzd", tmp_out.transpose(0, 1), w_vc)
            attn_output = attn_bmm_output.transpose(0, 1)
            ref_out = attn_output
        else:
            ref_out = tmp_out
        
        # print(f"ref: {ref_out[124, 8, 92]}")
        # print(f"tri: {tri_out[124, 8, 92]}")
        
        print("first 10 outputs:")
        print(f"ref: {ref_out.flatten()[:]}") 
        print(f"tri: {tri_out.flatten()[:]}") 
        torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
        print("Output matches!")


def ref_forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, 
                       fuse_gemms, w_kc, w_vc, attn_impl, kv_lora_rank, qk_rope_head_dim, v_head_dim, H):
    if fuse_gemms:
        if attn_impl == "absorbed":
            q_input = torch.empty((*q_extend.shape[:-1],kv_lora_rank+qk_rope_head_dim), dtype=q_extend.dtype, device=q_extend.device)
            q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
            q_nope = q_extend[..., :v_head_dim]
            q_nope = torch.bmm(q_nope.transpose(0, 1), w_kc.transpose(1,2))
            q_input[..., :kv_lora_rank] = q_nope.transpose(0, 1)
            
            tmp_out = torch.empty((*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)
        else:
            q_input = q_extend
        
            k_extend_c = torch.einsum('zc,hcd->zhd', k_extend[..., :kv_lora_rank], w_kc)
            k_extend_r = k_extend[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
            k_extend = torch.cat((k_extend_c, k_extend_r), dim=-1)
            
            k_buffer_c = torch.einsum('zc,hcd->zhd', k_buffer[..., :kv_lora_rank], w_kc)
            k_buffer_r = k_buffer[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
            k_buffer = torch.cat((k_buffer_c, k_buffer_r), dim=-1)
            
            v_extend = torch.einsum('zc,hcd->zhd', v_extend, w_vc)
            v_buffer = torch.einsum('zc,hcd->zhd', v_buffer, w_vc)
            
            tmp_out = torch.empty((*q_extend.shape[:-1], v_head_dim), dtype=q_extend.dtype, device=q_extend.device)
    else:
        q_input = q_extend
        tmp_out = torch.empty((*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)

    
    extend_attention_fwd(q_input, k_extend, v_extend, tmp_out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend)
    
    if not fuse_gemms: # sanity check for the function body correctness without gemm fusion
        return tmp_out
    else:
        if attn_impl == "absorbed":
            attn_bmm_output = torch.bmm(tmp_out.transpose(0, 1), w_vc)
            attn_output = attn_bmm_output.transpose(0, 1)
            ref_out = attn_output
        else:
            ref_out = tmp_out

    return ref_out




def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
                    (1, 16, 1024, 1024, 512, 64, 128),
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
        do_gemms = args.do_gemms

        if do_gemms:
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper_fused(
                B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
        else:
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend = input_helper(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, dtype, device)
            w_kc, w_vc = None, None
        
        if "ref" in provider:
            # print(q_extend.flatten()[:10])
            fn = lambda: ref_forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend,
                               fuse_gemms=do_gemms, w_kc=w_kc, w_vc=w_vc, attn_impl="absorbed", kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim, H=H)
            # ret = fn()
            # print(ret.flatten()[:10])
            # print(args.do_gemms)

        if "fused" in provider:
            # print(q_extend.flatten()[:10])
            fn = lambda: extend_fused_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend,
                                                    fuse_gemms=do_gemms, w_kc=w_kc, w_vc=w_vc, absorb_w_kc=False)
            # fn()
            # print(o_extend.flatten()[:10])

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
    parser.add_argument("-B", type=int, default=0)
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
        # os.environ["TRITON_ALWAYS_COMPILE"] = "1"
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


if __name__ == "__main__":
    main()


