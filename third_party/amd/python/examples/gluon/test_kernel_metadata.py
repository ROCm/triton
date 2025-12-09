"""
This file tests and ensure that generated code for BF16 BSHD Flash Attention
kernel on GFX1250 do not have unexpected change or regression.
"""

# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import json
import re
import os
import pytest

from .f16_fa_gfx1250 import attn_fwd_pipelined_kernel as f16_attn_fwd_pipelined_kernel
from .f16_fa_gfx1250 import run_attention as run_f16_attention
from .mxfp_fa_gfx1250 import run_attention as run_mxfp_attention

GOLDEN_METADATA_OVERWRITE = (os.getenv('GOLDEN_METADATA_OVERWRITE', 'False').lower() in ('true', '1'))


def static_metadata_check(kernel, config: str):
    """Check kernel metadata against golden values.

    Args:
        kernel: The compiled kernel object with asm['amdgcn'] attribute
        config: The pytest config name (e.g., 'config0', 'config1')

    If GOLDEN_METADATA_OVERWRITE env var is set to True, updates the golden JSON file.
    Otherwise, loads the golden JSON and asserts values match.
    """
    amdgcn = kernel.asm['amdgcn']

    # Extract metadata from assembly
    sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
    sgpr_spill_count = int(re.search(r'\.sgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
    vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    scratch_size = int(re.search(r';\s+ScratchSize:\s+(\d+)', amdgcn).group(1))
    code_len_in_byte = int(re.search(r';\s+codeLenInByte\s+=\s+(\d+)', amdgcn).group(1))
    occupancy = int(re.search(r';\s+Occupancy:\s+(\d+)', amdgcn).group(1))

    # Current metadata values
    current_metadata = {
        "sgpr_count": sgpr_count,
        "sgpr_spill_count": sgpr_spill_count,
        "vgpr_count": vgpr_count,
        "vgpr_spill_count": vgpr_spill_count,
        "scratch_size": scratch_size,
        "code_len_in_byte": code_len_in_byte,
        "occupancy": occupancy,
    }

    golden_dir = os.path.dirname(os.path.realpath(__file__))
    golden_file = "golden_metadata.json"
    golden_filepath = os.path.join(golden_dir, golden_file)

    if GOLDEN_METADATA_OVERWRITE:
        # Load existing golden data or create new
        if os.path.exists(golden_filepath):
            with open(golden_filepath, 'r') as f:
                golden_data = json.load(f)
        else:
            golden_data = {}

        # Update with current metadata
        golden_data[config] = current_metadata

        # Write back to file
        with open(golden_filepath, 'w') as f:
            json.dump(golden_data, f, indent=2)
            # Writes EOF to comply with pre-commit.
            f.write("\n")

        print(f"Updated golden metadata for {config}")
        return

    # Load golden data and compare
    assert os.path.exists(golden_filepath), (f"Golden metadata file not found: {golden_filepath}. "
                                             f"Run with GOLDEN_METADATA_OVERWRITE=1 to create it.")

    with open(golden_filepath, 'r') as f:
        golden_data = json.load(f)

    assert config in golden_data, (f"Config '{config}' not found in golden metadata. "
                                   f"Available configs: {list(golden_data.keys())}")

    golden_metadata = golden_data[config]

    # Compare each metadata field
    for key in current_metadata.keys():
        assert key in golden_metadata, f"Key '{key}' not found in golden metadata for {config}"
        current_val = current_metadata[key]
        golden_val = golden_metadata[key]
        assert current_val == golden_val, (f"Metadata mismatch for {config}.{key}: "
                                           f"current={current_val}, golden={golden_val}")


def generate_f16_attention_configs():
    base_configs = [
        # Tests for pipelined attention fwd kernel
        pytest.param({
            "BATCH": 8,  #
            "SEQLEN_Q": 512, "SEQLEN_K": 512,  #
            "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8,  #
            "HEAD_SZ": 128,  #
            "BLOCK_M": 128, "BLOCK_N": 64,  #
            "ATTN_FN": f16_attn_fwd_pipelined_kernel,  #
        }),
        pytest.param({
            "BATCH": 8,  #
            "SEQLEN_Q": 1024, "SEQLEN_K": 1024,  #
            "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8,  #
            "HEAD_SZ": 128,  #
            "BLOCK_M": 128, "BLOCK_N": 128,  #
            "ATTN_FN": f16_attn_fwd_pipelined_kernel,  #
        }),
    ]
    return base_configs


@pytest.mark.parametrize("config", generate_f16_attention_configs())
def test_f16_attention_kernel_metadata(config):
    # TODO: figure out correctness issue and re-enable testing
    attn_kernel = run_f16_attention(config)

    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    attn_fn = "f16_attn_fwd_pipelined_kernel"

    # Generate config name from pytest request
    config_name = f"BATCH{BATCH}_SEQLENQ{SEQLEN_Q}_SEQLENK{SEQLEN_K}_QHEADS{NUM_Q_HEADS}_KVHEADS{NUM_K_HEADS}_HEADSZ{HEAD_SZ}_BM{BLOCK_M}_BN{BLOCK_N}_{attn_fn}"
    static_metadata_check(attn_kernel, config_name)


def generate_mxfp_attention_configs():
    base_configs = [
        # Tests for pipelined attention fwd kernel
        pytest.param({
            "q_type": "e4m3",
            "kv_type": "e4m3",  #
            "batch": 1,  #
            "seqlen_q": 1024,
            "seqlen_k": 1024,  #
            "num_q_heads": 1,
            "num_k_heads": 1,  #
            "head_sz": 128,  #
            "block_m": 128,
            "block_n": 128,  #
            "scale_type": "block",  #
            "p_k_width": 8,  #
            "subtile": False,
        }),
        pytest.param({
            "q_type": "e4m3",
            "kv_type": "e4m3",  #
            "batch": 1,  #
            "seqlen_q": 1024,
            "seqlen_k": 1024,  #
            "num_q_heads": 1,
            "num_k_heads": 1,  #
            "head_sz": 128,  #
            "block_m": 128,
            "block_n": 128,  #
            "scale_type": "global",  #
            "p_k_width": 8,  #
            "subtile": False,
        }),
        pytest.param({
            "q_type": "e4m3",
            "kv_type": "e4m3",  #
            "batch": 1,  #
            "seqlen_q": 1024,
            "seqlen_k": 1024,  #
            "num_q_heads": 1,
            "num_k_heads": 1,  #
            "head_sz": 128,  #
            "block_m": 256,
            "block_n": 128,  #
            "scale_type": "block",  #
            "p_k_width": 8,  #
            "subtile": True,
        }),
        pytest.param({
            "q_type": "e4m3",
            "kv_type": "e4m3",  #
            "batch": 1,  #
            "seqlen_q": 1024,
            "seqlen_k": 1024,  #
            "num_q_heads": 1,
            "num_k_heads": 1,  #
            "head_sz": 128,  #
            "block_m": 256,
            "block_n": 128,  #
            "scale_type": "global",  #
            "p_k_width": 8,  #
            "subtile": True,
        }),
    ]
    return base_configs


@pytest.mark.parametrize("config", generate_mxfp_attention_configs())
def test_mxfp_attention_kernel_metadata(config):
    config["pipelined"] = True
    config["disable_p_scaling"] = True
    config["scale_preshuffled"] = (config["scale_type"] == "block")
    attn_kernel = run_mxfp_attention(**config)

    QT = config["q_type"]
    KVT = config["kv_type"]
    BATCH = config["batch"]
    SEQLEN_Q = config["seqlen_q"]
    SEQLEN_K = config["seqlen_k"]
    NUM_Q_HEADS = config["num_q_heads"]
    NUM_K_HEADS = config["num_k_heads"]
    HEAD_SZ = config["head_sz"]
    BLOCK_M = config["block_m"]
    BLOCK_N = config["block_n"]
    SCALE_TYPE = config["scale_type"]
    PKWIDTH = config["p_k_width"]
    SUBTILE = config["subtile"]
    attn_fn = "mxfp_attn_fwd"
    if SUBTILE:
        attn_fn += "_subtile"
    if config["pipelined"]:
        attn_fn += "_pipelined"

    # Generate config name from pytest request
    config_name = f"{QT}x{KVT}_{SCALE_TYPE}_BATCH{BATCH}_SEQLENQ{SEQLEN_Q}_SEQLENK{SEQLEN_K}_QHEADS{NUM_Q_HEADS}_KVHEADS{NUM_K_HEADS}_HEADSZ{HEAD_SZ}_BM{BLOCK_M}_BN{BLOCK_N}_PKWIDTH{PKWIDTH}_{attn_fn}"
    static_metadata_check(attn_kernel, config_name)
