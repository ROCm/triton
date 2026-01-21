#!/usr/bin/env python3
"""
process_code_and_wave_avg_with_iterations.py

Usage:
    python process_code_and_wave_avg_with_iterations.py /path/to/folder

Description:
  1. Reads code.json to find:
        - first instruction index in loop
        - first instruction index in epilogue
        - hitcounts to compute number of iterations
  2. Processes se0_sm0_sl0_wv*.json files.
  3. For each wave:
        - finds first occurrence clock of loop entry and epilogue entry
        - computes loop duration = epilogue_clock - loop_clock
  4. Reports per-wave durations, average loop duration,
     and average duration per iteration.
"""

import argparse
import json
import os
import sys
import re
from glob import glob

# MFMA instruction cycle mapping for non-scaled versions
MFMA_CYCLE_MAP = {
    "v_mfma_f32_16x16x32_f16": 16,
    "v_mfma_f32_32x32x16_f16": 32,
    #"v_mfma_f32_4x4x4_f16": 4,
    #"v_mfma_f32_16x16x4_f32": 16,
    #"v_mfma_f32_32x32x2_f32": 32,
    #"v_mfma_f32_4x4x1_f32": 4,
    # Add more standard MFMA instructions as needed
}

# MFMA scaled instruction cycle rules
MFMA_SCALE_CYCLE_RULES = {
    "v_mfma_scale_f32_16x16x128_f8f6f4": {"cbsz_or_blgp_le_1": 32, "default": 16},
    "v_mfma_scale_f32_32x32x64_f8f6f4": {"cbsz_or_blgp_le_1": 64, "default": 32},
    # Add more scaled MFMA instructions as needed
}


def extract_cbsz_blgp(instruction):
    """
    Extract cbsz and blgp values from an MFMA instruction.

    Args:
        instruction: String containing the MFMA instruction

    Returns:
        tuple: (cbsz_value, blgp_value) or (None, None) if not found
    """
    cbsz_match = re.search(r'cbsz:(\d+)', instruction)
    blgp_match = re.search(r'blgp:(\d+)', instruction)

    cbsz = int(cbsz_match.group(1)) if cbsz_match else None
    blgp = int(blgp_match.group(1)) if blgp_match else None

    return cbsz, blgp


def get_mfma_opcode(instruction):
    """
    Extract the MFMA opcode from the instruction.

    Args:
        instruction: String containing the MFMA instruction

    Returns:
        str: The opcode (e.g., "v_mfma_f32_16x16x32_f16")
    """
    # Match pattern like "v_mfma_..." up to the first space or end
    match = re.match(r'(v_mfma_\S+?)(?:\s|$)', instruction.strip())
    if match:
        return match.group(1)
    return None


def get_mfma_cycles(instruction):
    """
    Determine the cycle count for an MFMA instruction.

    Args:
        instruction: String containing the MFMA instruction

    Returns:
        int: Number of cycles for the instruction, or 0 if not an MFMA
    """
    instruction_lower = instruction.lower()

    # Check if it's an MFMA instruction
    if "v_mfma" not in instruction_lower:
        return 0

    # Extract the opcode
    opcode = get_mfma_opcode(instruction_lower)
    if not opcode:
        return 0

    # Check if it's a scaled MFMA instruction
    is_scale = "v_mfma_scale" in opcode

    if is_scale:
        # Find matching rule in MFMA_SCALE_CYCLE_RULES
        matching_rule = None
        for rule_opcode in MFMA_SCALE_CYCLE_RULES:
            if rule_opcode in opcode:
                matching_rule = rule_opcode
                break

        if not matching_rule:
            # Unknown scaled MFMA, return 0 or a default value
            print(f"Warning: Unknown scaled MFMA instruction: {opcode}", file=sys.stderr)
            return 0

        # Extract cbsz and blgp values
        cbsz, blgp = extract_cbsz_blgp(instruction_lower)

        # Apply the rule: if either cbsz or blgp is <= 1
        rule = MFMA_SCALE_CYCLE_RULES[matching_rule]
        if (cbsz is not None and cbsz <= 1) or (blgp is not None and blgp <= 1):
            return rule["cbsz_or_blgp_le_1"]
        else:
            return rule["default"]

    else:
        # Non-scaled MFMA instruction
        # Try to find exact match in the cycle map
        if opcode in MFMA_CYCLE_MAP:
            return MFMA_CYCLE_MAP[opcode]

        # If not found, try to infer from dimensions
        # Pattern: v_mfma_<type>_<M>x<N>x<K>_<input_type>
        match = re.search(r'v_mfma_\w+_(\d+)x(\d+)x\d+', opcode)
        if match:
            m_dim = int(match.group(1))
            n_dim = int(match.group(2))
            # Typically, cycle count equals max(M, N)
            cycles = max(m_dim, n_dim)
            print(f"Warning: Inferred cycles for {opcode}: {cycles}", file=sys.stderr)
            return cycles

        # Unknown MFMA instruction
        print(f"Warning: Unknown MFMA instruction: {opcode}", file=sys.stderr)
        return 0


def load_code_json(folder):
    path = os.path.join(folder, "code.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"code.json not found in {folder}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "code" in data:
        return data["code"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected code.json format")


def analyze_code(code_list):
    """Extract loop info and compute iteration count."""
    sorted_code = sorted(code_list, key=lambda x: x[2])
    hitcounts = [ins[6] for ins in sorted_code]
    indices = [ins[2] for ins in sorted_code]
    names = [ins[0] for ins in sorted_code]  # Keep original case for instruction parsing

    max_hit = max(hitcounts)
    loop_first_pos = next(i for i, h in enumerate(hitcounts) if h == max_hit)
    loop_last_pos = max(i for i, h in enumerate(hitcounts) if h == max_hit)
    epilogue_first_pos = loop_last_pos + 1 if loop_last_pos + 1 < len(hitcounts) else None

    # Compute iterations: hitcount_loop / hitcount_epilogue
    loop_hit = hitcounts[loop_first_pos]
    epilogue_hit = hitcounts[epilogue_first_pos] if epilogue_first_pos is not None else 1
    num_iterations = loop_hit / epilogue_hit if epilogue_hit != 0 else None

    # Count MFMA instructions and their total cycles
    total_mfma_cycles = 0
    mfma_count = 0
    for i in range(loop_first_pos, loop_last_pos + 1):
        cycles = get_mfma_cycles(names[i])
        if cycles > 0:
            mfma_count += 1
            total_mfma_cycles += cycles

    return {
        "loop_first_index": indices[loop_first_pos],
        "epilogue_first_index": None if epilogue_first_pos is None else indices[epilogue_first_pos],
        "mfma_count_in_loop": mfma_count,
        "total_mfma_cycles_in_loop": total_mfma_cycles,
        "loop_hitcount": loop_hit,
        "epilogue_hitcount": epilogue_hit,
        "num_iterations": num_iterations,
    }


def process_wave_file(path, loop_index, epilogue_index):
    """Find first occurrence clocks in a wave JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "wave" not in data or "instructions" not in data["wave"]:
        return None

    instructions = data["wave"]["instructions"]
    loop_clock = None
    epilogue_clock = None
    start_clock = None
    end_clock = None

    for clock, _, _, _, idx in instructions:
        if loop_clock is None and idx == loop_index:
            loop_clock = clock
        if epilogue_clock is None and idx == epilogue_index:
            epilogue_clock = clock
        if loop_clock is not None and epilogue_clock is not None:
            break

    start_clock = instructions[0][0]
    end_clock = instructions[-1][0]

    if loop_clock is None or epilogue_clock is None:
        return None

    pro = loop_clock - start_clock
    epi = end_clock - epilogue_clock

    return pro, epilogue_clock - loop_clock, epi


def analyze_waves(folder, loop_index, epilogue_index):
    """Process se0_sm0_sl0_wv*.json files and compute durations + average."""
    pattern = os.path.join(folder, "se0_sm0_sl0_wv*.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    durations = {}
    pro_dur = {}
    epi_dur = {}
    for path in files:
        result = process_wave_file(path, loop_index, epilogue_index)
        if result is not None:
            pro, dur, epi = result
            durations[os.path.basename(path)] = dur
            pro_dur[os.path.basename(path)] = pro
            epi_dur[os.path.basename(path)] = epi

    if not durations:
        raise ValueError("No valid loop durations found in any se0_sm0_sl0_wv*.json file")

    avg_duration = sum(durations.values()) / len(durations)
    avg_pro_dur = sum(pro_dur.values()) / len(pro_dur)
    avg_epi_dur = sum(epi_dur.values()) / len(epi_dur)
    return durations, avg_duration, pro_dur, avg_pro_dur, epi_dur, avg_epi_dur


def main():
    parser = argparse.ArgumentParser(description="Compute average loop and per-iteration durations.")
    parser.add_argument("folder", help="Folder containing code.json and se0_sm0_sl0_wv*.json files")
    args = parser.parse_args()

    try:
        code_list = load_code_json(args.folder)
        code_info = analyze_code(code_list)

        durations, avg_loop_duration, _, avg_pro, _, avg_epi = analyze_waves(
            args.folder,
            code_info["loop_first_index"],
            code_info["epilogue_first_index"],
        )

        avg_iteration_duration = (avg_loop_duration / code_info["num_iterations"]
                                  if code_info["num_iterations"] and code_info["num_iterations"] > 0 else None)

        mfma_efficiency = (code_info["total_mfma_cycles_in_loop"] /
                           avg_iteration_duration if avg_iteration_duration and avg_iteration_duration > 0 else None)
        total_dur = avg_loop_duration + avg_pro + avg_epi

        result = {
            **code_info,
            "wave_durations": durations,
            "average_loop_duration": avg_loop_duration,
            "average_prologue_duration": avg_pro,
            "average_epilogue_duration": avg_epi,
            "pro_ratio": f"{avg_pro / total_dur * 100:.2f}%",
            "loop_ratio": f"{avg_loop_duration / total_dur * 100:.2f}%",
            "epi_ratio": f"{avg_epi / total_dur * 100:.2f}%",
            "average_iteration_duration": avg_iteration_duration,
            "mfma efficiency": f"{mfma_efficiency * 100:.2f}%" if mfma_efficiency is not None else "N/A",
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
