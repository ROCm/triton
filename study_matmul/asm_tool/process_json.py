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
from glob import glob


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
    names = [ins[0].lower() for ins in sorted_code]

    max_hit = max(hitcounts)
    loop_first_pos = next(i for i, h in enumerate(hitcounts) if h == max_hit)
    loop_last_pos = max(i for i, h in enumerate(hitcounts) if h == max_hit)
    epilogue_first_pos = loop_last_pos + 1 if loop_last_pos + 1 < len(hitcounts) else None

    # Compute iterations: hitcount_loop / hitcount_epilogue
    loop_hit = hitcounts[loop_first_pos]
    epilogue_hit = hitcounts[epilogue_first_pos] if epilogue_first_pos is not None else 1
    num_iterations = loop_hit / epilogue_hit if epilogue_hit != 0 else None

    mfma_count = sum(1 for i in range(loop_first_pos, loop_last_pos + 1) if "mfma" in names[i])

    return {
        "loop_first_index": indices[loop_first_pos],
        "epilogue_first_index": None if epilogue_first_pos is None else indices[epilogue_first_pos],
        "mfma_count_in_loop": mfma_count,
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

    for clock, _, _, _, idx in instructions:
        if loop_clock is None and idx == loop_index:
            loop_clock = clock
        if epilogue_clock is None and idx == epilogue_index:
            epilogue_clock = clock
        if loop_clock is not None and epilogue_clock is not None:
            break

    if loop_clock is None or epilogue_clock is None:
        return None

    return epilogue_clock - loop_clock


def analyze_waves(folder, loop_index, epilogue_index):
    """Process se0_sm0_sl0_wv*.json files and compute durations + average."""
    pattern = os.path.join(folder, "se0_sm0_sl0_wv*.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    durations = {}
    for path in files:
        dur = process_wave_file(path, loop_index, epilogue_index)
        if dur is not None:
            durations[os.path.basename(path)] = dur

    if not durations:
        raise ValueError("No valid loop durations found in any se0_sm0_sl0_wv*.json file")

    avg_duration = sum(durations.values()) / len(durations)
    return durations, avg_duration


def main():
    parser = argparse.ArgumentParser(description="Compute average loop and per-iteration durations.")
    parser.add_argument("folder", help="Folder containing code.json and se0_sm0_sl0_wv*.json files")
    args = parser.parse_args()

    try:
        code_list = load_code_json(args.folder)
        code_info = analyze_code(code_list)

        durations, avg_loop_duration = analyze_waves(
            args.folder,
            code_info["loop_first_index"],
            code_info["epilogue_first_index"],
        )

        avg_iteration_duration = (avg_loop_duration / code_info["num_iterations"]
                                  if code_info["num_iterations"] and code_info["num_iterations"] > 0 else None)

        mfma_efficiency = code_info["mfma_count_in_loop"] * 16 / avg_iteration_duration

        result = {
            **code_info,
            "wave_durations": durations,
            "average_loop_duration": avg_loop_duration,
            "average_iteration_duration": avg_iteration_duration,
            "mfma efficiency": f"{mfma_efficiency * 100:.2f}%",
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
