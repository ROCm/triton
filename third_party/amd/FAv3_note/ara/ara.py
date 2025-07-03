#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def run_python(script, *args):
    """Helper to run a Python script and return its stdout."""
    result = subprocess.run(
        [sys.executable, script, *map(str, args)],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <input_asm_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()
    script_dir = Path(__file__).resolve().parent

    # Paths to other scripts
    split_script = script_dir / "split.py"
    analyze_script = script_dir / "analyze.py"
    overlap_script = script_dir / "overlap.py"

    # Run split.py
    subprocess.run([sys.executable, str(split_script), str(input_file)], check=True)

    # Analyze specific files and columns
    output_dir = script_dir / "output_clusters"
    results = {
        "QK": run_python(analyze_script, output_dir / "cluster_0_mfma.txt", 0),
        "K":  run_python(analyze_script, output_dir / "cluster_0_mfma.txt", 1),
        "PP": run_python(analyze_script, output_dir / "cluster_2_exp.txt", 0),
        "P":  run_python(analyze_script, output_dir / "cluster_0_cvt.txt", 0),
        "V":  run_python(analyze_script, output_dir / "cluster_2_mfma.txt", 1),
        "Q":  run_python(analyze_script, output_dir / "cluster_0_mfma.txt", 2),
        "ACC":  run_python(analyze_script, output_dir / "cluster_2_mfma.txt", 0),
    }

    # Write register usage summary
    reg_usage_path = script_dir / "reg_usage.txt"
    with open(reg_usage_path, "w") as f:
        for key in ["QK", "PP", "P", "K", "V", "Q", "ACC"]:
            f.write(f"{key}: {results[key]}\n")

    # Run overlap.py
    subprocess.run([sys.executable, str(overlap_script), str(reg_usage_path)], check=True)

if __name__ == "__main__":
    main()
