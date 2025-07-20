import re
import sys
from pathlib import Path

def process_assembly(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    loop_label = None
    loop_start_idx = loop_end_idx = None

    # 1. Find the loop start
    for i, line in enumerate(lines):
        if "=>This Inner Loop Header: Depth=1" in line:
            match = re.match(r'(\.\S+):', line.strip())
            if match:
                loop_label = match.group(1)
                loop_start_idx = i
                break

    if loop_label is None:
        print("Error: Loop header not found.")
        sys.exit(1)

    # 2. Find the loop end
    for i in range(loop_start_idx + 1, len(lines)):
        if lines[i].strip() == f"s_cbranch_scc1 {loop_label}":
            loop_end_idx = i
            break

    if loop_end_idx is None:
        print("Error: Loop end not found.")
        sys.exit(1)

    # 3. Extract loop body
    loop_body = lines[loop_start_idx + 1:loop_end_idx]

    # 4. Split loop by sched_barrier
    clusters = []
    current_cluster = []

    for line in loop_body:
        if "; sched_barrier" in line:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
        else:
            current_cluster.append(line)

    if current_cluster:
        clusters.append(current_cluster)

    # 5. Set fixed output dir relative to the script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output_clusters"
    output_dir.mkdir(exist_ok=True)

    # 6. Write cluster_i.txt files
    for i, cluster in enumerate(clusters):
        cluster_path = output_dir / f"cluster_{i}.txt"
        with open(cluster_path, "w") as f:
            f.writelines(cluster)

    print(f"✅ Extracted {len(clusters)} clusters to: {output_dir}")

    # 7. Sort v_ instructions in cluster_0 and cluster_2
    for i in [0, 2]:
        if i >= len(clusters):
            continue

        instr_groups = {}

        for line in clusters[i]:
            line = line.strip()
            if line.startswith("v_"):
                match = re.match(r'v_([a-zA-Z0-9]+)_', line)
                if match:
                    instrname = match.group(1)
                    instr_groups.setdefault(instrname, []).append(line + "\n")

        for instrname, lines in sorted(instr_groups.items()):
            out_path = output_dir / f"cluster_{i}_{instrname}.txt"
            with open(out_path, "w") as f:
                f.writelines(lines)

        if instr_groups:
            print(f"✅ Sorted v_ instructions in cluster_{i} to separate files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_and_sort_clusters.py <assembly_file.s>")
        sys.exit(1)

    asm_file = sys.argv[1]
    process_assembly(asm_file)
