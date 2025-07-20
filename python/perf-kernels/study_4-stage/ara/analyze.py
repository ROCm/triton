import re
import sys

def extract_registers(text):
    """Extract register indices from a string like v5, v[3:7]"""
    regs = set()

    # Match v[3:7]
    for match in re.finditer(r'v\[(\d+):(\d+)\]', text):
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end:
            regs.update(range(start, end + 1))

    # Match v5 (but not v[...])
    for match in re.finditer(r'\bv(\d+)\b(?!\[)', text):
        regs.add(int(match.group(1)))

    return regs

def compress_ranges(reg_list):
    """Turn sorted list into x,y:z,w format"""
    if not reg_list:
        return ""

    reg_list = sorted(reg_list)
    ranges = []
    start = prev = reg_list[0]

    for r in reg_list[1:]:
        if r == prev + 1:
            prev = r
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}:{prev}")
            start = prev = r

    # Add last range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}:{prev}")

    return ",".join(ranges)

def analyze_file(filename, column_indices):
    used_regs = set()

    with open(filename) as f:
        for line in f:
            parts = line.strip().split(',')
            for col in column_indices:
                if col < len(parts):
                    used_regs.update(extract_registers(parts[col]))

    sorted_regs = sorted(used_regs)
    compressed = compress_ranges(sorted_regs)
    print(f"{len(used_regs)};{compressed}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_registers.py <file> <column_index1> [<column_index2> ...]")
        sys.exit(1)

    file_path = sys.argv[1]
    column_indices = [int(i) for i in sys.argv[2:]]
    analyze_file(file_path, column_indices)
