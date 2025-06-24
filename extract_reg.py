import re
import sys

def extract_registers(cell):
    """
    Extract all register indices from a string,
    supporting v[x:y] and vX formats, ignoring unrelated strings.
    """
    regs = set()

    # Extract v[x:y] ranges
    for match in re.finditer(r'v\[(\d+):(\d+)\]', cell):
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end:
            regs.update(range(start, end + 1))

    # Extract single-register vX (but not v[...] or invalid forms)
    # \b ensures whole word matching
    for match in re.finditer(r'\bv(\d+)\b(?!\[)', cell):
        regs.add(int(match.group(1)))

    return regs

def contiguous_ranges(indices):
    """Convert sorted list of integers into range strings like 0:3,5:7"""
    if not indices:
        return []

    indices = sorted(indices)
    ranges = []
    start = prev = indices[0]

    for num in indices[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append(f"{start}:{prev}")
            start = prev = num
    ranges.append(f"{start}:{prev}")
    return ranges

def extract_column_usage(file_path, column_index, delimiter=","):
    used_regs = set()

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if column_index >= len(parts):
                continue
            regs = extract_registers(parts[column_index])
            used_regs.update(regs)

    return contiguous_ranges(sorted(used_regs)), len(used_regs)

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file> <column_index>")
        sys.exit(1)

    filename = sys.argv[1]
    col_index = int(sys.argv[2])

    ranges, total_regs = extract_column_usage(filename, col_index)

    print("Register usage ranges:", ",".join(ranges))
    print("Total registers used:", total_regs)
