import sys
import itertools

def parse_range(rng_str):
    """Convert a string like 2:17,82:97 into a set of integers."""
    regs = set()
    parts = rng_str.strip().split(',')
    for part in parts:
        if ':' in part:
            start, end = map(int, part.split(':'))
            regs.update(range(start, end + 1))
        else:
            regs.add(int(part))
    return regs

def compress_ranges(regs):
    """Convert a set of integers into a compact range string."""
    if not regs:
        return ""
    regs = sorted(regs)
    ranges = []
    start = prev = regs[0]
    for r in regs[1:]:
        if r == prev + 1:
            prev = r
        else:
            ranges.append(f"{start}" if start == prev else f"{start}:{prev}")
            start = prev = r
    ranges.append(f"{start}" if start == prev else f"{start}:{prev}")
    return ",".join(ranges)

def read_register_file(filename):
    """Read symbol: count;range from file and return dict of symbol to set of registers."""
    symbol_map = {}
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                symbol, rest = line.strip().split(':', 1)
                _, range_str = rest.strip().split(';', 1)
                symbol_map[symbol.strip()] = parse_range(range_str.strip())
            except ValueError:
                print(f"Skipping malformed line: {line}")
    return symbol_map

def compute_overlaps(symbol_map):
    symbols = list(symbol_map.keys())
    for sym1, sym2 in itertools.combinations(symbols, 2):
        reg1 = symbol_map[sym1]
        reg2 = symbol_map[sym2]
        overlap = reg1 & reg2
        compressed = compress_ranges(overlap)
        print(f"{sym1}-{sym2}: {len(overlap)};{compressed}")

    # Total register usage
    all_regs = set()
    for regs in symbol_map.values():
        all_regs.update(regs)

    total_range = compress_ranges(all_regs)
    print(f"total: {len(all_regs)};{total_range}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_register_overlap.py <register_file.txt>")
        sys.exit(1)

    filename = sys.argv[1]
    symbol_map = read_register_file(filename)
    compute_overlaps(symbol_map)
