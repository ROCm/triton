import argparse
import json
import os
import sys
from typing import List, Set


def read_expected_failures(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    expected: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected.add(line)
    return expected


def extract_failures_from_lit_json(json_path: str) -> List[str]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing lit results file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    failing: List[str] = []
    for t in tests:
        code = t.get("code") or t.get("status")
        if code in ("FAIL", "XPASS", "UNRESOLVED", "TIMEOUT"):
            name = t.get("name") or t.get("file") or t.get("path")
            if isinstance(name, list):
                name = os.path.join(*name)
            if name:
                failing.append(name)
    return failing


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare lit JSON failures against expected list.")
    parser.add_argument("--lit-json", required=True, help="Path to llvm-lit JSON results (-o file)")
    parser.add_argument("--expected", required=True, help="Path to failing-tests.txt")
    parser.add_argument("--warn-only", action="store_true", help="Do not change exit code; print warnings only")
    args = parser.parse_args()

    expected = read_expected_failures(args.expected)
    actual_failures = set(extract_failures_from_lit_json(args.lit_json))

    unexpected_failures = sorted(actual_failures - expected)
    resolved_failures = sorted(expected - actual_failures)

    summary = {
        "unexpected_failures": unexpected_failures,
        "resolved_failures": resolved_failures,
        "expected_failures": sorted(expected),
    }
    print(json.dumps(summary, indent=2))

    # Emit human-friendly warnings to stderr
    if resolved_failures:
        print("Resolved expected failures (now passing):", file=sys.stderr)
        for t in resolved_failures:
            print(f"  - {t}", file=sys.stderr)
    if unexpected_failures:
        print("New failing tests (not in failing-tests.txt):", file=sys.stderr)
        for t in unexpected_failures:
            print(f"  - {t}", file=sys.stderr)
    else:
        print("All failing tests were expected (in failing-tests.txt):", file=sys.stderr)
    if args.warn_only:
        return 0

    return 1 if unexpected_failures else 0


if __name__ == "__main__":
    sys.exit(main())
