import re
import sys
from collections import deque


# -------------------------------
# Helpers
# -------------------------------
def escape_shape(shape: str) -> str:
    """Escape < and > for Mermaid"""
    return shape.replace("<", "&lt;").replace(">", "&gt;")


# -------------------------------
# Parse MLIR lines
# -------------------------------
def parse_mlir_lines(lines):
    """
    Parse MLIR lines into a mapping: var -> (operation, dependencies, shape)
    Handles:
    - last ':' as delimiter for code/type
    - extract output type
    - simplify tensor shapes
    - extract op name correctly
    """
    ir_map = {}
    dep_pattern = re.compile(r"(%[\w\d]+)")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split at last colon
        last_colon = line.rfind(':')
        if last_colon == -1:
            continue
        code_part = line[:last_colon].strip()
        type_part = line[last_colon + 1:].strip()

        # Extract variable
        m_var = re.match(r"^(%[\w\d]+)\s*=", code_part)
        if not m_var:
            continue
        var = m_var.group(1)

        # Extract op name: first token after '='
        eq_index = code_part.find('=')
        op_name = code_part[eq_index + 1:].strip().split()[0]

        # Extract dependencies
        deps = dep_pattern.findall(code_part)
        deps = [d for d in deps if d != var]

        # Extract output type
        if '->' in type_part:
            _, out_type = type_part.split('->', 1)
            out_type = out_type.strip()
        else:
            out_type = type_part

        # Tensor or scalar
        if out_type.startswith('tensor<'):
            inner = out_type[len('tensor<'):].rstrip('>')
            shape = inner.split(',')[0].strip()
        else:
            shape = out_type

        # Escape for Mermaid
        shape = shape.replace("<", "&lt;").replace(">", "&gt;")

        ir_map[var] = {"op": op_name, "deps": deps, "shape": shape, "full": f"{var} = {code_part} : {shape}"}

    return ir_map


# -------------------------------
# Build backward dependency chain
# -------------------------------
def build_dependency_chain(ir_map, target_var):
    visited = set()
    queue = deque([target_var])
    chain = {}

    while queue:
        cur = queue.popleft()
        if cur in visited or cur not in ir_map:
            continue
        visited.add(cur)
        node = ir_map[cur]
        chain[cur] = node
        for dep in node["deps"]:
            if dep not in visited:
                queue.append(dep)
    return chain


# -------------------------------
# Generate Mermaid graph
# -------------------------------
def generate_mermaid(chain):
    lines = []
    lines.append("flowchart TD")

    # Sort numeric variables first, then others
    sorted_nodes = sorted(chain.keys(), key=lambda x: (0, int(x[1:])) if x[1:].isdigit() else (1, x))

    # Create nodes
    for var in sorted_nodes:
        node = chain[var]
        label = f"{var} = {node['op']} {' '.join(node['deps'])} : {node['shape']}"
        lines.append(f'    {var[1:]}["{label}"]')

    # Create edges
    for var in sorted_nodes:
        for dep in chain[var]["deps"]:
            if dep in chain:
                lines.append(f"    {var[1:]} --> {dep[1:]}")

    # Highlight special ops
    for var in sorted_nodes:
        op = chain[var]["op"]
        if "make_range" in op:
            lines.append(f"    style {var[1:]} fill:#d4f5d4,stroke:#2e7d32,stroke-width:2px")
        elif "splat" in op:
            lines.append(f"    style {var[1:]} fill:#ffe6cc,stroke:#e65100,stroke-width:2px")

    return "\n".join(lines)


# -------------------------------
# Main
# -------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_mermaid.py <mlir_file> <target_var>")
        sys.exit(1)

    mlir_file = sys.argv[1]
    target_var = sys.argv[2]

    with open(mlir_file, "r") as f:
        lines = f.readlines()

    ir_map = parse_mlir_lines(lines)

    if target_var not in ir_map:
        print(f"Error: {target_var} not found in IR.")
        sys.exit(1)

    chain = build_dependency_chain(ir_map, target_var)
    mermaid_code = generate_mermaid(chain)

    print(mermaid_code)


if __name__ == "__main__":
    main()
