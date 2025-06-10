import json
from dataclasses import dataclass
from pyvis.network import Network
from collections import defaultdict
import networkx as nx
import argparse


@dataclass
class Instruction:
    name: str
    inputs: list
    outputs: list
    hit_count: int
    num_cycles: int
    full_str: str
    index: int


def create_graph(instrs, ignore_scalar_reg=True):
    selected_insts = ["mfma", "load", "store"]
    graph = defaultdict(lambda: {"forward": set(), "backward": set()})
    # Add edges based on dependencies
    for i, inst_a in enumerate(instrs):
        overridden = set()
        for inst_b in instrs[i + 1 :]:
            # Check if any of inst_a's outputs are in inst_b's inputs
            added = False
            for out in inst_a.outputs:
                # We need to consider if output is overridden by something that comes later, then no dependency
                if out in overridden:
                    continue
                for inp in inst_b.inputs:
                    if (
                        (not ignore_scalar_reg or out[0] in ["v", "a"])
                        and out == inp
                        and not added
                    ):
                        added = True
                        if (
                            True
                            or any(filt in inst_a.full_str for filt in selected_insts)
                            or any(filt in inst_b.full_str for filt in selected_insts)
                        ):
                            graph[inst_a.name]["forward"].add(inst_b.name)
                            graph[inst_b.name]["backward"].add(inst_a.name)
                            break
            for out in inst_b.outputs:
                overridden.add(out)
    return graph


def run_DFS(graph, starting_nodes, search_dir="backward"):
    new_graph = defaultdict(lambda: {"forward": set(), "backward": set()})
    other_dir = "backward" if search_dir == "forward" else "forward"
    stack = starting_nodes[:]
    while stack:
        top = stack.pop()
        for nbr in graph[top][search_dir]:
            new_graph[top][search_dir].add(nbr)
            new_graph[nbr][other_dir].add(top)
            stack.append(nbr)
    return new_graph


def visualize_graph(my_graph, file_name="node_dependencies.html"):
    nx_graph = nx.DiGraph()
    for node in my_graph.keys():
        nx_graph.add_node(node)
        for e in my_graph[node]["forward"]:
            nx_graph.add_edge(node, e)
    net = Network(directed=True, height="1200px")
    net.from_nx(nx_graph)
    # net.show_buttons(filter_=True)
    # Set hierarchical layout options
    net.set_options(
        """
    {
      "configure": {
            "enabled": false
      },
      "nodes": {
        "font": {
          "size": 12,
          "strokeWidth": 9
        },
         "color": {
            "highlight": "#FF0000"
        }
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "vertical",
          "roundness": 0.4
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 200,
          "nodeSpacing": 200,
          "treeSpacing": 200,

          "direction": "UD",
          "sortMethod": "directed"
        }
      }
    }
    """
    )

    # Save the network visualization to an HTML file
    net.write_html(file_name)
    # Optionally, open the file immediately:
    # net.show(file_name)


def expand_registers(str):
    reg_type = str[0]
    if reg_type not in ["v", "s", "a"]:
        return []
    if ":" in str:
        range_part = str[2:-1].split(":")
        # Handle range
        res = [
            f"{reg_type}{num}"
            for num in range(int(range_part[0]), int(range_part[1]) + 1)
        ]
    else:
        # Handle single register
        res = [str.strip()]

    return res


def process_instruction(instruction, is_store=False):
    # Remove commas and split the instruction into parts
    parts = instruction.replace(",", "").split()

    # Extract input registers
    inputs = []
    reg_cnt = 0
    # 1-> name
    for i, p in enumerate(parts[1:]):
        if p[0] in ["v", "s", "a"]:
            reg_cnt += 1
            # 0 -> output
            if i > 0:
                inputs.append(p)
    if reg_cnt == 0:
        return [], []

    # Extract output registers
    outputs = [
        parts[1],
    ]
    # if store, they are all inputs
    if is_store:
        inputs.extend(outputs)
        outputs = []

    outputs = [expand_registers(el) for el in outputs]
    inputs = [expand_registers(el) for el in inputs]

    # flattent the list
    inputs = sum(inputs, [])
    outputs = sum(outputs, [])

    return inputs, outputs


def read_json_code(path, start_line, end_line):
    with open(path) as fptr:
        data = json.load(fptr)

    instructions = []

    data = data["code"]
    started = False
    for i, line in enumerate(data):
        if start_line in line:
            started = True
            continue
        if started and end_line in line:
            break
        if started and len(line[0]) > 0 and line[0] != ".":
            total_cycles = int(line[-1])
            hit_count = int(line[-2])
            line = line[0]
            if "load" in line or "store" in line or "read" in line or "write" in line:
                name = line.split()[0]
                name = f"{str(i)}-{name}"
                full_str = f"{str(i)}-{line}"
                is_store = "store" in line or "write" in line
                inputs, outputs = process_instruction(line, is_store)
                inst = Instruction(
                    name, inputs, outputs, hit_count, total_cycles, full_str, i
                )
                instructions.append(inst)
            elif len(line) > 1 and line[0:2] in ["s_", "v_"]:
                name = line.split()[0]
                name = f"{str(i)}-{name}"
                full_str = f"{str(i)}-{line}"
                is_store = False
                inputs, outputs = process_instruction(line, is_store)
                inst = Instruction(
                    name, inputs, outputs, hit_count, total_cycles, full_str, i
                )
                instructions.append(inst)
            else:
                print("Skipped line:", line)

    return instructions


def read_amdgcn_code(path, start_line, end_line):
    instructions = []
    hit_count = 0
    total_cycles = 0
    with open(path) as fptr:
        i = 0
        started = False
        for line in fptr:
            i += 1
            line = line.strip()
            if start_line in line:
                started = True
                continue
            if started and end_line in line:
                break
            if started and len(line) > 0 and line[0] != ".":

                if (
                    "load" in line
                    or "store" in line
                    or "read" in line
                    or "write" in line
                ):
                    name = line.split()[0]
                    name = f"{str(i)}-{name}"
                    full_str = f"{str(i)}-{line}"
                    is_store = "store" in line or "write" in line
                    inputs, outputs = process_instruction(line, is_store)
                    inst = Instruction(
                        name, inputs, outputs, hit_count, total_cycles, full_str, i
                    )
                    instructions.append(inst)
                elif len(line) > 1 and line[0:2] in ["s_", "v_"]:
                    name = line.split()[0]
                    name = f"{str(i)}-{name}"
                    full_str = f"{str(i)}-{line}"
                    is_store = False
                    inputs, outputs = process_instruction(line, is_store)
                    inst = Instruction(
                        name, inputs, outputs, hit_count, total_cycles, full_str, i
                    )
                    instructions.append(inst)
                else:
                    print("Skipped line:", line)

    return instructions


parser = argparse.ArgumentParser(description="AMD-GCN graph generation")
parser.add_argument(
    "--gcn_file",
    type=str,
    default="amdgcn.out",
    help="Path for the file containing the ASM",
)
parser.add_argument(
    "--out_file", type=str, default="graph.html", help="Path for the output HTML"
)
parser.add_argument(
    "--start_line",
    type=str,
    default="s_cbranch_scc1 .LBB0_2",
    help="Analyzes starting from this line (exclusive)",
)
parser.add_argument(
    "--end_line",
    type=str,
    default="s_endpgm",
    help="Analyzes ends at this line (exclusive)",
)
parser.add_argument(
    "--ignore_scalars",
    type=int,
    default=1,
    help="Flag to indicate whether to ignore instructions with scalar outputs",
)

args = parser.parse_args()
print(args)

if args.gcn_file.endswith("json"):
    instructions = read_json_code(
        args.gcn_file, start_line=args.start_line, end_line=args.end_line
    )
else:
    instructions = read_amdgcn_code(
        args.gcn_file, start_line=args.start_line, end_line=args.end_line
    )
buffer_store_insts = [inst for inst in instructions if "buffer_store" in inst.name]

graph = create_graph(instructions, ignore_scalar_reg=args.ignore_scalars)
sub_graph = run_DFS(
    graph,
    [
        buffer_store_insts[0].name,
    ],
)
visualize_graph(sub_graph, args.out_file)