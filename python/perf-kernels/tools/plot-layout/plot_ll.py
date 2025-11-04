#!/usr/bin/env python3
import argparse
import ast
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as cm
import matplotlib.colors as mcolors
from io import StringIO
from collections import defaultdict

def get_color(i):
    """
    Return an RGBA color for index i in [0, 63].
    - 4 hue families (Blues, Oranges, Greens, Reds)
    - Within each hue family (16 values), light → dark
    """
    if not (0 <= i < 64):
        raise ValueError("threadsPerWarp must be in range [0, 63]")

    # Define 4 base colormaps (different hues)
    hue_maps = ['Blues', 'Oranges', 'Greens', 'Reds']

    hue_index = i // 16      # which hue group (0–3)
    shade_index = i % 16     # how dark/light within the group

    # Sample linearly from light (0.3) to dark (0.9)
    levels = np.linspace(0.3, 0.9, 16)
    cmap = cm.colormaps.get_cmap(hue_maps[hue_index])
    return cmap(levels[shade_index])

class LinearLayout:
    def __init__(self, register_bases=None, lane_bases=None, warp_bases=None):
        # Default to empty lists if not provided
        self.register_bases = register_bases or []
        self.lane_bases = lane_bases or []
        self.warp_bases = warp_bases or []

    def apply(self, register_id: int = 0, lane_id: int = 0, warp_id: int = 0) -> List[int]:
        """Compute the tensor coordinates from given hardware indices."""
        def apply_domain(bases, idx):
            if not bases:
                return np.zeros(self.output_dim(), dtype=int)
            result = np.zeros(len(bases[0]), dtype=int)
            for i, base in enumerate(bases):
                if (idx >> i) & 1:
                    result ^= np.array(base)
            return result

        reg_part = apply_domain(self.register_bases, register_id)
        lane_part = apply_domain(self.lane_bases, lane_id)
        warp_part = apply_domain(self.warp_bases, warp_id)
        return list(reg_part ^ lane_part ^ warp_part)

    def output_dim(self):
        """Infer the output tensor dimensionality."""
        for domain in [self.register_bases, self.lane_bases, self.warp_bases]:
            if domain:
                return len(domain[0])
        return 0

    def print_layout(self):
        """Pretty print the layout."""
        def format_bases(name: str, bases: List[List[int]]):
            if not bases:
                print(f"- {name}: <none>")
                return
            print(f"- {name}:")
            for i, base in enumerate(bases):
                bit_val = 1 << i  # powers of two
                base_str = ", ".join(str(x) for x in base)
                print(f"  {name}={bit_val} --> ({base_str})")

        format_bases("register", self.register_bases)
        format_bases("lane", self.lane_bases)
        format_bases("warp", self.warp_bases)

    def get_layout_str(self):
        """Return the pretty-printed layout as a string."""
        buf = StringIO()

        def format_bases(name: str, bases: List[List[int]]):
            if not bases:
                buf.write(f"- {name}: <none>\n")
                return
            buf.write(f"- {name}:\n")
            for i, base in enumerate(bases):
                bit_val = 1 << i  # powers of two
                base_str = ", ".join(str(x) for x in base)
                buf.write(f"  {name}={bit_val} --> ({base_str})\n")

        format_bases("register", self.register_bases)
        format_bases("lane", self.lane_bases)
        format_bases("warp", self.warp_bases)

        return buf.getvalue()

    def tensorSize(self) -> List[int]:
        """Compute tensor shape as double the largest coordinate in each dimension."""
        all_bases = self.register_bases + self.lane_bases + self.warp_bases
        if not all_bases:
            return []

        dim_count = len(all_bases[0])
        max_per_dim = [0] * dim_count

        for base in all_bases:
            for d in range(dim_count):
                max_per_dim[d] = max(max_per_dim[d], base[d])

        return [2 * x for x in max_per_dim]

    def vectorSize(self):
        """
        Determine the vectorized dimension and vector size along the register domain.

        Algorithm:
          - For each output dimension `d`:
            * Walk register bases in bit order: base[0], base[1], base[2], ...
            * Start from value = base[0][d]. If it's zero, no vector found on that dim.
            * For j = 1..n-1: if base[j][d] == 2 * value: value = base[j][d] and continue.
              otherwise stop the run.
            * After the run, vector_size = 2 * value (even if the run length is 1).
            * Return (d, vector_size) for the first dimension where value != 0.
          - If no dimension yields a non-zero starting value, return (None, 1).
        """
        if not self.register_bases:
            return None, 1

        num_dims = len(self.register_bases[0])
        num_bases = len(self.register_bases)

        for dim in range(num_dims):
            # starting value is the bit0 base for this dim
            start_val = self.register_bases[0][dim]
            if start_val == 0:
                # no vector run starting at bit0 for this dimension
                continue

            val = start_val
            # walk subsequent bases in bit order, stopping when doubling pattern breaks
            for j in range(1, num_bases):
                curr = self.register_bases[j][dim]
                if curr == 2 * val:
                    val = curr
                    continue
                else:
                    break

            vector_dim = dim
            vector_size = 2 * val
            return vector_dim, vector_size

        return None, 1

def drawVec(dim0, dim1, vecDim, vecSize, shape, lanes, ax, cmap, fontSize):
    x = shape[0] - dim0 - 1
    if vecDim == 0:
        x += 1
    y = dim1
    width = vecSize if vecDim == 1 else 1
    height = 1 if vecDim == 1 else -vecSize

    # Pick color based on first lane
    tid = lanes[0]
    rect = Rectangle((y, x), width, height,
                     facecolor=get_color(tid), edgecolor='black', lw=0.3)
    ax.add_patch(rect)

    # Combine all lane IDs into one string
    lane_text = ", ".join(f"t{lane}" for lane in lanes)
    ax.text(y+0.5*width, x+0.5*height,
            lane_text,
            ha='center', va='center', fontsize=fontSize, color='black')


def plot(layout, warpId, out_file):
    shape = layout.tensorSize()
    fig, ax = plt.subplots(figsize=(shape[0], shape[1]))
    cmap = cm.colormaps.get_cmap("Set1")
    vecDim , vecSize = layout.vectorSize()
    fontSize = 40
    ratio = max(shape[0]/shape[1], shape[1]/shape[0])
    fontSize /= ratio

    regSize = 2** len(layout.register_bases)
    laneSize = 2** len(layout.lane_bases)

    coord_to_lanes = defaultdict(list)

    for lane in range(laneSize):
        for reg in range(0, regSize, vecSize):
            dim0, dim1 = layout.apply(reg, lane, warpId)
            coord_to_lanes[(dim0, dim1)].append(lane)

    for (dim0, dim1), lanes in coord_to_lanes.items():
        drawVec(dim0, dim1, vecDim, vecSize, shape, lanes, ax, cmap, fontSize)

    ax.text(-0.5, 0.5*shape[0],
            f"{shape[0]}",
            ha='center', va='center', fontsize=fontSize, color='black')
    ax.text(0.5*shape[1], shape[0]+.5,
            f"{shape[1]}",
            ha='center', va='center', fontsize=fontSize, color='black')

    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout()
    if out_file is None:
        out_file = "linear_layout_plot.pdf"
    else:
        out_file += ".pdf"
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Layout visualization saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Parse and print a Triton linear layout.")
    parser.add_argument("--regBase", help='Register bases, e.g. "[[0,1],[0,2],[0,4],[0,16],[0,32],[0,64]]"')
    parser.add_argument("--laneBase", help='Lane bases, e.g. "[[1,0],[2,0],[4,0],[8,0],[16,0],[0,8]]"')
    parser.add_argument("--warpBase", help='Warp bases, e.g. "[[32,0],[64,0],[128,0]]"')
    parser.add_argument("--warpId", type=int, default=0)
    parser.add_argument("--o")

    args = parser.parse_args()

    def parse_bases(s):
        return ast.literal_eval(s) if s else []

    register_bases = parse_bases(args.regBase)
    lane_bases = parse_bases(args.laneBase)
    warp_bases = parse_bases(args.warpBase)
    warpId = args.warpId
    out_file = args.o

    layout = LinearLayout(register_bases, lane_bases, warp_bases)

    layout.print_layout()
    size = layout.tensorSize()
    if size:
        print("\nTensor shape:", size)
    else:
        print("\n(no bases provided — tensor shape undefined)")

    vector_dim, vector_size = layout.vectorSize()
    if vector_dim is not None:
        print(f"Vector dimension: {vector_dim}, vector size: {vector_size}")
    else:
        print("No vectorized dimension found.")


    warpSize = 2** len(layout.warp_bases)
    if warpId >= warpSize:
        print(f"warpId must be < {warpSize}, but got {warpId}")
        exit(0)

    plot(layout, warpId, out_file)


if __name__ == "__main__":
    main()
