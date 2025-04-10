from dataclasses import dataclass


@dataclass
class BlockedConfig:
    sizePerThread: tuple
    threadsPerWarp: tuple
    warpsPerCTA: tuple
    order: tuple

def draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, blockedConfig):
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (TL) at (0,0);
    \\def\\dimColName{{{dim0Name}}}
    \\def\\dimRowName{{{dim1Name}}}
    \\drawBlockedTensor{{{dim0}}}{{{dim1}}}{{{blockedConfig.sizePerThread[0]}}}{{{blockedConfig.sizePerThread[1]}}}{{{blockedConfig.threadsPerWarp[0]}}}{{{blockedConfig.warpsPerCTA[0]}}}{{{blockedConfig.warpsPerCTA[1]}}}{{{blockedConfig.order[0]}}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def generate_blocked_tex(args):
    assert args.plot_type == "blocked", \
        f"parsing the wrong arguments. Want blocked but have {args.plot_type}"
    # preprocess the args
    tShape = args.tensorShape
    dim0 = tShape[0]
    dim1 = tShape[1]
    dim0Name = args.rowName
    dim1Name = args.colName
    sizePerThread = args.sizePerThread
    threadsPerWarp = args.threadsPerWarp
    warpsPerCTA = args.warpsPerCTA
    order = args.order
    blockedConfig = BlockedConfig(sizePerThread, threadsPerWarp, warpsPerCTA, order)

    # checks and logging
    print(f"Plotting tensor {dim0Name}={dim0},{dim1Name}={dim1} with blocked layout:")
    print(f"{sizePerThread=}", end=" ")
    print(f"{threadsPerWarp=}", end=" ")
    print(f"{warpsPerCTA=}", end=" ")
    print(f"{order=}", end=" ")
    CTAShape = [
        sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0],
        sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1],
    ]
    print(f"CTAShape={CTAShape}")
    assert dim0 != 0 and CTAShape[0] <= dim0 and dim0 % CTAShape[0] == 0, "bad tensor dimension " + dim0Name
    assert dim1 != 0 and CTAShape[1] <= dim1 and dim1 % CTAShape[1] == 0, "bad tensor dimension " + dim1Name

    # write the tex file
    with open("myplot.tex", 'w') as f_plot:
        with open("utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_blockedLayout_str = draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, blockedConfig)
        f_plot.write("\input{blocked/blockedLayout}\n")
        f_plot.write(draw_blockedLayout_str)