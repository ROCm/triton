import argparse
import sys
import os

from blocked import generate_blocked_tex
from dot import generate_dot_tex
from lds import generate_lds_tex
from wmma import generate_wmma_tex
from utils import run_bash_command


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Draw triton layouts",
        allow_abbrev=True,
    )
    parser.add_argument("--output", type=str, default="myplot", help='output pdf file name (without surfix)')
    parser.add_argument("--keep", action='store_true', default=False, help='If set, keep the generated .tex file')
    subparsers = parser.add_subparsers(
        dest="plot_type",
        metavar="PLOT_TYPE",
        required=True,
        title="subcommands",
        description="Choose to plot blocked, lds, dot or wmma",
        help="Choose one of the four plot mode"
    )
    ## blocked layout parameters
    blocked_parser = subparsers.add_parser("blocked", help="plot blocked layout for global memory access")
        ## tensor shapes
    blocked_parser.add_argument("--tensorShape", type=int, nargs=2, default=(128, 64),
                        help='2D block shape in the form of (dim0, dim1)')
    blocked_parser.add_argument("--rowName", type=str, default="M", help='tensor dim0 name')
    blocked_parser.add_argument("--colName", type=str, default="K", help='tensor dim1 name')
    blocked_parser.add_argument("--sizePerThread", type=int, nargs=2, default=(1, 4))
    blocked_parser.add_argument("--threadsPerWarp", type=int, nargs=2, default=(16, 4))
    blocked_parser.add_argument("--warpsPerCTA", type=int, nargs=2, default=(1, 4))
    blocked_parser.add_argument("--order", type=int, nargs=2, default=(1, 0))
    blocked_parser.add_argument("--blockShape", type=int, nargs=2, default=(128, 64),
                        help='2D block size (dim0, dim1) to override the inferred size from blocked information')
    ## dot layout parameters
    dot_parser = subparsers.add_parser("dot", help="plot dot layout for MFMA")
    dot_parser.add_argument("--warpsPerCTA", type=int, nargs=2, default=(1, 4))
    dot_parser.add_argument("--dotShape", type=int, nargs=3, default=(32, 128, 64), help='Dot op shape in the form of M,N,K')
    dot_parser.add_argument("--nonKDim", type=int, default=16, choices=[16, 32], help='mfma instruction dim')
    dot_parser.add_argument("--kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements per thread')
    dot_parser.add_argument("--kGroup", type=int, default=1, choices=[1, 2],
                        help='total number of elements / kWidth per mfma instruction')
    dot_parser.add_argument("--dtype-a", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand A')
    dot_parser.add_argument("--dtype-b", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand B')
    dot_parser.add_argument("--mfmaTrans", action='store_true', default=False, help='If set, then use mfma.trans layout')
    dot_parser.add_argument("--scale", action='store_true', default=False,
                        help='If set, plot the scale tensor for mfma_f8f6f4 instructions')
    ## LDS access parameters
    lds_parser = subparsers.add_parser("lds", help="plot LDS (shared memory) layout")
    lds_parser.add_argument("--tensorShape", type=int, nargs=2, default=(128, 64),
                        help='2D block shape in the form of (dim0, dim1)')
    lds_parser.add_argument("--kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements per thread')
    lds_parser.add_argument("--dtype", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of tensor to be stored in LDS')
    lds_parser.add_argument("--nonKDim", type=int, default=16, choices=[16, 32], help='mfma instruction dim')
    lds_parser.add_argument("--banks", type=int, default=32, choices=[32, 64], help='choose the number of banks in LDS')
    lds_parser.add_argument("--lds-layout", type=str, default="none", choices=['swizzle', 'padding', 'none'],
                        help='choose the LDS data layout')
    lds_parser.add_argument("--lds-access", type=str, default="none", choices=['read', 'write', 'none'],
                        help='choose LDS access mode')
    lds_parser.add_argument("--mnContig", action='store_true', default=False,
                        help='If set, the tensor is K x N and n-contig')
    lds_parser.add_argument("--mfma-trans-load", action='store_true', default=False,
                        help='If set, use MFMA transpose load instructions')
    lds_parser.add_argument("--swizzleVec", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements in a vector to swizzle')
    lds_parser.add_argument("--padInterval", type=int, default=1, help='Add padding for every padInterval bytes')
    lds_parser.add_argument("--padAmount", type=int, default=0, help='Pad padAmount bytes for every padInterval bytes')
    ## wmma instruction layout parameter
    wmma_parser = subparsers.add_parser("wmma", help="plot dot layout for wmma")
    wmma_parser.add_argument("--wave-size", type=int, default=32, choices=[32, 64], help='choose the wmma instruction mode')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ofilename = args.output
    keepSrc = args.keep

    match args.plot_type:
        case "blocked":
            generate_blocked_tex(args)
        case "dot":
            generate_dot_tex(args)
        case "lds":
            generate_lds_tex(args)
        case "wmma":
            generate_wmma_tex(args)
        case _:
            raise NotImplementedError(f"Only blocked, dot, lds and wmma supported, you entered {args.plot_type}")

    run_bash_command(f"pdflatex -jobname {ofilename} myplot.tex")
    print(f"plot saved in {ofilename}.pdf")

    # Remove auxiliary files
    os.remove(f"{ofilename}.aux")
    os.remove(f"{ofilename}.log")
    if not keepSrc:
        os.remove("myplot.tex")
        run_bash_command("rm -rf ./auto")


if __name__ == '__main__':
    sys.exit(main())
