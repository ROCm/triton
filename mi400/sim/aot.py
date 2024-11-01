import hashlib
import importlib.util
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from .llvm2sp3 import convertLLVMAsmFileIntoSp3
from .sim_arguments import ShaderInfo

import triton
import triton.backends
from triton.backends.compiler import GPUTarget

desc = """
Triton ahead-of-time compiler for mi400. THis is a copy of the aot compiler for Triton
but it will  produce a file that can be run on the mi400 simulator
"""


def aot_compile(args):
    # execute python sources and extract functions wrapped in JITFunction
    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else os.getcwd()

    arg_path = Path(args.path)
    out_path = Path(out_path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = 'constexpr'
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    # attrs = triton.backends.compiler.AttrsDescriptor.from_hints(hints)
    attrs = {k: [("tt.divisibility", 16)] for k, v in hints.items() if v == 16}
    # for p, v in attrs.get_constants().items():
    #     constants.update({kernel.arg_names[p]: v})
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    warp_size = 32
    mi400Target = GPUTarget("hip", args.arch, warp_size)
    opts = {
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "warp_size": warp_size,
        "allow_flush_denorm": args.flush_denorm,
        "num_ctas": args.num_cta,
        "global_prefetch": args.global_prefetch,
    }
    ccinfo = triton.compile(src, target=mi400Target, options=opts)

    # Save assembly
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    out_asm_path = os.path.join(out_path, out_name + ".s")
    with open(out_asm_path, 'w') as f:
        print(ccinfo.asm["amdgcn"], file=f)
    out_bin_path = os.path.join(out_path, out_name + ".hsaco")
    with open(out_bin_path, 'wb') as f:
        f.write(ccinfo.asm["hsaco"])
    out_triton_path = os.path.join(out_path, out_name + ".ttgir")
    with open(out_triton_path, 'w') as f:
        print(ccinfo.asm["ttgir"], file=f)
    out_llvm_path = os.path.join(out_path, out_name + ".ll")
    with open(out_llvm_path, 'w') as f:
        print(ccinfo.asm["llir"], file=f)

    if args.enable_sp3:
        shader_info = convertLLVMAsmFileIntoSp3(args.arch, out_path, out_asm_path, args.kernel_name)
    else:
        shader_info = ShaderInfo()
    shader_info.lds_bytes = ccinfo.metadata.shared
    shader_info.cluster_dim = ccinfo.metadata.cluster_dims
    return shader_info


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--num-warps", "-w", type=int, default=2, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=2,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument('--flush-denorm', action='store_true', help='Allow flushing denorms')
    args = parser.parse_args()

    out_path = aot_compile(args)
