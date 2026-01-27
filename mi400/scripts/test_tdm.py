from triton.tools.mi400.aot import aot_compile
from triton.tools.mi400.mi400Simulator import MI400Simulator
from triton.tools.mi400.sim_arguments import Arguments
import torch

M = 64
N = 64
# K = 128
# blockSizeM = 16
# blockSizeN = 16
# blockSizeK = 128
# groupSizeM = 1
# cStrideM = 1

args = Arguments()
args.kernel_name = "tdm_kernel"
args.path = "/local/grossini/mi400-triton/python/test/mi400_kenrels/simple_tdm.py"
args.signature = "*fp16:16,*fp16:16"
args.out_path = "/local/grossini/mi400-triton/tdm"
args.num_warps = 4
args.num_stages = 1
shaderInfo = aot_compile(args)

# For reproducibility and debuggability
torch.manual_seed(42)
torch.set_printoptions(edgeitems=30, linewidth=100000)

sim = MI400Simulator(args.out_path)
a = torch.randint(1, 3, (M, N)).to(torch.float16)
c = a
# print(c)
addressA = sim.createInputSurface(a)
addressC = sim.createOutputSurface(c)
grid = [1, 1, 1]
sim.createArgs([addressA, addressC], grid)
sim.launch(args.num_warps, grid, shaderInfo)
sim.done()
