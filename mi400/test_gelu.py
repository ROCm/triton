from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments
from triton.tools.env import getTritonBasePath
import os
import torch
from scipy.constants import pi

N = 4096
M = 1024
BLOCK_SIZE = 16384
pi = pi

args = Arguments()
args.kernel_name = "gelu_kernel"
args.path = os.path.join(getTritonBasePath(), "mi400/kernels/gelu_kernel.py")
args.signature = f"*fp32,*fp32,{pi},{BLOCK_SIZE}"
args.out_path = os.path.join(getTritonBasePath(), "gelu")
args.num_warps = 32
args.num_stages = 2
args.flush_denorm = 1
shaderInfo = aot_compile(args)

# For reproducibility and debuggability
torch.manual_seed(42)
torch.set_printoptions(edgeitems=30, linewidth=100000)

sim = MI400Simulator(args.out_path)
# input = torch.randint(1, 2, (M, N)).to(torch.float32)
input = torch.randn(M, N, dtype=torch.float32)
output = torch.nn.GELU(approximate='tanh')(input)
# output = naive_softmax(input)
# print(output)
# print(input.stride(0))
# print(output.stride(0))

addressInput = sim.createInputSurface(input)
addressOutput = sim.createOutputSurface(output)

numBlocks = 256
grid = [numBlocks, 1, 1]
sim.createArgs([addressOutput, addressInput, N, N, M, N], grid)
sim.launch(args.num_warps, grid, shaderInfo)
sim.done()
