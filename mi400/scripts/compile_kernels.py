import os
from triton.tools.env import getTritonBasePath

kernels = [
    # {
    #     "path": getTritonBasePath() + "/microbenchmarks/softmax/micro_softmax.py",
    #     "signature": "*fp32:16,*fp32:16,i32:16,i32:16,i32:16,2048",
    #     "name":"softmax_kernel",
    #     "options": "--num-warps=2"
    # },
    {
        "path": os.path.join(getTritonBasePath(), "microbenchmarks/hgemm/micro_hgemm.py"), "signature":
        "*fp16:16,*fp16:16,*fp32:16,i32:16,1,i32:16,1,i32:16,1,1024, 1024, 1024, 128, 128, 128", "name": "kernel",
        "options": "--num-warps=8 --num-stages=2"
    }
]
for kernel in kernels:
    cmd = f'python3 -m triton.tools.compile_mi400_sim {kernel["path"]} --signature=\"{kernel["signature"]}\" --kernel-name=\"{kernel["name"]}\" {kernel["options"]}'
    print(cmd)
    os.system(cmd)
