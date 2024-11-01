import os
from .sim_arguments import ShaderInfo


def hack_for_vgpr_count(sp3, vgpr_count):
    f = open(sp3, "r")
    newSp3 = ""
    for l in f:
        newSp3 += l
        if "wave_size" in l:
            newSp3 += (f"vgpr_count({vgpr_count})")
    f.close()
    f = open(sp3, "w")
    print(newSp3, file=f)
    f.close()


def hack_for_sellow(sp3):
    f = open(sp3, "r")
    newSp3 = ""
    for l in f:
        m = re.search("(v_wmma.*) (v.*), (v.*), (v.*), sel_lo\((v.*)\)", l)
        if m:
            print(l)
            l = f"{m.group(1)} {m.group(2)}, {m.group(3)}, {m.group(4)}, {m.group(5)}\n"

        newSp3 += l
    f.close()
    f = open(sp3, "w")
    print(newSp3, file=f)
    f.close()


def convertLLVMAsmFileIntoSp3(arch, run_dir, llvmInputFile, kernel_name):
    llvm_bin = os.getenv("LLVM_BIN_PATH")
    ffm_bin = os.getenv("FFM_BIN_PATH")
    llvm_mc = os.path.join(llvm_bin, "llvm-mc")
    llvm_objcopy = os.path.join(llvm_bin, "llvm-objcopy")
    sp3_disasm = os.path.join(ffm_bin, "sp3disasm")
    _, tail = os.path.split(llvmInputFile)
    testname, ext = os.path.splitext(tail)
    configBase = os.path.join(run_dir, testname)
    objFile = configBase + ".o"
    binFile = configBase + ".bin"
    outFile = configBase + ".sp3"
    num_lds_bytes = 0
    num_vgprs = 0
    use_scratch = False
    with open(llvmInputFile) as fin:
        for line in fin:
            if "; NumVgprs:" in line:
                num_vgprs = int(line.split(" ")[2])
            if "; LDSByteSize: " in line:
                num_lds_bytes = int(line.split(" ")[2])
            if "; ScratchSize:" in line:
                scratch_size = int(line.split(" ")[2])
                use_scratch = (scratch_size > 0)

    os.system(f"{llvm_mc} --triple=amdgcn-amd-amdhsa -mcpu={arch} --filetype=obj -o {objFile} {llvmInputFile}")
    os.system(f"{llvm_objcopy} --dump-section .text={binFile} {objFile} ")
    os.system(f"{sp3_disasm} {binFile} {outFile}")
    hack_for_vgpr_count(outFile, num_vgprs)
    return ShaderInfo(num_vgprs=num_vgprs, lds_bytes=num_lds_bytes, sp3filename=outFile, use_scratch=use_scratch)
