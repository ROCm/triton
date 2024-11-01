import os
import numpy
import torch
from .sim_arguments import ShaderInfo


def get_higher_32_bits(value):
    return (value >> 32) & 0xFFFFFFFF


def get_lower_32_bits(value):
    return value & 0xFFFFFFFF


def tofile(tensor, fname):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    if tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2:
        tensor = tensor.view(torch.uint8)
    tensor.numpy().tofile(fname)


class MI400Simulator:

    def __init__(self, runDir):
        # The base address is 0x7f8769080000
        self.runDir = runDir
        self.curAddress = numpy.ulonglong(140219559444480)
        self.numInputSurfaces = 0
        self.numOutputSurfaces = 0
        self.argsAddress = None
        self.lines = []

    def _addkeyval(self, key, value):
        self.lines.append("{}={}\n".format(key, value))

    def createOutputSurface(self, yRef):
        idx = self.numOutputSurfaces
        Obuf = torch.zeros(yRef.numel(), dtype=yRef.dtype)
        Obuf_filename = os.path.join(self.runDir, "Obuf.bin")
        tofile(Obuf, Obuf_filename)
        self._addkeyval(f"-GPU_MEM_SURF_BIN_PATH_{idx}", Obuf_filename)
        self._addkeyval(f"-GPU_MEM_SURF_BASE_{idx}", hex(self.curAddress))
        self._addkeyval(f"-GPU_MEM_SURF_SIZE_BYTES_{idx}", hex(Obuf.untyped_storage().size()))
        if yRef.dtype == torch.float16 or yRef.dtype == torch.bfloat16:
            self._addkeyval(f"-GPU_MEM_DATA_TYPE_{idx}", "\"int16\"")  # TODO: float?
        elif yRef.dtype == torch.float32:
            self._addkeyval(f"-GPU_MEM_DATA_TYPE_{idx}", "\"float\"")  # TODO: float?
        else:
            assert ("Invalid data type")
        self._addkeyval(f"-GPU_MEM_FILE_TYPE_{idx}", "\"OUT\"")
        outputAddress = self.curAddress
        self.curAddress += numpy.ulonglong(Obuf.numel() * Obuf.itemsize)
        refFilename = os.path.join(self.runDir, "RefData.bin")

        tofile(yRef, refFilename)
        self._addkeyval(f"-GOLD_MEM_SURF_BIN_PATH_{idx}", refFilename)
        self._addkeyval(f"-GOLD_MEM_SURF_SIZE_BYTES_{idx}", hex(yRef.untyped_storage().size()))
        self.numOutputSurfaces += 1
        return outputAddress

    def createInputSurface(self, tensor):
        idx = self.numInputSurfaces
        Ibuf = tensor.ravel()
        IbufFilename = os.path.join(self.runDir, f"Ibuf{idx}.bin")
        tofile(Ibuf, IbufFilename)
        self._addkeyval(f"-MEM_SURF_BIN_PATH_{idx}", IbufFilename)
        self._addkeyval(f"-MEM_SURF_BASE_{idx}", hex(self.curAddress))
        self._addkeyval(f"-MEM_SURF_SIZE_BYTES_{idx}", hex(Ibuf.untyped_storage().size()))
        inputAddress = self.curAddress
        self.curAddress += numpy.ulonglong(Ibuf.numel() * Ibuf.itemsize)
        self.numInputSurfaces += 1
        return inputAddress

    def createArgs(self, kernel_params, grid_dim):
        # Base addresses. Note that the addresses are 64 bits.  Please note that
        # this represents the arguments of the kernel.
        #
        # kernel(void * input, void *output, ...)
        # addressesFilename = os.path.join(self.runDir, "addresses.bin")
        # numpy.array(addresses).tofile(addressesFilename)

        # Scalar parameters. These are the parameters used by the algorithm and
        # passed as 32 bit values. Note that x_stride and y_stride are the same for
        # softmax
        #
        # kernel(..., x_stride, y_stride, cols, BLOCK_SIZE)
        idx = self.numInputSurfaces
        scalarsFileName = os.path.join(self.runDir, "scalars.bin")
        # totLen = len(params) * 4 + len(addresses) * 8
        totLen = 0
        with open(scalarsFileName, 'wb') as f:
            # for address in addresses:
            #     f.write(numpy.int64(address).tobytes())
            for param in kernel_params:
                if isinstance(param, numpy.int32) or isinstance(param, int):
                    f.write(numpy.int32(param).tobytes())
                    totLen += 4
                elif isinstance(param, numpy.ulonglong):
                    f.write(param.tobytes())
                    totLen += 8
                elif isinstance(param, numpy.float32) or isinstance(param, float):
                    f.write(numpy.float32(param).tobytes())
                    totLen += 4
                else:
                    assert ("invalid parameter type")

            def getPadding(val, align):
                return (align - (val % align)) % align

            def align(totLen, f, align):
                padding_bytes = getPadding(totLen, align)
                written = 0
                assert padding_bytes % 4 == 0, "Kernelargs must be a multiple of 4 bytes"
                for i in range(0, padding_bytes // 4):
                    f.write(numpy.uint32(0xDEADBEEF).tobytes())
                    written += 4
                return written

            # There is an ptr argument added after the explicit ones
            totLen += align(totLen, f, 8)  # Ptrs have alignment 8
            f.write(numpy.uint64(0xDEADBEEF))
            totLen += 8

            # Followed by the explicit args are the hidden ones, they start with the grid_dim_x/y/z
            f.write(numpy.uint32(grid_dim[0]))
            f.write(numpy.uint32(grid_dim[1]))
            f.write(numpy.uint32(grid_dim[2]))
            totLen += 3 * 4

        self._addkeyval(f"-MEM_SURF_BIN_PATH_{idx}", scalarsFileName)
        self._addkeyval(f"-MEM_SURF_BASE_{idx}", hex(self.curAddress))
        self._addkeyval(f"-MEM_SURF_SIZE_BYTES_{idx}", hex(totLen))

        self.argsAddress = int(self.curAddress)
        self.curAddress += numpy.ulonglong(totLen)
        self.numInputSurfaces += 1

    def done(self):
        self._addkeyval("-NUM_MEM_SURFACE", self.numInputSurfaces)
        self._addkeyval("-GOLD_NUM_MEM_SURFACE", self.numOutputSurfaces)
        iniFileName = os.path.join(self.runDir, "memory_surface.ini")
        with open(iniFileName, "w") as f:
            f.writelines(self.lines)
        return iniFileName

    def launch(self, num_warps, num_cta, grid, shaderInfo: ShaderInfo):
        workgroup_size_x, workgroup_size_y, workgroup_size_z = grid
        localWorkSize = 32 * num_warps
        iniFilename = os.path.join(self.runDir, "reg_seq.ini")
        with open(iniFilename, "w") as f:

            def param(key, value):
                f.write("{}={}\n".format(key, hex(value)))

            param("-iCompute_START_X", 0x0)
            param("-iCompute_START_Y", 0x0)
            param("-iCompute_START_X", 0x0)
            param("-iCOMPUTE_NUM_THREAD_X", localWorkSize)
            param("-iCOMPUTE_NUM_THREAD_Y", 0x1)
            param("-iCOMPUTE_NUM_THREAD_Z", 0x1)
            param("-iCOMPUTE_PGM_LO", 0x85500010)  #0xfe4a415b
            param("-iCOMPUTE_PGM_HI", 0x7f)
            param("-iCOMPUTE_DISPATCH_PKT_ADDR_LO", 0x4a2a3280)
            param("-iCOMPUTE_DISPATCH_PKT_ADDR_HI", 0xfe)
            param("-iCOMPUTE_DISPATCH_SCRATCH_BASE_LO", 0x0)
            param("-iCOMPUTE_DISPATCH_SCRATCH_BASE_HI", 0x0)

            #RSRC1 fields
            # 5:0 VGPR count = 256
            # 9:6 SGPR count = 128
            # 11:10 priority = 0
            # 13:12 roundmodes =
            # 15:14 denorm modes for fp16
            # WGP=1
            # fp16_overflow
            # enable IEE mode
            # 128KB = lds size
            #TODO:need a way to pass this information 256 or 512 vgprs
            #param("-iCOMPUTE_PGM_RSRC1",0x248f57ff) #512 vgprs
            #param("-iCOMPUTE_PGM_RSRC1",0x248f57df) #256 vgprs

            # set resource registers
            # Used to be:
            # param("-iCOMPUTE_PGM_RSRC1", 0x248f57c8)  #64
            param("-iCOMPUTE_PGM_RSRC1", 0x248f57c0 + int((shaderInfo.num_vgprs - 1) / 16))
            # Used to be:
            # param("-iCOMPUTE_PGM_RSRC2", 0x5007a0)
            print(shaderInfo.lds_bytes)
            param("-iCOMPUTE_PGM_RSRC2", 0x0007a0 | ((1 + int((shaderInfo.lds_bytes - 1) / 1024)) << 15))
            param("-iCOMPUTE_PGM_RSRC3", 0x0)
            param("-iCOMPUTE_RESOURCE_LIMITS", 0x0)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE0", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE1", 0xffffffff)
            param("-iCOMPUTE_TMPRING_SIZE", 0x0)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE2", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE3", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE4", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE5", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE6", 0xffffffff)
            param("-iCOMPUTE_STATIC_THREAD_MGMT_SE7", 0xffffffff)
            param("-iCOMPUTE_RESTART_X", 0x0)
            param("-iCOMPUTE_RESTART_Y", 0x0)
            param("-iCOMPUTE_RESTART_Z", 0x0)
            # Split the kernel base address in two dword(32bits) parts
            param("-iCOMPUTE_USER_DATA_0", get_lower_32_bits(self.argsAddress))
            param("-iCOMPUTE_USER_DATA_1", get_higher_32_bits(self.argsAddress))
            param("-iCOMPUTE_USER_DATA_2", 0x0)
            param("-iCOMPUTE_USER_DATA_3", 0x0)
            param("-iCOMPUTE_USER_DATA_4", 0x0)
            param("-iCOMPUTE_USER_DATA_5", 0x0)
            param("-iCOMPUTE_USER_DATA_6", 0x0)
            param("-iCOMPUTE_USER_DATA_7", 0x0)
            param("-iCOMPUTE_USER_DATA_8", 0x0)
            param("-iCOMPUTE_USER_DATA_9", 0x0)
            param("-iCOMPUTE_USER_DATA_10", 0x0)
            param("-iCOMPUTE_USER_DATA_11", 0x0)
            param("-iCOMPUTE_USER_DATA_12", 0x0)
            param("-iCOMPUTE_USER_DATA_13", 0x0)
            param("-iCOMPUTE_USER_DATA_14", 0x0)
            param("-iCOMPUTE_USER_DATA_15", 0x0)
            # param("-iCOMPUTE_DISPATCH_INITIATOR", 0xc01)
            enable_clusters = any(c != 1 for c in shaderInfo.cluster_dim)
            enable_interleave_2d = enable_clusters
            param("-iCOMPUTE_DISPATCH_INITIATOR", 0xc01 | (enable_clusters << 20) | (enable_interleave_2d << 18))
            if any(c > 16 for c in shaderInfo.cluster_dim):
                raise Exception("cluster_dim_XYZ must not be larger than 31")
            if shaderInfo.cluster_dim[0] * shaderInfo.cluster_dim[1] * shaderInfo.cluster_dim[2] > 16:
                raise Exception("Cannot have more than 16 cluster")
            param("-iCOMPUTE_DISPATCH_INTERLEAVE",
                  shaderInfo.cluster_dim[0] << 16 | shaderInfo.cluster_dim[1] << 21 | shaderInfo.cluster_dim[2] << 26)
            param("-iCOMPUTE_DIM_X", workgroup_size_x)
            param("-iCOMPUTE_DIM_Y", workgroup_size_y)
            param("-iCOMPUTE_DIM_Z", workgroup_size_z)
        return iniFilename
