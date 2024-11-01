import ctypes
from hip import hip, hiprtc
import numpy as np
import os
from .hip_common import hip_check

ROCM_ROOT = None

# Adapted from https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html#launching-kernels


def compile_source(source, kernel_name, arch):
    global ROCM_ROOT
    if isinstance(kernel_name, str):
        kernel_name = kernel_name.encode()
    if isinstance(source, str):
        source = source.encode()
    if ROCM_ROOT is None:
        if "ROCM_ROOT" in os.environ:
            ROCM_ROOT = os.environ["ROCM_ROOT"]
        elif os.path.isdir("/opt/rocm"):
            ROCM_ROOT = "/opt/rocm"
        else:
            raise RuntimeError("Could not determine ROCM_ROOT")
    prog = hip_check(hiprtc.hiprtcCreateProgram(source, kernel_name, 0, [], []))
    cflags = [
        b"--offload-arch=" + arch,
        b"-std=c++20",
        b"-I" + ROCM_ROOT.encode() + b"/include",
    ]
    err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        raise RuntimeError(log.decode())
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, kernel_name))
    return prog, module, kernel


def calculate_dims(props, n):
    block_size = props.warpSize
    num_threads = min(props.maxThreadsPerMultiProcessor * props.multiProcessorCount, n)
    num_blocks = (num_threads + block_size - 1) // block_size
    return hip.dim3(num_blocks), hip.dim3(block_size)


def binary_op_kernel(self, other, op_name, op_impl, host_impl):
    # avoid circular import
    from .tensor import Tensor

    if isinstance(other, Tensor):
        assert self.device == other.device
    if self.data_h is not None:
        assert self.dtype.is_native_numpy_type()
        if isinstance(other, Tensor):
            assert other.dtype.is_native_numpy_type()
            return Tensor(host_impl(self.data_h, other.data_h))
        return Tensor(host_impl(self.data_h, other))
    assert isinstance(other, Tensor)
    assert self.dtype == other.dtype
    assert self.shape == other.shape
    assert self.dtype.is_native_clang_type()
    type_str = self.dtype.hip_type_str
    kernel_name = f"notorch_{op_name}_{type_str}"
    source = f"""
        extern "C" __global__ void {kernel_name}({type_str} *in1_arr, {type_str} *in2_arr, size_t n, {type_str} *out_arr) {{
          for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {{
            {type_str} in1 = in1_arr[idx];
            {type_str} in2 = in2_arr[idx];
            out_arr[idx] = {op_impl};
          }}
        }}
    """

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    prog, module, kernel = compile_source(source, kernel_name, arch)

    n = self.numel()
    out_d = hip_check(hip.hipMalloc(self.data_d.size))
    grid_dim, block_dim = calculate_dims(props, n)

    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid_dim,
            *block_dim,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=[self.data_d, other.data_d, ctypes.c_uint64(n), out_d],
        ))
    hip_check(hip.hipStreamSynchronize(0))

    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    return Tensor(out_d, dtype=self.dtype, shape=self.shape, device=self.device)


def unary_op_kernel(self, op_name, op_impl, host_impl):
    # avoid circular import
    from .tensor import Tensor

    if self.data_h is not None:
        assert self.dtype.is_native_numpy_type()
        return Tensor(host_impl(self.data_h))
    assert self.dtype.is_native_clang_type()
    type_str = self.dtype.hip_type_str
    kernel_name = f"notorch_{op_name}_{type_str}"
    source = f"""
        extern "C" __global__ void {kernel_name}({type_str} *in1_arr, size_t n, {type_str} *out_arr) {{
          for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {{
            {type_str} in1 = in1_arr[idx];
            out_arr[idx] = {op_impl};
          }}
        }}
    """

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    prog, module, kernel = compile_source(source, kernel_name, arch)

    n = self.numel()
    out_d = hip_check(hip.hipMalloc(self.data_d.size))
    grid_dim, block_dim = calculate_dims(props, n)

    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid_dim,
            *block_dim,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=[self.data_d, ctypes.c_uint64(n), out_d],
        ))
    hip_check(hip.hipStreamSynchronize(0))

    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    return Tensor(out_d, dtype=self.dtype, shape=self.shape, device=self.device)


def conversion_kernel(self, other_type):
    # avoid circular import
    from .tensor import Tensor

    assert self.data_d is not None
    in_type_str = self.dtype.hip_type_str
    out_type_str = other_type.hip_type_str
    kernel_name = f"notorch_{in_type_str}_to_{out_type_str}"
    source = f"""
        extern "C" __global__ void {kernel_name}({in_type_str} *in1_arr, size_t n, {out_type_str} *out_arr) {{
          for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {{
            {in_type_str} in1 = in1_arr[idx];
            out_arr[idx] = static_cast<{out_type_str}>(in1);
          }}
        }}
    """
    if self.dtype.is_fp8_type() or other_type.is_fp8_type():
        # FP8 header does not work with hipRTC
        # TODO: create JIRA ticket for this
        source = """
            typedef struct __attribute__((aligned(2))) {
              unsigned short x;
            } __hip_bfloat16_raw;

            struct __attribute__((aligned(2))) __hip_bfloat16 {
              unsigned short __x;

              __device__ operator float() const { return (float)(*(__bf16*)&__x); }
              __device__ operator __hip_bfloat16_raw() const { return __hip_bfloat16_raw{__x}; }
              __device__ __hip_bfloat16(const __hip_bfloat16_raw& val) : __x(val.x) {}
              __device__ __hip_bfloat16(const float val) { __bf16 temp = (__bf16)val; __x = *(unsigned short*)&temp; }
              __device__ __hip_bfloat16() = default;
            };

            typedef struct __attribute__((aligned(4))) {
              unsigned short x;
              unsigned short y;
            } __hip_bfloat162_raw;

            struct __attribute__((aligned(4))) __hip_bfloat162 {
              __hip_bfloat16 x;
              __hip_bfloat16 y;
              __device__ __hip_bfloat162() = default;
              __device__ __hip_bfloat162(const __hip_bfloat162_raw& h2r)
                  : x(__hip_bfloat16(__hip_bfloat16_raw{h2r.x})),
                    y(__hip_bfloat16(__hip_bfloat16_raw{h2r.y})) {}
              __device__ operator __hip_bfloat162_raw() const {
                __hip_bfloat16_raw l = x;
                __hip_bfloat16_raw r = y;
                return __hip_bfloat162_raw{l.x, r.x};
              }
              __device__ operator float2() const {
                return float2{float(x), float(y)};
              }
            };

            #  define SCHAR_MIN     (-128)
            #  define SCHAR_MAX     127
            #  define UCHAR_MAX	255
            #   define CHAR_MIN	0
            #   define CHAR_MAX	UCHAR_MAX
            #  define SHRT_MIN	(-32768)
            #  define SHRT_MAX	32767

            #include <hip/hip_fp8.h>
        """ + source

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    prog, module, kernel = compile_source(source, kernel_name, arch)

    n = self.numel()
    out_d = hip_check(hip.hipMalloc(n * other_type.bit_width // 8))
    grid_dim, block_dim = calculate_dims(props, n)

    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid_dim,
            *block_dim,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=[self.data_d, ctypes.c_uint64(n), out_d],
        ))
    hip_check(hip.hipStreamSynchronize(0))

    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    return Tensor(out_d, dtype=other_type, shape=self.shape, device=self.device)


def reduction_kernel(self, host_impl):
    # avoid circular import
    from .tensor import Tensor

    assert self.dtype.is_native_numpy_type()
    if self.data_h is not None:
        return Tensor(host_impl(self.data_h))
    # Perform calculation on host, then copy back to device
    data_h = self.cpu().data_h
    out_h = host_impl(data_h)
    out_size = self.dtype.bit_width
    out_d = hip_check(hip.hipMalloc(out_size))
    hip_check(hip.hipMemcpyHtoD(out_d, out_h, out_size))
    return Tensor(out_d, dtype=self.dtype, shape=tuple(), device=self.device)
