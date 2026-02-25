TRITON_LLVM_DEBUG_ONLY=tritongpu-prefetch \
  /home/developer/repos/triton/python/build/bin/triton-opt test/TritonGPU/prefetch.mlir -split-input-file -tritongpu-prefetch -canonicalize

# cd python/build
#TRITON_ALWAYS_COMPILE=1 \
#  TRITON_HIP_USE_LDS_PREFETCH=1 \
#  TRITON_LLVM_DEBUG_ONLY=tritongpu-prefetch \
#  MLIR_ENABLE_DUMP=0 \
#  lit -v python/build/test/

