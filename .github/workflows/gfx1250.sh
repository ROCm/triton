#!/usr/bin/bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!

set -xeo pipefail

pwd && ls

echo "=== Setup Environment ==="

export FFM_PATH=/data/mi450
export FFM_BIN_PATH=$FFM_PATH/_builds/Release/bin
export HSA_MODEL_LIB=$FFM_PATH/_builds/Release/lib/libhsakmtmodel.so
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_MODEL_TOPOLOGY=$FFM_PATH/topology
export HSA_MODEL_NUM_THREADS=$(nproc)
export ROCM_PATH=/opt/rocm
export TARGET_ARCH=gfx1250
export LD_LIBRARY_PATH=$ROCM_PATH/lib

echo "=== Build and Install Triton ==="

pip uninstall -y triton pytorch-triton pytorch-triton-rocm

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/data/build/amd-mlir-debug LLVM_SYSPATH=/data/build/amd-mlir-debug pip3 install --no-build-isolation .

echo "=== Run GEMM Tests ==="

python3 mi400/test_mxgemm_hipdriver.py

echo "=== Run Attention Tests ==="

python3 mi400/test_mxfa_hipdriver.py -c 0
python3 mi400/test_mxfa_hipdriver.py -c 1

echo "=== Run Gluon GEMM Tests ==="

python3 mi400/test_gemm_gluon.py
