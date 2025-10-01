#!/usr/bin/bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!

set -xeo pipefail

pwd && ls

echo "=== Clean up cache ==="
rm -rf ~/.triton/cache

echo "=== Setup Environment ==="

export FFM_PATH=/data/mi450-git
export FFM_BIN_PATH=$FFM_PATH/_builds/Release/bin
export HSA_MODEL_LIB=$FFM_PATH/_builds/Release/lib/libhsakmtmodel.so
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_MODEL_TOPOLOGY=$FFM_PATH/topology
export HSA_MODEL_NUM_THREADS=1
export ROCM_PATH=/opt/rocm
export TARGET_ARCH=gfx1250
export LD_LIBRARY_PATH=$ROCM_PATH/lib

pip uninstall -y triton pytorch-triton pytorch-triton-rocm
pip install pytest-repeat

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/data/build/amd-mlir-9f0b4533535f-debug-install LLVM_SYSPATH=/data/build/amd-mlir-9f0b4533535f-debug-install \
    pip3 install --no-build-isolation .

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Run Gluon Tests ==="

pytest --count=16 -n 16 -s -v third_party/amd/python/test/test_gluon_gfx1250.py
pytest --count=16 -n 16 -s -v python/test/gluon/test_frontend.py

echo "=== Run E2E Upstream Tests ==="

pytest --count=16 -n 16 -s -v python/test/unit/language/test_conversions.py::test_typeconvert_downcast_clamping
pytest --count=16 -n 16 -s -v python/test/unit/language/test_conversions.py::test_typeconvert_upcast
# pytest --count=16 -n 16 -s -v python/test/unit/language/test_conversions.py::test_typeconvert_downcast # TODO: fix hang

echo "=== Run GEMM Tests ==="

export PYTHONPATH=$PWD/mi400
pytest --count=16 -n 16 -s -v mi400/test_gemm_hipdriver.py
pytest --count=16 -n 16 -s -v mi400/test_mxgemm_hipdriver.py
pytest --count=16 -n 16 -s -v mi400/test_mxgemm_gluon.py

echo "=== Run Attention Tests ==="

pytest --count=16 -n 16 -v -s mi400/test_mxfa_hipdriver.py
