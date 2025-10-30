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

cd /ffm && source ffmlite_env.sh && cd -
export HSA_MODEL_NUM_THREADS=1

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Sanity Check ==="

python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

echo "=== Run Lit Tests ==="

make test-lit

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Run Gluon Unit Tests ==="

pytest --count=1 -n 32 third_party/amd/python/test/test_gluon_gfx1250.py
pytest --count=1 -n 16 python/test/gluon/test_frontend.py

echo "=== Run Gluon GEMM/Attention Tests ==="

pytest --count=1 -n 16 third_party/amd/python/examples/gluon/*

echo "=== Run E2E Upstream Tests ==="

#pytest --count=1 -n 16 python/test/unit/language/test_conversions.py::test_typeconvert_downcast_clamping
#pytest --count=1 -n 16 python/test/unit/language/test_conversions.py::test_typeconvert_upcast
# pytest --count=1 -n 16 python/test/unit/language/test_conversions.py::test_typeconvert_downcast # TODO: fix hang

echo "=== Run Triton GEMM/Attention Tests ==="

PYTHONPATH=$PWD/mi400 pytest --count=1 -n 16 \
    mi400/test_gemm_hipdriver.py \
    mi400/test_mxgemm_hipdriver.py \
    mi400/test_mxfa_hipdriver.py
