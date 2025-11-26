#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!
set -xeo pipefail

echo "=== Clean up cache ==="

rm -rf ~/.triton/cache

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Setup Environment ==="

cd /ffm && source ffmlite_env.sh && cd -
export HSA_MODEL_NUM_THREADS=1
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check ==="

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Run Triton Unit Tests ==="

pytest --count=1 -n 12 python/test/unit/language/test_core.py::test_override_arch
pytest --count=1 -n 12 python/test/unit/language/test_core.py::test_num_ctas_pre_sm90
pytest --count=1 -n 12 python/test/unit/language/test_core.py::test_scan2d
