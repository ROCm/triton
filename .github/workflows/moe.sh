#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!
set -xeo pipefail

echo "=== Clean up cache ==="

sudo rm -rf ~/.triton/cache

echo "=== Build and Install Triton ==="

git config --global --add safe.directory /code

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Setup Environment ==="

source /ffm-base/ffmlite_env.sh
export LD_LIBRARY_PATH=/ffm-update:$LD_LIBRARY_PATH
export HSA_MODEL_LIB=/ffm-update/libhsakmtmodel.so
export HSA_MODEL_NUM_THREADS=1
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check ==="

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Install triton_kernels ==="

cd python/triton_kernels && pip3 install -e . && cd -

echo "=== Run Gluon MoE Tests ==="

# Bypass a FFM issue: https://github.com/AMD-GFX-Modeling/ffm/issues/3164
export HSA_ENABLE_SDMA=0

# MoE tests require NPI PyTorch, so we test them in Triton pipeline.
# TODO: FFM can't fully clean up the model at this moment, so we need to use --forked to run each test in a separate subprocess.
# Otherwise there will be segfaults when running multiple tests in the same process.
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 32 --forked --durations=10 third_party/amd/python/examples/gluon/moe_gfx1250.py
