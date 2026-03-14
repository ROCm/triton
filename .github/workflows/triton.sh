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
#export LD_LIBRARY_PATH=/ffm-update:$LD_LIBRARY_PATH
#export HSA_MODEL_LIB=/ffm-update/libhsakmtmodel.so
export HSA_MODEL_NUM_THREADS=1
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check ==="

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Run FpSan Tests ==="
pytest --count=1 -n 1 --durations=10 python/test/gluon/test_fpsan.py -v --tb=short


echo "=== Run Triton Unit Tests ==="

# Array of test patterns to exclude
EXCLUDE_PATTERNS=(
    # Exclude pattern for test_core.py
    "test_dot" # Failing 8 test cases
    "test_load_scope_sem_coop_grid_cta_one" # coop group not supported in FFM
    "test_load_store_same_ptr" # takes >60 mins
    "test_cat_nd" # Failure after merge in upstream with #500
    # Excluse pattern for test_matmul.py
    "test_preshuffle_scale_mxfp_cdna4"
    "test_batched_mxfp"
    "test_mxfp8_mxfp4_matmul"
    "test_block_scale_fp4"
    # Exclude patterns for runtime tests:
    "test_async_compile_mock" # hangs indefinitely in FFM (threading/async issues in simulation)
)

# Build the -k expression: "not (pattern1 or pattern2 or ...)"
K_EXPR="not ("
for i in "${!EXCLUDE_PATTERNS[@]}"; do
    if [ $i -gt 0 ]; then
        K_EXPR="$K_EXPR or "
    fi
    K_EXPR="$K_EXPR${EXCLUDE_PATTERNS[$i]}"
done
K_EXPR="$K_EXPR)"

echo "Running pytest with filter: $K_EXPR"

# Use -p no:forked to disable forking to avoid RuntimeError: Cannot re-initialize CUDA in forked subprocess.
pytest -n 80 \
    --durations=20 \
    -k "$K_EXPR" \
    -p no:forked \
    python/test/unit/language/test_core.py \
    python/test/unit/language/test_matmul.py \
    python/test/unit/runtime \
    python/test/unit/test_debug.py

echo "=== Run AMD-specific Tests ==="

pytest --durations=10 third_party/amd/python/test/test_compiler_fence_gfx1250.py

echo "=== Test TDM widh async_copy disabled"

TRITON_HIP_USE_ASYNC_COPY=0 pytest -n 16 -s ./python/test/unit/language/test_tensor_descriptor.py::test_make_tensor_descriptor_matmul
