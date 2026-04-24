#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!
set -xeo pipefail

# Forward termination signals to all child processes (pytest workers, etc.)
# Without this, pytest-xdist workers survive when the script is killed.
cleanup() {
    echo "=== Cleaning up child processes ==="
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 3
    pkill -KILL -P $$ 2>/dev/null || true
}
trap cleanup EXIT

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
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite--we need libhipblaslt.so there.
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check ==="

pip install pytest-timeout

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
    "test_load_store_same_ptr" # takes >60 mins
    # Excluse pattern for test_matmul.py, each takes > 1 hr
    "test_preshuffle_scale_mxfp_cdna4"
    "test_batched_mxfp"
    "test_mxfp8_mxfp4_matmul"
    "test_block_scale_fp4"
    # Exclude patterns for test_tensor_descriptor.py
    "test_tensor_descriptor_rank_reducing_matmul[float32]" # fails, but rank_reducing_load passes
    "test_tensor_descriptor_reduce"
    "test_tma_scatter"
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

uptime

# Use -p no:forked to disable forking to avoid RuntimeError: Cannot re-initialize CUDA in forked subprocess.
pytest -n 64 --durations=20 --maxfail=1 -k "$K_EXPR" -p no:forked python/test/unit/language/test_core.py

pytest -n 32 --durations=20 --maxfail=1 -k "$K_EXPR" -p no:forked python/test/unit/language/test_matmul.py \
    --deselect 'python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-512-64-32-2-float64-float64]' # Can take 10min!

pytest -n 32 --durations=20 --maxfail=1 -k "$K_EXPR" -p no:forked \
    python/test/unit/test_debug.py \
    python/test/unit/runtime

pytest -n 64 --durations=20 --maxfail=1 -k "$K_EXPR" -p no:forked -vv --timeout=300 --timeout-method=thread \
    python/test/unit/language/test_tensor_descriptor.py \
    --deselect 'python/test/unit/language/test_tensor_descriptor.py::test_host_tensor_descriptor_in_tuple[int16]' # Fails after #657 due to s_trap change from upstream #9692

pytest -n 16 --durations=10 python/test/unit/language/test_conversions.py

echo "=== Run Triton GEMM/Attention Tests ==="

# Enable time_slicing for mxfp_fa.py - causes numeric issues (https://github.com/ROCm/triton-internal/issues/1683)
unset HSA_MODEL_ARGS
# TODO: After #668 the following are failing due to NaNs. Investigate and fix.
pytest --count=1 -n 16 --durations=2 third_party/amd/python/examples/mxfp_fa.py \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[True-3-e4m3-e4m3-64-128-256-16-1]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[True-3-e4m3-e4m3-64-128-256-1-1]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e2m1-e4m3-64-128-256-1-2]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e2m1-e4m3-64-128-256-16-1]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e2m1-e4m3-64-128-256-1-1]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e2m1-e4m3-64-128-256-16-2]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e4m3-e4m3-64-128-256-16-1]' \
    --deselect 'third_party/amd/python/examples/mxfp_fa.py::test_mha[False-3-e4m3-e4m3-64-128-256-16-2]'

PYTHONPATH=$PWD/mi400 pytest --count=1 -n 16 --durations=10 \
    mi400/test_gemm_hipdriver.py \
    mi400/test_mxgemm_hipdriver.py \
    mi400/test_softmax_hipdriver.py

echo "=== Run AMD-specific Tests ==="

# gfx1250 doesn't support sanitizer yet
# test_pointer_optimization.py passes but takes > 30 mins
# test_gluon_gfx1250.py is run in gluon.sh
pytest -n 8  --durations=10 third_party/amd/python/test/ \
                --ignore=third_party/amd/python/test/test_scalarize_packed_fops.py \
                --ignore=third_party/amd/python/test/test_address_sanitizer.py \
                --ignore=third_party/amd/python/test/test_gluon_gfx1250_consan.py \
                --ignore=third_party/amd/python/test/test_pointer_optimization.py \
                --ignore=third_party/amd/python/test/test_gluon_gfx1250.py

echo "=== Test TDM widh async_copy disabled"

TRITON_HIP_USE_ASYNC_COPY=0 pytest -n 16 -s ./python/test/unit/language/test_tensor_descriptor.py::test_make_tensor_descriptor_matmul

echo "=== Install triton_kernels ==="

cd python/triton_kernels && pip3 install -e . && cd -

echo "=== Run Triton MoE Tests ==="

# Apply the patch to walk around Simulator issue: https://github.com/AMD-GFX-Modeling/ffm/issues/3174
# We don't want to modify common path shared with upstream.
# Instead we patch the files directly.
sed -i 's/assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)/assert_close(ref_y.cpu(), tri_y.cpu(), maxtol=maxtol, rmstol=rmstol)/g' python/triton_kernels/tests/test_matmul.py
sed -i "s/buffer = alloc_rand(buffer_shape, device=device, dtype=buffer_dtype)/buffer = alloc_rand(buffer_shape, device='cpu', dtype=buffer_dtype).to(device)/" python/triton_kernels/triton_kernels/testing.py

# Due to time limit, we can only select a very limited set of tests to run.
# Node IDs must match python/triton_kernels/tests/test_matmul.py::test_op parametrization
TRITON_MOE_TESTS=(
    "test_op[None-False-False-True-True-None-128-768-512-1024-batched-float16-float16-None-10-1-False-False-False-None-False-False-False-True-None]"
    "test_op[None-False-True-False-True-None-128-16-256-256-ragged-float8_e5m2-mxfloat4_e2m1-None-10-1-False-True-False-None-False-False-False-True-None]"
    "test_op[None-False-True-False-True-None-128-300-400-832-ragged-float8_e5m2-mxfloat4_e2m1-None-10-1-False-False-False-None-False-False-False-True-None]"
    "test_op[None-False-False-False-False-None-16-727-577-859-ragged-float16-float16-None-10-1-False-False-False-None-False-False-False-True-None]"
)

K_EXPR=""
for i in "${!TRITON_MOE_TESTS[@]}"; do
    if [ $i -gt 0 ]; then
        K_EXPR="$K_EXPR or "
    fi
    K_EXPR="$K_EXPR${TRITON_MOE_TESTS[$i]}"
done

export HSA_ENABLE_SDMA=0
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 4 --durations=0 -k "$K_EXPR" python/triton_kernels/tests/test_matmul.py

# Revert the patch
sed -i 's/assert_close(ref_y.cpu(), tri_y.cpu(), maxtol=maxtol, rmstol=rmstol)/assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)/g' python/triton_kernels/tests/test_matmul.py
sed -i "s/buffer = alloc_rand(buffer_shape, device='cpu', dtype=buffer_dtype).to(device)/buffer = alloc_rand(buffer_shape, device=device, dtype=buffer_dtype)/" python/triton_kernels/triton_kernels/testing.py
