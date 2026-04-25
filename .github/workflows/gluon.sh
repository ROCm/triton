#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!

set -xeo pipefail

cleanup() {
    echo "=== Cleaning up child processes ==="
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 3
    pkill -KILL -P $$ 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Clean up cache ==="

rm -rf ~/.triton/cache

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
export HSA_MODEL_TOML=".github/workflows/ffm_config.toml"
export HSA_MODEL_ARGS=ffm_enable_time_slicing

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Sanity Check ==="

pip install pytest-timeout

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

# Check if FFM configurations are enabled properly
grep "dona.component.jitcu.enable_time_slicing=true" ./hierarchy_runtime_params.conf

echo "=== Run Lit Tests ==="

make test-nogpu

echo "=== Run Gluon Unit Tests ==="

pytest --count=1 -n 48 --durations=10 -vv --maxfail=1 third_party/amd/python/test/test_gluon_gfx1250.py
pytest --count=1 -n 16 --durations=10 python/test/gluon/test_frontend.py

echo "=== Run Gluon GEMM/Attention Tests ==="

uptime

HSA_MODEL_NUM_THREADS=2 pytest --count=1 -n 32 --durations=10 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py
HSA_MODEL_NUM_THREADS=2 pytest --count=1 -n 16 --durations=10 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py \
    --deselect 'third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py::test_streamk_gemm_tdm_4warps[True-258-258-510-False-2-32-32-64]' \
    --deselect 'third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py::test_streamk_gemm_tdm_4warps[False-258-258-510-False-2-32-32-64]' \
    --deselect 'third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py::test_streamk_gemm_tdm_4warps[False-258-258-510-False-4-32-32-64]' \
    --deselect 'third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py::test_streamk_gemm_tdm_8warps[258-258-510-False-3-32-32-64]' \
    --deselect 'third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py::test_streamk_gemm_tdm_8warps[258-258-510-True-3-32-32-64]'
HSA_MODEL_NUM_THREADS=2 pytest --count=1 -n 16 --durations=10 third_party/amd/python/examples/gluon/stream_copy_gfx1250.py

unset HSA_MODEL_ARGS
# TODO: Fix failues in test_runtime_mxgemm_tdm_8warps_pipeline
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 32 --forked --durations=10 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py
# TODO: Fix failure in test_block_scaled_attn_fwd[e4m3-e4m3-1-1024-1024-1-1-128-128-128-True-True-False-8-8] when ffm_enable_time_slicing is enabled.
# It seems ffm_enable_time_slicing gives incorrect numerics when >~70 VGPR spills.
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 16 --durations=10 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 16 --durations=10 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py
HSA_MODEL_NUM_THREADS=8 pytest --count=1 -n 4 --durations=10 mi400/test_kernel_metadata.py

echo "=== Install triton_kernels ==="

cd python/triton_kernels && pip3 install -e . && cd -

echo "=== Run Gluon MoE Tests ==="

# Bypass a FFM issue: https://github.com/AMD-GFX-Modeling/ffm/issues/3164
export HSA_ENABLE_SDMA=0

# MoE tests require NPI PyTorch, so we test them in Triton pipeline.
# TODO: FFM can't fully clean up the model at this moment, so we need to use --forked to run each test in a separate subprocess.
# Otherwise there will be segfaults when running multiple tests in the same process.
HSA_MODEL_NUM_THREADS=4 pytest --count=1 -n 32 --forked --durations=10 third_party/amd/python/examples/gluon/moe_gfx1250.py
export HSA_MODEL_ARGS=ffm_enable_time_slicing
