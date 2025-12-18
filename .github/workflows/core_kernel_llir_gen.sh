#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!

set -xeo pipefail

echo "=== Clean up cache ==="

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Setup Environment ==="

cd /ffm && source ffmlite_env.sh && cd -
export HSA_MODEL_NUM_THREADS=1
export HSA_MODEL_TOML=".github/workflows/ffm_config.toml"
export HSA_MODEL_ARGS=ffm_enable_time_slicing

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Sanity Check ==="

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

# Check if FFM configurations are enabled properly
grep "dona.component.jitcu.enable_time_slicing=true" ./hierarchy_runtime_params.conf

echo "=== Setting up working directory ==="

export TRITON_HOME="/llir"
rm -rf $TRITON_HOME/.triton/cache

echo "=== Gathering CORE BF16 Gluon GEMM/Attention Kernels ==="
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250_tbuf_wp.py -M 1024 -N 1024 -K 1024 --num-warps 8
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py --num-warps=12 --num-buffers=2 --persistent --warp-specialized
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py --pipeline

# TODO: Fix failures in mxfp variants when ffm_enable_time_slicing is enabled.
echo "=== Gathering CORE MXFP Gluon GEMM/Attention Kernels ==="
unset HSA_MODEL_ARGS
# TODO: uncomment. Temp disabled for experimenting
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 8192 -N 8192 -K 8192 -BM 256 -BN 256 -BK 256 --num_warps 4 --num_buffers 2 --dtype_a float8_e4m3 --dtype_b float8_e4m3 --scale_preshuffled --with_a_scale --single_warp_schedule
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 1 --num_k_heads 1 --head_sz 128 --block_m 128 --block_n 128 --scale_type global --p_k_width 8 --pipelined

echo "=== Saving LLIR into llir_kernels directory ==="
cd $TRITON_HOME
rm -rf llir_kernels && mkdir llir_kernels/
find .triton/cache -type f -name "*.llir" -exec cp -t ./llir_kernels {} +
