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

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Install triton_kernels ==="

cd python/triton_kernels && pip3 install -e . && cd -

echo "=== Setup Environment ==="

source /ffm-base/ffmlite_env.sh
export LD_LIBRARY_PATH=/ffm-update:$LD_LIBRARY_PATH
export HSA_MODEL_LIB=/ffm-update/libhsakmtmodel.so
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
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py -M 1024 -N 1024 -K 1024
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py --num-warps=12 --num-buffers=2 --persistent --warp-specialized
HSA_MODEL_NUM_THREADS=2 python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py --attention-type pipeline

# TODO: Fix failures in mxfp variants when ffm_enable_time_slicing is enabled.
echo "=== Gathering CORE MXFP Gluon GEMM/Attention Kernels ==="
unset HSA_MODEL_ARGS
# TODO: uncomment. Temp disabled for experimenting
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 8192 -N 8192 -K 8192 -BM 256 -BN 256 -BK 256 --num_warps 4 --num_buffers 2 --dtype_a float8_e4m3 --dtype_b float8_e4m3 --scale_preshuffled --with_a_scale --schedule 'sliceK'
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 1 --num_k_heads 1 --head_sz 128 --block_m 256 --block_n 128 --scale_type global --pipelined --num_warps 4

export HSA_ENABLE_SDMA=0
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/moe_gfx1250.py -b 256 -d1 512 -d2 3072 -a dispatch -et 128 -ea 4 --num_warps 4 --num_buffers 2 -bm 256 -bn 256 -bk 256 --schedule sliceNK
HSA_MODEL_NUM_THREADS=4 python3 third_party/amd/python/examples/gluon/moe_gfx1250.py -b 256 -d1 512 -d2 3072 -a combine -et 128 -ea 4 --num_warps 4 --num_buffers 2 -bm 256 -bn 256 -bk 256 --schedule sliceNK

echo "=== Saving LLIR into llir_kernels directory ==="
cd $TRITON_HOME
rm -rf llir_kernels && mkdir llir_kernels/
find .triton/cache -type f -name "*.llir" -exec cp -t ./llir_kernels {} +
