#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!
#
# This script generates a corpus of kernels with llir, ttgir, amdgcn
# organized by category directories.

set -xeo pipefail

cleanup() {
    echo "=== Cleaning up child processes ==="
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 3
    pkill -KILL -P $$ 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Build and Install Triton ==="

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

export TRITON_HIP_USE_ASYNC_COPY=1

echo "=== Sanity Check ==="

pip show torch
pip show triton

echo "=== Setting up working directory ==="

export TRITON_HOME="/llir"
export CORPUS_OUTPUT="$TRITON_HOME/corpus_kernels"
rm -rf $CORPUS_OUTPUT && mkdir -p $CORPUS_OUTPUT

# Counter for unique kernel directories within each category
declare -A KERNEL_COUNTERS

# Function to run a kernel command and collect artifacts
run_and_collect() {
    local category="$1"
    local cmd="$2"

    # Initialize counter for this category if not exists
    if [[ -z "${KERNEL_COUNTERS[$category]}" ]]; then
        KERNEL_COUNTERS[$category]=0
    fi

    # Increment counter
    KERNEL_COUNTERS[$category]=$((KERNEL_COUNTERS[$category] + 1))
    local kernel_num="${KERNEL_COUNTERS[$category]}"

    # Create output directory structure
    local kernel_dir="$CORPUS_OUTPUT/$category/kernel_$(printf '%03d' $kernel_num)"
    mkdir -p "$kernel_dir"

    # Save the command
    echo "$cmd" > "$kernel_dir/command.txt"

    # Clear triton cache before running
    rm -rf $TRITON_HOME/.triton/cache

    echo "=== Running: $cmd ==="
    echo "=== Category: $category, Kernel: $kernel_num ==="

    # Run the kernel command
    HSA_MODEL_NUM_THREADS=16 eval "$cmd" || {
        echo "ERROR: Command failed: $cmd"
        return 1
    }

    # Collect artifacts (all in same directory)
    find $TRITON_HOME/.triton/cache -type f -name "*.llir" -exec cp -t "$kernel_dir/" {} + 2>/dev/null || true
    find $TRITON_HOME/.triton/cache -type f -name "*.ttgir" -exec cp -t "$kernel_dir/" {} + 2>/dev/null || true
    find $TRITON_HOME/.triton/cache -type f -name "*.amdgcn" -exec cp -t "$kernel_dir/" {} + 2>/dev/null || true

    echo "=== Artifacts collected for $category/kernel_$kernel_num ==="
}

echo "=== Gathering Corpus Kernels ==="

# ============================================================================
# MHA prefill (mxfp_fa_gfx1250.py)
# ============================================================================
run_and_collect "MHA_prefill" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 64 --block_m 256 --block_n 128 --pipelined --num_warps 4 --scale_type block --disable_p_scaling"
run_and_collect "MHA_prefill" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 64 --block_m 256 --block_n 128 --pipelined --num_warps 4 --scale_type block --disable_p_scaling"
run_and_collect "MHA_prefill" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 128 --block_m 256 --block_n 128 --pipelined --num_warps 4 --scale_type block --disable_p_scaling"
run_and_collect "MHA_prefill" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 1 --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 128 --block_m 256 --block_n 128 --pipelined --num_warps 4 --scale_type block --disable_p_scaling"

# ============================================================================
# MHA decode (mxfp_fa_gfx1250.py)
# ============================================================================
run_and_collect "MHA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MHA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MHA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MHA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 16 --num_k_heads 16 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"

# ============================================================================
# MQA decode (mxfp_fa_gfx1250.py)
# ============================================================================
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 2 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 2 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 2 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 2 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 1 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 1 --head_sz 64 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 1 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"
run_and_collect "MQA_decode" "python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 --batch 64 --seqlen_q 1 --seqlen_k 8192 --num_q_heads 64 --num_k_heads 1 --head_sz 128 --block_m 16 --block_n 128 --pipelined --num_warps 1 --scale_type block --disable_p_scaling"

# ============================================================================
# mxfp_gemm_gfx1250 (MXFP GEMM kernels)
# ============================================================================
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 8192 -N 8192 -K 8192 -BM 256 -BN 256 -BK 256 --num_warps 4 --num_buffers 2 --dtype_a float8_e4m3 --dtype_b float8_e4m3 --scale_preshuffled --with_a_scale --schedule 'sliceK'"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 256 -N 256 -K 512 -BM 128 -BN 128 -BK 256 --num_warps 4 --num_buffers 2 --dtype_a float8_e5m2 --dtype_b float4 --scale_preshuffled --with_a_scale --schedule 'sliceK'"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 256 -N 256 -K 512 -BM 256 -BN 256 -BK 256 --num_warps 4 --num_buffers 2 --dtype_a float4 --dtype_b float8_e4m3 --scale_preshuffled --with_a_scale --schedule 'sliceNK' --async_copy_scale"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 256 -N 256 -K 512 -BM 128 -BN 128 -BK 128 --num_warps 4 --num_buffers 2 --dtype_a float8_e4m3 --dtype_b float8_e5m2 --scale_preshuffled --with_a_scale --schedule 'baseline'"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 512 -N 512 -K 512 -BM 256 -BN 256 -BK 128 --num_warps 8 --num_buffers 3 --dtype_a float8_e5m2 --dtype_b float4 --scale_preshuffled --schedule 'baseline' --pingpong"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 512 -N 512 -K 512 -BM 128 -BN 128 -BK 256 --num_warps 8 --num_buffers 3 --dtype_a float8_e4m3 --dtype_b float8_e5m2 --scale_preshuffled --with_a_scale --schedule 'sliceK' --pingpong"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 512 -N 512 -K 512 -BM 128 -BN 128 -BK 256 --num_warps 8 --num_buffers 3 --dtype_a float4 --dtype_b float8_e4m3 --scale_preshuffled --with_a_scale --schedule 'sliceK' --pingpong"
run_and_collect "mxfp_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py -M 512 -N 512 -K 512 -BM 256 -BN 256 -BK 128 --num_warps 8 --num_buffers 3 --dtype_a float4 --dtype_b float4 --scale_preshuffled --schedule 'baseline' --pingpong"

# ============================================================================
# f16_gemm_streamk_gfx1250 (F16 StreamK GEMM kernels)
# ============================================================================
export HSA_MODEL_ARGS=ffm_enable_time_slicing

# Check if ffm_enable_time_slicing is enabled properly
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"
grep "dona.component.jitcu.enable_time_slicing=true" ./hierarchy_runtime_params.conf

run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 256 -N 256 -K 512 --block-m 32 --block-n 32 --block-k 64 --num-warps 4 --num-buffers 2"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 256 -N 256 -K 512 --block-m 32 --block-n 32 --block-k 64 --num-warps 4 --num-buffers 2 --prefetch"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 256 -N 256 -K 512 --block-m 32 --block-n 32 --block-k 64 --num-warps 4 --num-buffers 4 --prefetch"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 258 -N 258 -K 510 --block-m 32 --block-n 32 --block-k 64 --num-warps 4 --num-buffers 2 --prefetch"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 258 -N 258 -K 510 --block-m 32 --block-n 32 --block-k 64 --num-warps 4 --num-buffers 4 --prefetch"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 256 -N 256 -K 512 --block-m 32 --block-n 32 --block-k 64 --num-warps 8 --num-buffers 3"
run_and_collect "f16_gemm_streamk_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py -M 258 -N 258 -K 510 --block-m 32 --block-n 32 --block-k 64 --num-warps 8 --num-buffers 3"
unset HSA_MODEL_ARGS


# ============================================================================
# f16_fa_gfx1250 (F16 Flash Attention kernels)
# ============================================================================
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 512 --seqlen-k 512 --num-heads-q 8 --num-heads-k 8 --head-size 128 --block-m 128 --block-n 64 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 1024 --seqlen-k 1024 --num-heads-q 8 --num-heads-k 8 --head-size 64 --block-m 128 --block-n 128 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 2000 --seqlen-k 2000 --num-heads-q 8 --num-heads-k 8 --head-size 64 --block-m 128 --block-n 128 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 3 --seqlen-k 32 --num-heads-q 4 --num-heads-k 4 --head-size 128 --block-m 128 --block-n 32 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 1 --seqlen-k 100 --num-heads-q 8 --num-heads-k 8 --head-size 32 --block-m 128 --block-n 32 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 1 --seqlen-k 30 --num-heads-q 8 --num-heads-k 8 --head-size 32 --block-m 128 --block-n 32 --attention-type pipeline"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 1024 --seqlen-k 1024 --num-heads-q 8 --num-heads-k 8 --head-size 128 --block-m 256 --block-n 64 --attention-type pingpong"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 300 --seqlen-k 300 --num-heads-q 8 --num-heads-k 8 --head-size 64 --block-m 256 --block-n 32 --attention-type pingpong"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 8 --seqlen-q 1 --seqlen-k 1024 --num-heads-q 8 --num-heads-k 8 --head-size 128 --block-m 128 --block-n 32 --attention-type decode"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 512 --seqlen-k 512 --num-heads-q 8 --num-heads-k 8 --head-size 128 --block-m 128 --block-n 32 --attention-type default"
run_and_collect "f16_fa_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_fa_gfx1250.py -b 1 --seqlen-q 1 --seqlen-k 30 --num-heads-q 8 --num-heads-k 8 --head-size 32 --block-m 128 --block-n 32 --attention-type default"

# ============================================================================
# f16_gemm_gfx1250 (F16 GEMM kernels)
# ============================================================================
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 1024 -N 1024 -K 1024 --block_m 256 --block_n 256 --block_k 128 --num-warps 4 --num-buffers 2 --prefetch-lds --prefetch-l2-distance 0 --single-warp-schedule"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 1024 -N 1024 -K 1024 --block_m 256 --block_n 256 --block_k 128 --num-warps 4 --num-buffers 2 --prefetch-lds --prefetch-l2-distance 2 --single-warp-schedule"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 1024 -N 1024 -K 1024 --block_m 128 --block_n 128 --block_k 128 --num-warps 4 --num-buffers 3 --prefetch-lds --prefetch-l2-distance 0 --single-warp-schedule"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 1024 -N 1024 -K 1024 --block_m 128 --block_n 128 --block_k 128 --num-warps 4 --num-buffers 3 --prefetch-lds --prefetch-l2-distance 2 --single-warp-schedule"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 64 --num-warps 4 --num-buffers 2 --prefetch-l2-distance 0"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 64 --num-warps 4 --num-buffers 4 --prefetch-l2-distance 2"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 64 --num-warps 8 --num-buffers 3 --prefetch-l2-distance 0"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 64 --num-warps 4 --num-buffers 2 --persistent --prefetch-l2-distance 0"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 64 --num-warps 8 --num-buffers 3 --persistent --prefetch-lds --prefetch-l2-distance 2"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 256 -N 256 -K 512 --block_m 32 --block_n 32 --block_k 128 --num-warps 4 --num-buffers 4 --single-warp-schedule --prefetch-lds --prefetch-l2-distance 2"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 512 -N 256 -K 512 --block_m 64 --block_n 32 --block_k 64 --num-warps 4 --num-buffers 2 --ctas-per-cga 2 1 --prefetch-l2-distance 0"
run_and_collect "f16_gemm_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py -M 512 -N 512 -K 512 --block_m 64 --block_n 64 --block_k 64 --num-warps 8 --num-buffers 3 --ctas-per-cga 2 2 --prefetch-l2-distance 2"

# ============================================================================
# f16_gemm_warp_pipeline_gfx1250 (F16 GEMM Warp Pipeline kernels)
# ============================================================================
run_and_collect "f16_gemm_warp_pipeline_gfx1250" "python3 third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py -M 2048 -N 2048 -K 2048 --num-buffers 3"

echo "=== Corpus Generation Complete ==="

# Generate summary
echo "=== Summary ===" | tee "$CORPUS_OUTPUT/summary.txt"
for category in "$CORPUS_OUTPUT"/*/; do
    if [[ -d "$category" ]]; then
        category_name=$(basename "$category")
        kernel_count=$(find "$category" -maxdepth 1 -type d -name "kernel_*" | wc -l)
        success_count=$(find "$category" -name "status.txt" -exec grep -l "SUCCESS" {} \; | wc -l)
        echo "$category_name: $kernel_count kernels ($success_count successful)" | tee -a "$CORPUS_OUTPUT/summary.txt"
    fi
done

echo "=== Corpus saved to $CORPUS_OUTPUT ==="
