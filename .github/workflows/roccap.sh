#!/usr/bin/env bash

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
export HSA_MODEL_NUM_THREADS=16
export HSA_MODEL_TOML=".github/workflows/ffm_config.toml"
#export HSA_MODEL_ARGS=ffm_enable_time_slicing

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Sanity Check ==="

python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

echo "=== Invoke roccap ==="

cd /roccap  # To make sure we generate the CAP file inside a known place
cp "$0" .   # To save the current command for later reference
echo "$(git rev-parse HEAD)" >> commit.txt

# Change the following to the command you'd like to run to generate CAP file
roccap capture --loglevel trace python3 \
    /code/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py \
        --q_type e4m3 --kv_type e4m3 --batch 1 \
        --seqlen_q 8192 --seqlen_k 8192 --num_q_heads 2 --num_k_heads 2 \
        --head_sz 128 --block_m 128 --block_n 128 --pipelined \
        --scale_type block --scale_preshuffled \
        --disable_p_scaling --p_k_width=16
