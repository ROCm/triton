#!/usr/bin/env bash

# This script is meant to be used by the CI bots so paths in the below
# are quite specific to the docker and file organizations within.
#
# Don't run it as a generally applicable script!

set -xeo pipefail

echo "=== Clean up cache ==="

rm -rf ~/.triton/cache

echo "=== Setup Environment ==="

cd /ffm && source ffmlite_env.sh && cd -
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Build and Install Triton ==="

export PYTHON="python3"
export TRITON_BUILD_WITH_CLANG_LLD="TRUE"
export TRITON_BUILD_WITH_CCACHE="TRUE"
export CCACHE_COMPRESS="true"

LLVM_LIBRARY_DIR=/llvm LLVM_SYSPATH=/llvm pip3 install --no-build-isolation .

echo "=== Sanity Check ==="

python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"
which roccap

echo "=== Invoke roccap ==="

GIT_SHA="$(git rev-parse HEAD)"
SCRIPT_PATH="$(realpath $0)"

cd /roccap  # To make sure we generate the CAP file inside a known place
rm -rf *    # Clean old data if any

# Save the commit and script for reference
echo $GIT_SHA >> commit.txt
cp ${SCRIPT_PATH} .

export HSA_KMT_MODEL_GPUVM_BASE=0x200000000
export HSA_KMT_MODEL_GPUVM_SIZE=0xF00000000
export HSA_MODEL_NUM_THREADS=16

# 1. Change the following arguments according to your workload

# Cap file will be generated as mxfp_fa_gfx1250_e4m3_e4m3_demo.cap
NAME="mxfp_fa_gfx1250_e4m3_e4m3_demo"

# Where you want to put your cap file later.
# !Make sure to put cap file to this directory and give rwx permission bits.
CAPFILE_ROOT="/proj/triton_regr/TRITON/MXFP_FA/01011970/"

# Enable itrace or not.
# Enabling itrace will take significant more time to finish.
ENABLE_ITRACE=false

# [Optional]Change the group name to better represent your workload.
# It's fine to not change.
GROUP_NAME="CU_Tile_GEMM"

# 2. Change the following to the command you'd like to run to generate CAP file
#cmd=(
#  roccap capture --loglevel trace python3
#  /code/third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py
#  -M 8192 -N 8192 -K 1024 --num-warps=4 --num-buffers=2
#  --prefetch-lds --single-warp-schedule
#)

cmd=(
  roccap capture --loglevel trace --file "${NAME}.cap" python3
  /code/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py
  --q_type e4m3 --kv_type e4m3 --batch 1
  --seqlen_q 8192 --seqlen_k 8192
  --num_q_heads 2 --num_k_heads 2
  --head_sz 128 --block_m 128 --block_n 128
  --pipelined
  --scale_type block --scale_preshuffled
  --disable_p_scaling --p_k_width=16
)

"${cmd[@]}"

find . -name "*.cap" -exec roccap play {} \;

# 3. Generate AM metadata(aqlfile.txt and group_file.txt)

gen_am_cmd=(
  python3 /code/mi400/tools/generate_am_metadata.py
  -n ${NAME}
  -r ${CAPFILE_ROOT}
  -g ${GROUP_NAME}
)

if $ENABLE_ITRACE; then
  gen_am_cmd+=('-it')
fi

"${gen_am_cmd[@]}"
