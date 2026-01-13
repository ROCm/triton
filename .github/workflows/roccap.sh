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

cd /ffm-base && source ffmlite_env.sh && cd -
export LD_LIBRARY_PATH=/ffm-update:$LD_LIBRARY_PATH
export HSA_MODEL_LIB=/ffm-update/libhsakmtmodel.so
# Prefer the NPI ROCm's libraries over the ones shipped with FFM Lite
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check ==="

pip show torch
pip show triton
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"
which roccap

echo "=== Invoke roccap ==="

env | grep "ROCCAP_"

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

# 1. Invoke roccap
cd /code/
rm -rf *.cap

export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="/roccap/"

CAP_DISPATCH="${ROCCAP_KERNEL_REGEX}/${ROCCAP_DISPATCH_NUM}"
roccap capture --loglevel trace --disp "${CAP_DISPATCH}" --file "${ROCCAP_NAME}.cap" "$@"

mv *.cap /roccap/
mv roc_capture.log /roccap/

find . -name "*.cap" -exec roccap play {} \;

# 2. Generate AM metadata(aqlfile.txt and group_file.txt)
cd /roccap

gen_am_cmd=(
  python3 /code/mi400/tools/generate_am_metadata.py
  -n ${ROCCAP_NAME}
  -r ${ROCCAP_CAPFILE_ROOT}
  -g ${ROCCAP_GROUP_NAME}
  --num_xcc ${ROCCAP_NUM_XCC}
)

if $ROCCAP_ENABLE_ITRACE; then
  gen_am_cmd+=('-it')
fi

if $ROCCAP_ENABLE_TTRACE; then
  gen_am_cmd+=('-tt')
fi

"${gen_am_cmd[@]}"
