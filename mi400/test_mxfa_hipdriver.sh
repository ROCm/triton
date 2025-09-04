#!/bin/bash

ROCPLAY_PATH=$(realpath ../../rocplaycap/rocplaycap-src-4.5.0)
ROCCAP_BIN=${ROCPLAY_PATH}/bin/roccap
ROCCAP_OPTIONS="capture --loglevel trace"

CURR_DIR=$PWD
TOP_OUTPUT_DIR="${CURR_DIR}/cap-mxfa"
rm -rf ${TOP_OUTPUT_DIR}
mkdir -p ${TOP_OUTPUT_DIR}

for CASE in 0 1 2 3; do
  for Q_TYPE in "e4m3"; do
    for KV_TYPE in "e4m3" "e2m1"; do
      ${ROCCAP_BIN} ${ROCCAP_OPTIONS} python3 ./test_mxfa_hipdriver.py --case ${CASE} --q-type ${Q_TYPE} --kv-type ${KV_TYPE}
      dir_name=${TOP_OUTPUT_DIR}/${CASE}/${Q_TYPE}-${KV_TYPE}
      mkdir -p ${dir_name}
      mv ./roc_capture* ${dir_name}
    done
  done
done
