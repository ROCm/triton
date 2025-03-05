#!/bin/bash

set -euxo pipefail

export HIP_VISIBLE_DEVICES=1

rm -rf origin changed

unset DISABLE_LLVM_OPT
rm -rf ~/.triton/cache/ && python MLA_decode.py
DIR=$(dirname $(find ~/.triton/cache/* -name *_fwd_grouped_persistent_kernel_stage1.llir))
cp -r $DIR origin


export DISABLE_LLVM_OPT="sink-insts-to-avoid-spills"
rm -rf ~/.triton/cache/ && python MLA_decode.py
DIR=$(dirname $(find ~/.triton/cache/* -name *_fwd_grouped_persistent_kernel_stage1.llir))
cp -r $DIR changed
