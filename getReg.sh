#! /bin/bash

LLIR_FILE=$(find ~/.triton/cache/* -name *.llir)
ASM_FILE="${LLIR_FILE%.llir}.amdgcn"
grep vgpr_count $ASM_FILE
grep vgpr_spill $ASM_FILE
echo $ASM_FILE

NEW_FILE="${ASM_FILE%.amdgcn}.s"

if [ $# -gt 0 ]; then
    echo "clean up the assembly"
    cp $ASM_FILE $NEW_FILE
    sed -i '/local_/! {/\.loc/d}' $NEW_FILE
    sed -i '/\.Ltmp.*:/d' $NEW_FILE
fi
