// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx1150 --convert-builtin-func-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir | sed 's/ptx_kernel/amdgpu_kernel/g' \
// RUN:   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -mattr=+wavefrontsize32 -o - | FileCheck %s

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx1151 --convert-builtin-func-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir | sed 's/ptx_kernel/amdgpu_kernel/g' \
// RUN:   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr=+wavefrontsize32 -o - | FileCheck %s

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx1152 --convert-builtin-func-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir | sed 's/ptx_kernel/amdgpu_kernel/g' \
// RUN:   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1152 -mattr=+wavefrontsize32 -o - | FileCheck %s

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx1153 --convert-builtin-func-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir | sed 's/ptx_kernel/amdgpu_kernel/g' \
// RUN:   | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1153 -mattr=+wavefrontsize32 -o - | FileCheck %s

// Verify that RDNA3 (and 3.5) cache modifier qualifiers produce the correct GLC/SLC/DLC
// flags on buffer_load_b32 / buffer_store_b32 instructions in AMDGCN assembly.
//
// DLC (bit 2) = non-temporal hint for MALL. DLC=1 -> skip MALL allocation.
//
//   .ca / .wb  -> cachepolicy 0  -> (no flags)
//   .cg load   -> cachepolicy 1  -> glc
//   .cs / .cv  -> cachepolicy 7  -> glc slc dlc
//   .wt        -> cachepolicy 7  -> glc slc dlc

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // .ca load + .wb store: no glc/slc/dlc flags
  // CHECK-LABEL: load_ca_store_wb:
  // CHECK-NOT: glc
  // CHECK-NOT: slc
  // CHECK-NOT: dlc
  // CHECK-LABEL: load_cg:
  tt.func @load_ca_store_wb(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %val = amdg.buffer_load %ptr[%offset] cacheModifier = ca : tensor<128xf32, #blocked>
    amdg.buffer_store %val, %ptr[%offset] cacheModifier = wb : tensor<128xf32, #blocked>
    tt.return
  }

  // .cg load: GLC only (bypass GL1, cache in GL2)
  // CHECK: buffer_load_b32 {{.*}} glc{{$}}
  // CHECK-NOT: dlc
  // CHECK-LABEL: load_cs_store_cs:
  tt.func @load_cg(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %val = amdg.buffer_load %ptr[%offset] cacheModifier = cg : tensor<128xf32, #blocked>
    amdg.buffer_store %val, %ptr[%offset] : tensor<128xf32, #blocked>
    tt.return
  }

  // .cs load + .cs store: GLC|SLC|DLC (non-temporal at all levels)
  // CHECK: buffer_load_b32 {{.*}} glc slc dlc
  // CHECK: buffer_store_b32 {{.*}} glc slc dlc
  tt.func @load_cs_store_cs(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %val = amdg.buffer_load %ptr[%offset] cacheModifier = cs : tensor<128xf32, #blocked>
    amdg.buffer_store %val, %ptr[%offset] cacheModifier = cs : tensor<128xf32, #blocked>
    tt.return
  }

  // .cv load: GLC|SLC|DLC (non-temporal, volatile)
  // CHECK-LABEL: load_cv:
  // CHECK: buffer_load_b32 {{.*}} glc slc dlc
  tt.func @load_cv(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %val = amdg.buffer_load %ptr[%offset] cacheModifier = cv : tensor<128xf32, #blocked>
    amdg.buffer_store %val, %ptr[%offset] : tensor<128xf32, #blocked>
    tt.return
  }

  // .wt store: GLC|SLC|DLC (write-through, non-temporal)
  // CHECK-LABEL: store_wt:
  // CHECK: buffer_store_b32 {{.*}} glc slc dlc
  tt.func @store_wt(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %val = amdg.buffer_load %ptr[%offset] : tensor<128xf32, #blocked>
    amdg.buffer_store %val, %ptr[%offset] cacheModifier = wt : tensor<128xf32, #blocked>
    tt.return
  }
}
