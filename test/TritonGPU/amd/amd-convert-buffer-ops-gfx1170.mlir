// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1170 analyze-small-tensor-ofst=true" | FileCheck %s

// gfx1170 buffer-atomic RMW gating (ConvertToBufferOps). gfx1170 has buffer
// atomic RMW instructions (supportsBufferAtomicRMW) but, like gfx11/RDNA3,
// only supports BUFFER_ATOMIC_ADD_F32 for floating-point add
// (supportsBufferAtomicFadd == F32 only). Integer RMW is universally
// supported. Float adds of other types (f16/bf16/f64) must not be converted to
// a buffer atomic; they stay as global tt.atomic_rmw (lowered later via CAS).

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1170", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func public @atomic_add_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %vals: tensor<1024xf32, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // f32 fadd is supported: converted to a buffer atomic RMW.
    // CHECK: amdg.buffer_atomic_rmw fadd
    // CHECK-NOT: tt.atomic_rmw
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %vals : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1170", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_add_bf16
  tt.func public @atomic_add_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %vals: tensor<1024xbf16, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<bf16>, i32
    %4 = tt.splat %3 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    // bf16 fadd is unsupported on gfx11 buffer atomics: not converted.
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw fadd
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %vals : (tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1170", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_add_f16
  tt.func public @atomic_add_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %vals: tensor<1024xf16, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    %4 = tt.splat %3 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    // f16 fadd is unsupported on gfx11 buffer atomics: not converted.
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw fadd
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %vals : (tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xf16, #blocked>) -> tensor<1024xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1170", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_add_f64
  tt.func public @atomic_add_f64(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %vals: tensor<1024xf64, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f64>, i32
    %4 = tt.splat %3 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    // f64 fadd is unsupported on gfx11 buffer atomics (F32 only): not converted.
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw fadd
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %vals : (tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xf64, #blocked>) -> tensor<1024xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1170", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_add_i32
  tt.func public @atomic_add_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %vals: tensor<1024xi32, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i32>, i32
    %4 = tt.splat %3 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // Integer RMW is universally supported: converted to a buffer atomic RMW.
    // CHECK: amdg.buffer_atomic_rmw add
    // CHECK-NOT: tt.atomic_rmw
    %6 = tt.atomic_rmw add, acq_rel, gpu, %5, %vals : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }
}
