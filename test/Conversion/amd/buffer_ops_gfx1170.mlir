// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s

// gfx1170 uses the gfx11/RDNA3 buffer encoding (it is not a gfx12 target).
// These tests lock in the RDNA3-specific pieces of the buffer-op lowering that
// differ from the CDNA (gfx942/gfx950) and gfx12 (gfx1250) targets covered by
// buffer_load_store.mlir / buffer_ops_gfx1250.mlir:
//   - the cache-modifier control bits (getCtrlBitsForCacheModifierOnRDNA3), and
//   - the buffer-atomic cache policy (getBufferAtomicCachePolicy, no SCOPE_DEV).

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: buffer_load_cs
    tt.func @buffer_load_cs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // The resource descriptor is built via make.buffer.rsrc (RDNA path).
        // The exact RDNA descriptor flags word is checked in
        // amd-buffer-cache-modifiers-rdna3.mlir; here we focus on the
        // cache-modifier aux value and the atomic cache policy.
        // CHECK: rocdl.make.buffer.rsrc
        // RDNA3 cache-modifier bits for `cs`: glc|slc|dlc = 0b111 = 7.
        // (CDNA gfx942/gfx950 encode `cs` as 3; this is the RDNA-specific value.)
        // CHECK: %[[AUX:.*]] = llvm.mlir.constant(7 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, %[[AUX]] : f32
        %ret = amdg.buffer_load %arg0[%offset] cacheModifier = cs : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: buffer_store_cs
    tt.func @buffer_store_cs(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: rocdl.make.buffer.rsrc
        // RDNA3 cache-modifier bits for `cs`: glc|slc|dlc = 0b111 = 7.
        // CHECK: %[[AUX:.*]] = llvm.mlir.constant(7 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[AUX]] : f32
        amdg.buffer_store %value, %arg0[%offset] cacheModifier = cs : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: buffer_atomic_rmw_fadd_f32
    tt.func @buffer_atomic_rmw_fadd_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}, %values : tensor<128xf32, #blocked0>) {
        // CHECK: rocdl.make.buffer.rsrc
        // A single release fence precedes the atomics.
        // CHECK: llvm.fence syncscope("agent") release
        // gfx1170 uses the RDNA/gfx11 buffer-atomic cache policy (cpol = 0 when
        // the result is unused). It must NOT emit the gfx12 SCOPE_DEV (16) or
        // SC0|SCOPE_DEV (17) encodings used by gfx1250.
        // CHECK-NOT: llvm.mlir.constant(16 : i32)
        // CHECK-NOT: llvm.mlir.constant(17 : i32)
        // sizePerThread=4 => 4 f32 atomic fadd calls (BUFFER_ATOMIC_ADD_F32).
        // CHECK-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // A single acquire fence follows the atomics.
        // CHECK: llvm.fence syncscope("agent") acquire
        %ret = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
  }
}
