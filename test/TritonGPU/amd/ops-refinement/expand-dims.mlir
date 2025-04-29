// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942' | FileCheck %s

// CHECK-LABEL: @bool_kernel
// CHECK-NOT: amdgpu.extract_slice
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bool_kernel(%arg0: tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi1, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
    tt.return %0 : tensor<32x1xi1, #blocked>
  }
}

// -----

// CHECK-LABEL: @ptr_kernel
// CHECK-COUNT-2: amdgpu.extract_slice
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ptr_kernel(%arg0: tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1x!tt.ptr<f32>, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    tt.return %0 : tensor<32x1x!tt.ptr<f32>, #blocked>
  }
}
