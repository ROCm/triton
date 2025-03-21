[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %171 = math.exp2 %170 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %190 = math.exp2 %189 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %196 = arith.truncf %187 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %219 = arith.addi %151, %c64_i32 : i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %219 = arith.addi %151, %c64_i32 : i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %456 = arith.maxnumf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %456 = arith.maxnumf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %167 = arith.maxnumf %154, %166 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %165 = arith.addf %164, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %480 = arith.addf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %480 = arith.addf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %219 = arith.addf %218, %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %194 = arith.subf %189, %193 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %237 = arith.subf %154, %191 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %164 = arith.mulf %163, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %265 = arith.mulf %152, %264 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
srcShapePerCtaTile: 128x8
numReps: 1x16
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %314 = arith.mulf %153, %262 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %417 = arith.addi %408, %c64_i32 : i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %417 = arith.addi %408, %c64_i32 : i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %421 = arith.addi %418, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %434 = arith.addi %420, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %457 = arith.andi %456, %391 : i1
[tritonamdgpu-refine-ops]: failed to refine binary op: %457 = arith.andi %456, %391 : i1
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: i32, %arg27: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg28: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_3 = arith.constant dense<0.127517432> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %cst_8 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %c63_i32 = arith.constant 63 : i32
    %c54848_i32 = arith.constant 54848 : i32
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c3428_i32 = arith.constant 3428 : i32
    %cst_12 = arith.constant dense<0x7F800000> : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %cst_13 = arith.constant dense<3428> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_14 = arith.constant dense<true> : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %16 = tt.addptr %arg23, %2 : !tt.ptr<i32>, i32
    %17 = tt.addptr %16, %c1_i32 : !tt.ptr<i32>, i32
    %18 = tt.addptr %arg24, %2 : !tt.ptr<i32>, i32
    %19 = tt.addptr %18, %c1_i32 : !tt.ptr<i32>, i32
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%20: i32):  // 4 preds: ^bb0, ^bb2, ^bb5, ^bb36
    %21 = arith.cmpi slt, %20, %c1_i32 : i32
    cf.cond_br %21, ^bb2, ^bb37
  ^bb2:  // pred: ^bb1
    %22 = tt.load %16 : !tt.ptr<i32>
    %23 = tt.load %17 : !tt.ptr<i32>
    %24 = arith.subi %23, %22 : i32
    %25 = arith.cmpi sle, %3, %24 : i32
    %26 = tt.load %18 : !tt.ptr<i32>
    %27 = tt.load %19 : !tt.ptr<i32>
    %28 = arith.subi %27, %26 : i32
    cf.cond_br %25, ^bb3, ^bb1(%c1_i32 : i32)
  ^bb3:  // pred: ^bb2
    %29 = arith.addi %28, %c63_i32 : i32
    %30 = arith.divsi %29, %c64_i32 : i32
    %31 = arith.addi %0, %c1_i32 : i32
    %32 = arith.muli %31, %c128_i32 : i32
    %33 = arith.addi %32, %28 : i32
    %34 = arith.subi %33, %24 : i32
    %35 = arith.addi %34, %c63_i32 : i32
    %36 = arith.divsi %35, %c64_i32 : i32
    %37 = arith.minsi %30, %36 : i32
    %38 = arith.cmpi sle, %37, %c0_i32 : i32
    %39 = arith.cmpi sgt, %37, %c0_i32 : i32
    cf.cond_br %38, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %40 = arith.muli %2, %arg14 : i32
    %41 = tt.addptr %arg4, %40 : !tt.ptr<f16>, i32
    %42 = arith.muli %1, %arg15 : i32
    %43 = tt.addptr %41, %42 : !tt.ptr<f16>, i32
    %44 = arith.muli %22, %arg16 : i32
    %45 = tt.addptr %43, %44 : !tt.ptr<f16>, i32
    %46 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %47 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %48 = arith.muli %46, %47 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %49 = tt.splat %45 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %50 = tt.addptr %49, %48 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %51 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %53 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %54 = tt.broadcast %52 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %55 = tt.addptr %53, %54 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %56 = tt.splat %24 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %57 = arith.cmpi slt, %46, %56 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %58 = tt.broadcast %57 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %55, %cst_11, %58 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %59 = arith.muli %2, %c54848_i32 : i32
    %60 = tt.addptr %arg3, %59 : !tt.ptr<f32>, i32
    %61 = arith.muli %1, %c3428_i32 : i32
    %62 = tt.addptr %60, %61 : !tt.ptr<f32>, i32
    %63 = tt.splat %62 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %64 = tt.addptr %63, %12 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %65 = arith.cmpi slt, %12, %cst_13 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %64, %cst_12, %65 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    cf.cond_br %39, ^bb6, ^bb1(%c1_i32 : i32)
  ^bb6:  // pred: ^bb5
    %66 = arith.cmpi slt, %28, %c64_i32 : i32
    cf.cond_br %66, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %67 = arith.subi %c64_i32, %28 : i32
    cf.br ^bb9(%67 : i32)
  ^bb8:  // pred: ^bb6
    %68 = arith.remsi %28, %c64_i32 : i32
    cf.br ^bb9(%68 : i32)
  ^bb9(%69: i32):  // 2 preds: ^bb7, ^bb8
    %70 = arith.muli %2, %arg5 : i32
    %71 = tt.addptr %arg0, %70 : !tt.ptr<f16>, i32
    %72 = arith.muli %1, %arg6 : i32
    %73 = tt.addptr %71, %72 : !tt.ptr<f16>, i32
    %74 = arith.muli %22, %arg7 : i32
    %75 = tt.addptr %73, %74 : !tt.ptr<f16>, i32
    %76 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %77 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %78 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %79 = arith.muli %77, %78 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %80 = tt.splat %75 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %81 = tt.addptr %80, %79 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %82 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %84 = tt.expand_dims %82 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %85 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %86 = tt.broadcast %81 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %87 = tt.broadcast %84 : tensor<1x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %88 = tt.broadcast %85 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %89 = tt.addptr %86, %88 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %90 = arith.muli %2, %arg8 : i32
    %91 = tt.addptr %arg1, %90 : !tt.ptr<f16>, i32
    %92 = arith.muli %1, %arg9 : i32
    %93 = tt.addptr %91, %92 : !tt.ptr<f16>, i32
    %94 = arith.muli %26, %arg10 : i32
    %95 = tt.addptr %93, %94 : !tt.ptr<f16>, i32
    %96 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %97 = tt.expand_dims %96 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %98 = tt.splat %95 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %99 = tt.addptr %98, %97 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %100 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %101 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %102 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %103 = arith.muli %100, %102 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %104 = tt.broadcast %99 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %105 = tt.broadcast %103 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %106 = tt.addptr %104, %105 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %107 = arith.muli %2, %arg11 : i32
    %108 = tt.addptr %arg2, %107 : !tt.ptr<f16>, i32
    %109 = arith.muli %1, %arg12 : i32
    %110 = tt.addptr %108, %109 : !tt.ptr<f16>, i32
    %111 = arith.muli %26, %arg13 : i32
    %112 = tt.addptr %110, %111 : !tt.ptr<f16>, i32
    %113 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %114 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %115 = arith.muli %113, %114 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %116 = tt.splat %112 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %117 = tt.addptr %116, %115 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %118 = tt.broadcast %117 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %119 = tt.broadcast %85 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %120 = tt.addptr %118, %119 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %121 = tt.splat %24 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %122 = tt.splat %24 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %123 = arith.cmpi slt, %76, %121 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %124 = arith.cmpi slt, %77, %122 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %125 = tt.broadcast %123 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %126 = tt.broadcast %124 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %127 = tt.load %89, %126, %cst_11 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %128 = ttg.local_alloc %127 {allocation.offset = 0 : i32} : (tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %129 = ttg.local_load %128 : !ttg.memdesc<128x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %130 = arith.cmpi eq, %69, %c0_i32 : i32
    %131 = arith.remsi %24, %c128_i32 : i32
    %132 = arith.cmpi eq, %131, %c0_i32 : i32
    %133 = arith.andi %130, %132 : i1
    %134 = arith.xori %133, %true : i1
    %135 = arith.extui %134 : i1 to i32
    %136 = arith.addi %135, %c2_i32 : i32
    %137 = arith.minsi %136, %37 : i32
    %138 = arith.subi %37, %137 : i32
    %139 = arith.muli %37, %c64_i32 : i32
    %140 = arith.cmpi sgt, %138, %c0_i32 : i32
    cf.cond_br %140, ^bb10, ^bb18(%cst_8, %cst_7, %cst_4, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, i32)
  ^bb10:  // pred: ^bb9
    %141 = arith.muli %138, %c64_i32 : i32
    %142 = arith.muli %arg10, %c64_i32 : i32
    %143 = tt.splat %142 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %144 = arith.muli %arg13, %c64_i32 : i32
    %145 = tt.splat %144 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %146 = arith.cmpi sgt, %141, %c0_i32 : i32
    %147 = tt.splat %146 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %148 = tt.load %106, %147 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %149 = ttg.local_alloc %148 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %150 = arith.subi %141, %c64_i32 : i32
    cf.br ^bb11(%c0_i32, %cst_4, %cst_7, %cst_8, %106, %120, %149 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb11(%151: i32, %152: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, %153: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %154: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %155: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %156: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %157: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb10, ^bb12
    %158 = arith.cmpi slt, %151, %150 : i32
    cf.cond_br %158, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %159 = tt.addptr %155, %143 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %160 = tt.load %159 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %161 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %162 = tt.load %156 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %163 = tt.dot %129, %161, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %164 = amdgpu.extract_slice %163 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %165 = amdgpu.extract_slice %cst_3 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %166 = arith.mulf %164, %165 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %167 = amdgpu.extract_slice %163 [0, 8] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %168 = amdgpu.extract_slice %cst_3 [0, 8] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %169 = arith.mulf %167, %168 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %170 = amdgpu.extract_slice %163 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %171 = amdgpu.extract_slice %cst_3 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %172 = arith.mulf %170, %171 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %173 = amdgpu.extract_slice %163 [0, 24] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %174 = amdgpu.extract_slice %cst_3 [0, 24] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %175 = arith.mulf %173, %174 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %176 = amdgpu.extract_slice %163 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %177 = amdgpu.extract_slice %cst_3 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %178 = arith.mulf %176, %177 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %179 = amdgpu.extract_slice %163 [0, 40] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %180 = amdgpu.extract_slice %cst_3 [0, 40] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %181 = arith.mulf %179, %180 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %182 = amdgpu.extract_slice %163 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %183 = amdgpu.extract_slice %cst_3 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %184 = arith.mulf %182, %183 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %185 = amdgpu.extract_slice %163 [0, 56] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %186 = amdgpu.extract_slice %cst_3 [0, 56] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %187 = arith.mulf %185, %186 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %188 = amdgpu.extract_slice %cst_2 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %189 = arith.addf %166, %188 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %190 = amdgpu.extract_slice %cst_2 [0, 8] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %191 = arith.addf %169, %190 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %192 = amdgpu.extract_slice %cst_2 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %193 = arith.addf %172, %192 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %194 = amdgpu.extract_slice %cst_2 [0, 24] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %195 = arith.addf %175, %194 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %196 = amdgpu.extract_slice %cst_2 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %197 = arith.addf %178, %196 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %198 = amdgpu.extract_slice %cst_2 [0, 40] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %199 = arith.addf %181, %198 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %200 = amdgpu.extract_slice %cst_2 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %201 = arith.addf %184, %200 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %202 = amdgpu.extract_slice %cst_2 [0, 56] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %203 = arith.addf %187, %202 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %204 = amdgpu.concat %189, %191, %193, %195, %197, %199, %201, %203 [1, 8] : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %205 = "tt.reduce"(%204) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %206 = arith.maxnumf %154, %205 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %207 = tt.expand_dims %206 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %208 = tt.broadcast %207 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %209 = amdgpu.extract_slice %208 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %210 = arith.subf %189, %209 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %211 = amdgpu.extract_slice %208 [0, 8] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %212 = arith.subf %191, %211 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %213 = amdgpu.extract_slice %208 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %214 = arith.subf %193, %213 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %215 = amdgpu.extract_slice %208 [0, 24] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %216 = arith.subf %195, %215 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %217 = amdgpu.extract_slice %208 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %218 = arith.subf %197, %217 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %219 = amdgpu.extract_slice %208 [0, 40] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %220 = arith.subf %199, %219 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %221 = amdgpu.extract_slice %208 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %222 = arith.subf %201, %221 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %223 = amdgpu.extract_slice %208 [0, 56] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %224 = arith.subf %203, %223 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %225 = math.exp2 %210 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %226 = math.exp2 %212 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %227 = math.exp2 %214 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %228 = math.exp2 %216 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %229 = math.exp2 %218 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %230 = math.exp2 %220 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %231 = math.exp2 %222 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %232 = math.exp2 %224 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %233 = amdgpu.concat %225, %226, %227, %228, %229, %230, %231, %232 [1, 8] : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %234 = "tt.reduce"(%233) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %235 = arith.subf %154, %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %236 = math.exp2 %235 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %237 = tt.expand_dims %236 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %238 = tt.broadcast %237 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %239 = amdgpu.extract_slice %152 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %240 = amdgpu.extract_slice %238 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %241 = arith.mulf %239, %240 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %242 = amdgpu.extract_slice %152 [0, 8] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %243 = amdgpu.extract_slice %238 [0, 8] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %244 = arith.mulf %242, %243 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %245 = amdgpu.extract_slice %152 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %246 = amdgpu.extract_slice %238 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %247 = arith.mulf %245, %246 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %248 = amdgpu.extract_slice %152 [0, 24] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %249 = amdgpu.extract_slice %238 [0, 24] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %250 = arith.mulf %248, %249 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %251 = amdgpu.extract_slice %152 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %252 = amdgpu.extract_slice %238 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %253 = arith.mulf %251, %252 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %254 = amdgpu.extract_slice %152 [0, 40] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %255 = amdgpu.extract_slice %238 [0, 40] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %256 = arith.mulf %254, %255 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %257 = amdgpu.extract_slice %152 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %258 = amdgpu.extract_slice %238 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %259 = arith.mulf %257, %258 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %260 = amdgpu.extract_slice %152 [0, 56] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %261 = amdgpu.extract_slice %238 [0, 56] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %262 = arith.mulf %260, %261 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %263 = amdgpu.extract_slice %152 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %264 = amdgpu.extract_slice %238 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %265 = arith.mulf %263, %264 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %266 = amdgpu.extract_slice %152 [0, 72] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %267 = amdgpu.extract_slice %238 [0, 72] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %268 = arith.mulf %266, %267 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %269 = amdgpu.extract_slice %152 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %270 = amdgpu.extract_slice %238 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %271 = arith.mulf %269, %270 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %272 = amdgpu.extract_slice %152 [0, 88] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %273 = amdgpu.extract_slice %238 [0, 88] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %274 = arith.mulf %272, %273 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %275 = amdgpu.extract_slice %152 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %276 = amdgpu.extract_slice %238 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %277 = arith.mulf %275, %276 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %278 = amdgpu.extract_slice %152 [0, 104] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %279 = amdgpu.extract_slice %238 [0, 104] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %280 = arith.mulf %278, %279 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %281 = amdgpu.extract_slice %152 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %282 = amdgpu.extract_slice %238 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %283 = arith.mulf %281, %282 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %284 = amdgpu.extract_slice %152 [0, 120] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %285 = amdgpu.extract_slice %238 [0, 120] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %286 = arith.mulf %284, %285 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %287 = amdgpu.concat %241, %244, %247, %250, %253, %256, %259, %262, %265, %268, %271, %274, %277, %280, %283, %286 [1, 16] : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %288 = arith.mulf %153, %236 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %289 = arith.addf %288, %234 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %290 = arith.truncf %225 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %291 = arith.truncf %226 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %292 = arith.truncf %227 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %293 = arith.truncf %228 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %294 = arith.truncf %229 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %295 = arith.truncf %230 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %296 = arith.truncf %231 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %297 = arith.truncf %232 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %298 = amdgpu.concat %290, %291, %292, %293, %294, %295, %296, %297 [1, 8] : tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %299 = ttg.convert_layout %298 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %300 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %301 = ttg.local_load %300 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    %302 = tt.dot %299, %301, %287, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %303 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %304 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %305 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%305, %302, %289, %206, %159, %303, %304 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb13:  // pred: ^bb11
    %306 = arith.addi %141, %c63_i32 : i32
    %307 = arith.divsi %306, %c64_i32 : i32
    %308 = arith.cmpi sge, %307, %c1_i32 : i32
    gpu.barrier
    %309 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %310 = tt.splat %308 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %311 = tt.load %156, %310 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    cf.cond_br %308, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb14:  // pred: ^bb13
    %312 = tt.dot %129, %309, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb15(%312 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb15(%313: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb13, ^bb14
    %314 = arith.mulf %313, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %315 = arith.addf %314, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %316 = "tt.reduce"(%315) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %317 = arith.maxnumf %154, %316 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %318 = tt.expand_dims %317 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %319 = tt.broadcast %318 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %320 = arith.subf %315, %319 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %321 = math.exp2 %320 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %322 = "tt.reduce"(%321) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %323 = arith.subf %154, %317 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %324 = math.exp2 %323 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %325 = tt.expand_dims %324 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %326 = tt.broadcast %325 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %327 = arith.mulf %152, %326 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %328 = arith.mulf %153, %324 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %329 = arith.addf %328, %322 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %330 = arith.truncf %321 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %331 = ttg.convert_layout %330 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %332 = ttg.local_alloc %311 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %333 = ttg.local_load %332 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %308, ^bb16, ^bb17(%327 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb16:  // pred: ^bb15
    %334 = tt.dot %331, %333, %327, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb17(%334 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb17(%335: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb15, ^bb16
    %336 = arith.select %308, %335, %152 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %337 = arith.select %308, %329, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %338 = arith.select %308, %317, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    cf.br ^bb18(%338, %337, %336, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, i32)
  ^bb18(%339: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %340: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %341: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, %342: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %343 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %343, ^bb19, ^bb31(%339, %340, %341 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb19:  // pred: ^bb18
    %344 = arith.subi %24, %28 : i32
    %345 = tt.splat %344 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %346 = arith.addi %14, %345 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %347 = arith.muli %138, %c64_i32 : i32
    %348 = arith.muli %347, %arg10 : i32
    %349 = tt.splat %348 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %350 = tt.addptr %106, %349 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %351 = arith.muli %347, %arg13 : i32
    %352 = tt.splat %351 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %353 = tt.addptr %120, %352 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %354 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %355 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %356 = tt.splat %28 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %357 = arith.cmpi ne, %69, %c0_i32 : i32
    %358 = tt.broadcast %76 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %359 = arith.muli %arg10, %c64_i32 : i32
    %360 = tt.splat %359 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %361 = arith.muli %arg13, %c64_i32 : i32
    %362 = tt.splat %361 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %363 = arith.cmpi slt, %342, %139 : i32
    %364 = tt.splat %342 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %365 = arith.addi %364, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %366 = tt.expand_dims %365 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %367 = arith.cmpi slt, %366, %354 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %368 = tt.broadcast %367 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %369 = tt.splat %363 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %370 = arith.andi %369, %368 : tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %371 = tt.load %350, %370, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %372 = ttg.local_alloc %371 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %373 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%342, %341, %340, %339, %350, %353, %372 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb20(%374: i32, %375: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, %376: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %377: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %378: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %379: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %380: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb19, ^bb23
    %381 = arith.cmpi slt, %374, %373 : i32
    cf.cond_br %381, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %382 = tt.addptr %378, %360 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %383 = arith.addi %374, %c64_i32 : i32
    %384 = tt.splat %383 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %385 = tt.splat %374 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %386 = tt.splat %374 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %387 = amdgpu.extract_slice %384 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %388 = amdgpu.extract_slice %13 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %389 = arith.addi %387, %388 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %390 = amdgpu.extract_slice %384 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %391 = amdgpu.extract_slice %13 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %392 = arith.addi %390, %391 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %393 = amdgpu.extract_slice %384 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %394 = amdgpu.extract_slice %13 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %395 = arith.addi %393, %394 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %396 = amdgpu.extract_slice %384 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %397 = amdgpu.extract_slice %13 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %398 = arith.addi %396, %397 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %399 = amdgpu.concat %389, %392, %395, %398 [4] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %400 = amdgpu.extract_slice %386 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %401 = amdgpu.extract_slice %15 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %402 = arith.addi %400, %401 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %403 = amdgpu.extract_slice %386 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %404 = amdgpu.extract_slice %15 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %405 = arith.addi %403, %404 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %406 = amdgpu.extract_slice %386 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %407 = amdgpu.extract_slice %15 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %408 = arith.addi %406, %407 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %409 = amdgpu.extract_slice %386 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %410 = amdgpu.extract_slice %15 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %411 = arith.addi %409, %410 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %412 = amdgpu.concat %402, %405, %408, %411 [4] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %413 = tt.expand_dims %399 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %414 = arith.cmpi slt, %413, %354 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %415 = tt.broadcast %414 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %416 = tt.load %382, %415, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %417 = ttg.local_load %380 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %418 = tt.expand_dims %412 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %419 = arith.cmpi slt, %418, %356 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %420 = tt.broadcast %419 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %421 = tt.load %379, %420, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %422 = arith.cmpi eq, %383, %139 : i32
    %423 = arith.andi %422, %357 : i1
    cf.cond_br %423, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb22:  // pred: ^bb21
    %424 = tt.splat %374 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %425 = arith.addi %424, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %426 = arith.cmpi slt, %425, %355 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %427 = arith.select %426, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %428 = tt.broadcast %427 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb23(%428 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb23(%429: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb21, ^bb22
    %430 = arith.addi %385, %346 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %431 = tt.expand_dims %430 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %432 = tt.broadcast %431 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %433 = arith.cmpi sge, %358, %432 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %434 = arith.select %433, %429, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %435 = tt.dot %129, %417, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %436 = arith.mulf %435, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %437 = arith.addf %434, %436 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %438 = "tt.reduce"(%437) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %439 = arith.maxnumf %377, %438 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %440 = tt.expand_dims %439 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %441 = tt.broadcast %440 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %442 = arith.subf %437, %441 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %443 = math.exp2 %442 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %444 = "tt.reduce"(%443) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %445 = arith.subf %377, %439 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %446 = math.exp2 %445 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %447 = tt.expand_dims %446 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %448 = tt.broadcast %447 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %449 = arith.mulf %375, %448 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %450 = arith.mulf %376, %446 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %451 = arith.addf %450, %444 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %452 = arith.truncf %443 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %453 = ttg.convert_layout %452 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %454 = ttg.local_alloc %421 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %455 = ttg.local_load %454 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    %456 = tt.dot %453, %455, %449, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %457 = tt.addptr %379, %362 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %458 = ttg.local_alloc %416 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %459 = arith.addi %374, %c64_i32 : i32
    cf.br ^bb20(%459, %456, %451, %439, %382, %457, %458 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb24:  // pred: ^bb20
    %460 = arith.subi %139, %342 : i32
    %461 = arith.addi %460, %c63_i32 : i32
    %462 = arith.divsi %461, %c64_i32 : i32
    %463 = arith.subi %462, %c1_i32 : i32
    %464 = arith.maxsi %463, %c0_i32 : i32
    %465 = arith.muli %464, %c64_i32 : i32
    %466 = arith.addi %342, %465 : i32
    %467 = arith.cmpi sge, %462, %c1_i32 : i32
    %468 = tt.splat %466 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %469 = tt.splat %466 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %470 = arith.addi %469, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    gpu.barrier
    %471 = ttg.local_load %380 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %472 = tt.expand_dims %470 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %473 = arith.cmpi slt, %472, %356 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %474 = tt.broadcast %473 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %475 = tt.splat %467 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %476 = arith.andi %475, %474 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %477 = tt.load %379, %476, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %478 = arith.addi %466, %c64_i32 : i32
    %479 = arith.cmpi eq, %478, %139 : i32
    %480 = arith.andi %479, %357 : i1
    cf.cond_br %480, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb25:  // pred: ^bb24
    %481 = tt.splat %466 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %482 = arith.addi %481, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %483 = arith.cmpi slt, %482, %355 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %484 = arith.select %483, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %485 = tt.broadcast %484 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb26(%485 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb26(%486: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb24, ^bb25
    %487 = arith.addi %468, %346 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %488 = tt.expand_dims %487 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %489 = tt.broadcast %488 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %490 = arith.cmpi sge, %358, %489 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %491 = arith.select %490, %486, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.cond_br %467, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb27:  // pred: ^bb26
    %492 = tt.dot %129, %471, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb28(%492 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb28(%493: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb26, ^bb27
    %494 = arith.mulf %493, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %495 = arith.addf %491, %494 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %496 = "tt.reduce"(%495) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %497 = arith.maxnumf %377, %496 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %498 = tt.expand_dims %497 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %499 = tt.broadcast %498 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %500 = arith.subf %495, %499 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %501 = math.exp2 %500 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %502 = "tt.reduce"(%501) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %503 = arith.subf %377, %497 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %504 = math.exp2 %503 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %505 = tt.expand_dims %504 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %506 = tt.broadcast %505 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %507 = arith.mulf %375, %506 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %508 = arith.mulf %376, %504 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %509 = arith.addf %508, %502 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %510 = arith.truncf %501 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %511 = ttg.convert_layout %510 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %512 = ttg.local_alloc %477 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %513 = ttg.local_load %512 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %467, ^bb29, ^bb30(%507 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb29:  // pred: ^bb28
    %514 = tt.dot %511, %513, %507, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb30(%514 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb30(%515: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb28, ^bb29
    %516 = arith.select %467, %515, %375 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %517 = arith.select %467, %509, %376 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %518 = arith.select %467, %497, %377 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    cf.br ^bb31(%518, %517, %516 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb31(%519: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %520: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %521: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb18, ^bb30
    %522 = tt.expand_dims %520 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %523 = arith.divf %cst_6, %522 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %524 = tt.broadcast %523 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %525 = arith.mulf %521, %524 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %526 = arith.subi %24, %28 : i32
    %527 = arith.truncf %525 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %528 = arith.cmpi sgt, %526, %3 : i32
    %529 = arith.cmpi slt, %526, %32 : i32
    %530 = arith.andi %528, %529 : i1
    cf.cond_br %530, ^bb32, ^bb33(%527 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb32:  // pred: ^bb31
    %531 = tt.splat %526 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %532 = arith.cmpi sge, %76, %531 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %533 = tt.broadcast %532 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %534 = arith.select %533, %527, %cst_5 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb33(%534 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb33(%535: tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb31, ^bb32
    %536 = arith.muli %2, %c54848_i32 : i32
    %537 = tt.addptr %arg3, %536 : !tt.ptr<f32>, i32
    %538 = arith.muli %1, %c3428_i32 : i32
    %539 = tt.addptr %537, %538 : !tt.ptr<f32>, i32
    %540 = tt.splat %539 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %541 = tt.addptr %540, %12 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %542 = arith.subi %32, %24 : i32
    %543 = arith.cmpi sgt, %542, %c0_i32 : i32
    cf.cond_br %543, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %544 = arith.subi %c128_i32, %542 : i32
    %545 = tt.splat %544 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %546 = arith.cmpi slt, %6, %545 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %547 = math.log2 %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %548 = arith.addf %519, %547 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    gpu.barrier
    %549 = ttg.convert_layout %548 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %541, %549, %546 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %550 = math.log2 %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %551 = arith.addf %519, %550 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    gpu.barrier
    %552 = ttg.convert_layout %551 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %541, %552 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %553 = arith.muli %2, %arg14 : i32
    %554 = tt.addptr %arg4, %553 : !tt.ptr<f16>, i32
    %555 = arith.muli %1, %arg15 : i32
    %556 = tt.addptr %554, %555 : !tt.ptr<f16>, i32
    %557 = arith.muli %22, %arg16 : i32
    %558 = tt.addptr %556, %557 : !tt.ptr<f16>, i32
    %559 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %560 = arith.muli %76, %559 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %561 = tt.splat %558 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %562 = tt.addptr %561, %560 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %563 = tt.broadcast %562 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %564 = tt.addptr %563, %87 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %565 = arith.select %543, %125, %cst_14 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    tt.store %564, %535, %565 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb37:  // pred: ^bb1
    tt.return
  }
}
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (0)
 - warp=1 -> (32)
   warp=2 -> (64)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
sizePerThread: 1
basesPerDim: 1
	numElementsPerThread: 1
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
	numElementsPerThread: 4
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
sizePerThread: 8x1
basesPerDim: 8x4
	numElementsPerThread: 8
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 64x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
 - lane=1 -> (0, 8)
   lane=2 -> (0, 16)
   lane=4 -> (0, 32)
   lane=8 -> (0, 64)
   lane=16 -> (1, 0)
   lane=32 -> (2, 0)
 - warp=1 -> (4, 0)
   warp=2 -> (8, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
sizePerThread: 1x8
basesPerDim: 4x8
	numElementsPerThread: 8
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
	numElementsPerThread: 4
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
	numElementsPerThread: 4
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
	numElementsPerThread: 4
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
packLLElements()
tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
packLLElements()
tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x32
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
packLLElements()
tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
packLLElements()
tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f16xf16xf16xf16
resultVals: %58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x1
ll: 
 - register=1 -> (0, 0)
   register=2 -> (0, 0)
   register=4 -> (0, 0)
   register=8 -> (0, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 0)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 1)]
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x1
basesPerDim: 16x1
	numElementsPerThread: 1
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x1
ll: 
 - register=1 -> (0, 0)
   register=2 -> (0, 0)
   register=4 -> (0, 0)
   register=8 -> (0, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 0)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 1)]
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x1
basesPerDim: 16x1
packLLElements()
tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: f32
resultVals: %66 = "llvm.bitcast"(%65) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (0)
 - warp=1 -> (32)
   warp=2 -> (64)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
sizePerThread: 1
basesPerDim: 1
packLLElements()
tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
elementTypes: f32
resultVals: %71 = "llvm.bitcast"(%70) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (0)
 - warp=1 -> (32)
   warp=2 -> (64)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
sizePerThread: 1
basesPerDim: 1
packLLElements()
tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
elementTypes: f32
resultVals: %77 = "llvm.bitcast"(%76) : (f32) -> f32
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 64x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
 - lane=1 -> (0, 8)
   lane=2 -> (0, 16)
   lane=4 -> (0, 32)
   lane=8 -> (0, 64)
   lane=16 -> (1, 0)
   lane=32 -> (2, 0)
 - warp=1 -> (4, 0)
   warp=2 -> (8, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
sizePerThread: 1x8
basesPerDim: 4x8
	numElementsPerThread: 8
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 64x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
 - lane=1 -> (0, 8)
   lane=2 -> (0, 16)
   lane=4 -> (0, 32)
   lane=8 -> (0, 64)
   lane=16 -> (1, 0)
   lane=32 -> (2, 0)
 - warp=1 -> (4, 0)
   warp=2 -> (8, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
sizePerThread: 1x8
basesPerDim: 4x8
packLLElements()
tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
elementTypes: f16xf16xf16xf16xf16xf16xf16xf16
resultVals: %87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16x%87 = "llvm.bitcast"(%86) : (f16) -> f16
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
sizePerThread: 8x1
basesPerDim: 8x4
	numElementsPerThread: 8
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
sizePerThread: 8x1
basesPerDim: 8x4
packLLElements()
tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
elementTypes: f16xf16xf16xf16xf16xf16xf16xf16
resultVals: %99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16x%99 = "llvm.bitcast"(%98) : (f16) -> f16
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
   register=32 -> (64, 0)
 - lane=1 -> (0, 8)
   lane=2 -> (0, 16)
   lane=4 -> (0, 32)
   lane=8 -> (0, 64)
   lane=16 -> (1, 0)
   lane=32 -> (2, 0)
 - warp=1 -> (4, 0)
   warp=2 -> (8, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
sizePerThread: 1x8
basesPerDim: 8x8
	numElementsPerThread: 8
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
   register=32 -> (64, 0)
 - lane=1 -> (0, 8)
   lane=2 -> (0, 16)
   lane=4 -> (0, 32)
   lane=8 -> (0, 64)
   lane=16 -> (1, 0)
   lane=32 -> (2, 0)
 - warp=1 -> (4, 0)
   warp=2 -> (8, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
sizePerThread: 1x8
basesPerDim: 8x8
packLLElements()
tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
elementTypes: f16xf16xf16xf16xf16xf16xf16xf16
resultVals: %111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16x%111 = "llvm.bitcast"(%110) : (f16) -> f16
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (32)
 - warp=1 -> (64)
   warp=2 -> (0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
sizePerThread: 1
basesPerDim: 1
	numElementsPerThread: 1
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (32)
 - warp=1 -> (64)
   warp=2 -> (0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
sizePerThread: 1
basesPerDim: 1
packLLElements()
tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
elementTypes: f32
resultVals: %133 = "llvm.bitcast"(%132) : (f32) -> f32
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (32)
 - warp=1 -> (64)
   warp=2 -> (0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
sizePerThread: 1
basesPerDim: 1
	numElementsPerThread: 1
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (32)
 - warp=1 -> (64)
   warp=2 -> (0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
sizePerThread: 1
basesPerDim: 1
packLLElements()
tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
elementTypes: i32
resultVals: %138 = "llvm.bitcast"(%137) : (i32) -> i32
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
	numElementsPerThread: 4
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 8)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x64
packLLElements()
tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
elementTypes: i1xi1xi1xi1
resultVals: %147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (0)
 - warp=1 -> (32)
   warp=2 -> (64)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
sizePerThread: 1
basesPerDim: 1
	numElementsPerThread: 1
packLLElements()
tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
elementTypes: i32
resultVals: %224 = "llvm.add"(%223, %168) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register=1 -> (16)
   register=2 -> (32)
   register=4 -> (64)
 - lane=1 -> (0)
   lane=2 -> (0)
   lane=4 -> (0)
   lane=8 -> (0)
   lane=16 -> (1)
   lane=32 -> (2)
 - warp=1 -> (4)
   warp=2 -> (8)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
sizePerThread: 1
basesPerDim: 8
	numElementsPerThread: 1
packLLElements()
tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
elementTypes: i32
resultVals: %291 = "llvm.add"(%276, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%292 = "llvm.add"(%278, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%293 = "llvm.add"(%280, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%294 = "llvm.add"(%282, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%295 = "llvm.add"(%284, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%296 = "llvm.add"(%286, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%297 = "llvm.add"(%288, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%298 = "llvm.add"(%290, %230) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:569:50: error:  size mismatch when packing elements for LLVM struct expected 1 but got 8
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                                                 ^
python3: /home/dtanner/.triton/llvm/llvm-2619c2ed-ubuntu-x64/include/llvm/ADT/ArrayRef.h:260: const T &llvm::ArrayRef<mlir::Type>::operator[](size_t) const [T = mlir::Type]: Assertion `Index < Length && "Invalid index!"' failed.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: i32, %arg27: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg28: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1x64xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #mma>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_3 = arith.constant dense<0.127517432> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #mma>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #mma>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_8 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %c63_i32 = arith.constant 63 : i32
    %c54848_i32 = arith.constant 54848 : i32
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c3428_i32 = arith.constant 3428 : i32
    %cst_12 = arith.constant dense<0x7F800000> : tensor<128xf32, #blocked2>
    %cst_13 = arith.constant dense<3428> : tensor<128xi32, #blocked2>
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_14 = arith.constant dense<true> : tensor<128x128xi1, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #blocked2>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #blocked2>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.addptr %arg23, %2 : !tt.ptr<i32>, i32
    %17 = tt.addptr %16, %c1_i32 : !tt.ptr<i32>, i32
    %18 = tt.addptr %arg24, %2 : !tt.ptr<i32>, i32
    %19 = tt.addptr %18, %c1_i32 : !tt.ptr<i32>, i32
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%20: i32):  // 4 preds: ^bb0, ^bb2, ^bb5, ^bb36
    %21 = arith.cmpi slt, %20, %c1_i32 : i32
    cf.cond_br %21, ^bb2, ^bb37
  ^bb2:  // pred: ^bb1
    %22 = tt.load %16 : !tt.ptr<i32>
    %23 = tt.load %17 : !tt.ptr<i32>
    %24 = arith.subi %23, %22 : i32
    %25 = arith.cmpi sle, %3, %24 : i32
    %26 = tt.load %18 : !tt.ptr<i32>
    %27 = tt.load %19 : !tt.ptr<i32>
    %28 = arith.subi %27, %26 : i32
    cf.cond_br %25, ^bb3, ^bb1(%c1_i32 : i32)
  ^bb3:  // pred: ^bb2
    %29 = arith.addi %28, %c63_i32 : i32
    %30 = arith.divsi %29, %c64_i32 : i32
    %31 = arith.addi %0, %c1_i32 : i32
    %32 = arith.muli %31, %c128_i32 : i32
    %33 = arith.addi %32, %28 : i32
    %34 = arith.subi %33, %24 : i32
    %35 = arith.addi %34, %c63_i32 : i32
    %36 = arith.divsi %35, %c64_i32 : i32
    %37 = arith.minsi %30, %36 : i32
    %38 = arith.cmpi sle, %37, %c0_i32 : i32
    %39 = arith.cmpi sgt, %37, %c0_i32 : i32
    cf.cond_br %38, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %40 = arith.muli %2, %arg14 : i32
    %41 = tt.addptr %arg4, %40 : !tt.ptr<f16>, i32
    %42 = arith.muli %1, %arg15 : i32
    %43 = tt.addptr %41, %42 : !tt.ptr<f16>, i32
    %44 = arith.muli %22, %arg16 : i32
    %45 = tt.addptr %43, %44 : !tt.ptr<f16>, i32
    %46 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %47 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked>
    %48 = arith.muli %46, %47 : tensor<128x1xi32, #blocked>
    %49 = tt.splat %45 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %50 = tt.addptr %49, %48 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %51 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %53 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %54 = tt.broadcast %52 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %55 = tt.addptr %53, %54 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %56 = tt.splat %24 : i32 -> tensor<128x1xi32, #blocked>
    %57 = arith.cmpi slt, %46, %56 : tensor<128x1xi32, #blocked>
    %58 = tt.broadcast %57 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    tt.store %55, %cst_11, %58 : tensor<128x128x!tt.ptr<f16>, #blocked>
    %59 = arith.muli %2, %c54848_i32 : i32
    %60 = tt.addptr %arg3, %59 : !tt.ptr<f32>, i32
    %61 = arith.muli %1, %c3428_i32 : i32
    %62 = tt.addptr %60, %61 : !tt.ptr<f32>, i32
    %63 = tt.splat %62 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %64 = tt.addptr %63, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
    %65 = arith.cmpi slt, %12, %cst_13 : tensor<128xi32, #blocked2>
    tt.store %64, %cst_12, %65 : tensor<128x!tt.ptr<f32>, #blocked2>
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    cf.cond_br %39, ^bb6, ^bb1(%c1_i32 : i32)
  ^bb6:  // pred: ^bb5
    %66 = arith.cmpi slt, %28, %c64_i32 : i32
    cf.cond_br %66, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %67 = arith.subi %c64_i32, %28 : i32
    cf.br ^bb9(%67 : i32)
  ^bb8:  // pred: ^bb6
    %68 = arith.remsi %28, %c64_i32 : i32
    cf.br ^bb9(%68 : i32)
  ^bb9(%69: i32):  // 2 preds: ^bb7, ^bb8
    %70 = arith.muli %2, %arg5 : i32
    %71 = tt.addptr %arg0, %70 : !tt.ptr<f16>, i32
    %72 = arith.muli %1, %arg6 : i32
    %73 = tt.addptr %71, %72 : !tt.ptr<f16>, i32
    %74 = arith.muli %22, %arg7 : i32
    %75 = tt.addptr %73, %74 : !tt.ptr<f16>, i32
    %76 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %77 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %78 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
    %79 = arith.muli %77, %78 : tensor<128x1xi32, #blocked>
    %80 = tt.splat %75 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %81 = tt.addptr %80, %79 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %82 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %84 = tt.expand_dims %82 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %85 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %86 = tt.broadcast %81 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %87 = tt.broadcast %84 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %88 = tt.broadcast %85 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %89 = tt.addptr %86, %88 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %90 = arith.muli %2, %arg8 : i32
    %91 = tt.addptr %arg1, %90 : !tt.ptr<f16>, i32
    %92 = arith.muli %1, %arg9 : i32
    %93 = tt.addptr %91, %92 : !tt.ptr<f16>, i32
    %94 = arith.muli %26, %arg10 : i32
    %95 = tt.addptr %93, %94 : !tt.ptr<f16>, i32
    %96 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %97 = tt.expand_dims %96 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %98 = tt.splat %95 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %99 = tt.addptr %98, %97 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %100 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %101 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %102 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked1>
    %103 = arith.muli %100, %102 : tensor<1x64xi32, #blocked1>
    %104 = tt.broadcast %99 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %105 = tt.broadcast %103 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %106 = tt.addptr %104, %105 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %107 = arith.muli %2, %arg11 : i32
    %108 = tt.addptr %arg2, %107 : !tt.ptr<f16>, i32
    %109 = arith.muli %1, %arg12 : i32
    %110 = tt.addptr %108, %109 : !tt.ptr<f16>, i32
    %111 = arith.muli %26, %arg13 : i32
    %112 = tt.addptr %110, %111 : !tt.ptr<f16>, i32
    %113 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %114 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked>
    %115 = arith.muli %113, %114 : tensor<64x1xi32, #blocked>
    %116 = tt.splat %112 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %117 = tt.addptr %116, %115 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %118 = tt.broadcast %117 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %119 = tt.broadcast %85 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %120 = tt.addptr %118, %119 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %121 = tt.splat %24 : i32 -> tensor<128x1xi32, #mma>
    %122 = tt.splat %24 : i32 -> tensor<128x1xi32, #blocked>
    %123 = arith.cmpi slt, %76, %121 : tensor<128x1xi32, #mma>
    %124 = arith.cmpi slt, %77, %122 : tensor<128x1xi32, #blocked>
    %125 = tt.broadcast %123 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %126 = tt.broadcast %124 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %127 = tt.load %89, %126, %cst_11 : tensor<128x128x!tt.ptr<f16>, #blocked>
    gpu.barrier
    %128 = ttg.local_alloc %127 {allocation.offset = 0 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    gpu.barrier
    %129 = ttg.local_load %128 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %130 = arith.cmpi eq, %69, %c0_i32 : i32
    %131 = arith.remsi %24, %c128_i32 : i32
    %132 = arith.cmpi eq, %131, %c0_i32 : i32
    %133 = arith.andi %130, %132 : i1
    %134 = arith.xori %133, %true : i1
    %135 = arith.extui %134 : i1 to i32
    %136 = arith.addi %135, %c2_i32 : i32
    %137 = arith.minsi %136, %37 : i32
    %138 = arith.subi %37, %137 : i32
    %139 = arith.muli %37, %c64_i32 : i32
    %140 = arith.cmpi sgt, %138, %c0_i32 : i32
    cf.cond_br %140, ^bb10, ^bb18(%cst_8, %cst_7, %cst_4, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32)
  ^bb10:  // pred: ^bb9
    %141 = arith.muli %138, %c64_i32 : i32
    %142 = arith.muli %arg10, %c64_i32 : i32
    %143 = tt.splat %142 : i32 -> tensor<128x64xi32, #blocked1>
    %144 = arith.muli %arg13, %c64_i32 : i32
    %145 = tt.splat %144 : i32 -> tensor<64x128xi32, #blocked>
    %146 = arith.cmpi sgt, %141, %c0_i32 : i32
    %147 = tt.splat %146 : i1 -> tensor<128x64xi1, #blocked1>
    %148 = tt.load %106, %147 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    gpu.barrier
    %149 = ttg.local_alloc %148 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %150 = arith.subi %141, %c64_i32 : i32
    cf.br ^bb11(%c0_i32, %cst_4, %cst_7, %cst_8, %106, %120, %149 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb11(%151: i32, %152: tensor<128x128xf32, #mma>, %153: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %154: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %155: tensor<128x64x!tt.ptr<f16>, #blocked1>, %156: tensor<64x128x!tt.ptr<f16>, #blocked>, %157: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>):  // 2 preds: ^bb10, ^bb12
    %158 = arith.cmpi slt, %151, %150 : i32
    cf.cond_br %158, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %159 = tt.addptr %155, %143 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %160 = tt.load %159 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    gpu.barrier
    %161 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %162 = tt.load %156 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %163 = tt.dot %129, %161, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    %164 = amdgpu.extract_slice %163 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %165 = amdgpu.extract_slice %cst_3 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %166 = arith.mulf %164, %165 : tensor<128x8xf32, #mma>
    %167 = amdgpu.extract_slice %163 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %168 = amdgpu.extract_slice %cst_3 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %169 = arith.mulf %167, %168 : tensor<128x8xf32, #mma>
    %170 = amdgpu.extract_slice %163 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %171 = amdgpu.extract_slice %cst_3 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %172 = arith.mulf %170, %171 : tensor<128x8xf32, #mma>
    %173 = amdgpu.extract_slice %163 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %174 = amdgpu.extract_slice %cst_3 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %175 = arith.mulf %173, %174 : tensor<128x8xf32, #mma>
    %176 = amdgpu.extract_slice %163 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %177 = amdgpu.extract_slice %cst_3 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %178 = arith.mulf %176, %177 : tensor<128x8xf32, #mma>
    %179 = amdgpu.extract_slice %163 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %180 = amdgpu.extract_slice %cst_3 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %181 = arith.mulf %179, %180 : tensor<128x8xf32, #mma>
    %182 = amdgpu.extract_slice %163 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %183 = amdgpu.extract_slice %cst_3 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %184 = arith.mulf %182, %183 : tensor<128x8xf32, #mma>
    %185 = amdgpu.extract_slice %163 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %186 = amdgpu.extract_slice %cst_3 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %187 = arith.mulf %185, %186 : tensor<128x8xf32, #mma>
    %188 = amdgpu.extract_slice %cst_2 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %189 = arith.addf %166, %188 : tensor<128x8xf32, #mma>
    %190 = amdgpu.extract_slice %cst_2 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %191 = arith.addf %169, %190 : tensor<128x8xf32, #mma>
    %192 = amdgpu.extract_slice %cst_2 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %193 = arith.addf %172, %192 : tensor<128x8xf32, #mma>
    %194 = amdgpu.extract_slice %cst_2 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %195 = arith.addf %175, %194 : tensor<128x8xf32, #mma>
    %196 = amdgpu.extract_slice %cst_2 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %197 = arith.addf %178, %196 : tensor<128x8xf32, #mma>
    %198 = amdgpu.extract_slice %cst_2 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %199 = arith.addf %181, %198 : tensor<128x8xf32, #mma>
    %200 = amdgpu.extract_slice %cst_2 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %201 = arith.addf %184, %200 : tensor<128x8xf32, #mma>
    %202 = amdgpu.extract_slice %cst_2 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %203 = arith.addf %187, %202 : tensor<128x8xf32, #mma>
    %204 = amdgpu.concat %189, %191, %193, %195, %197, %199, %201, %203 [1, 8] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x64xf32, #mma>
    %205 = "tt.reduce"(%204) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %206 = arith.maxnumf %154, %205 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %207 = tt.expand_dims %206 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %208 = tt.broadcast %207 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %209 = amdgpu.extract_slice %208 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %210 = arith.subf %189, %209 : tensor<128x8xf32, #mma>
    %211 = amdgpu.extract_slice %208 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %212 = arith.subf %191, %211 : tensor<128x8xf32, #mma>
    %213 = amdgpu.extract_slice %208 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %214 = arith.subf %193, %213 : tensor<128x8xf32, #mma>
    %215 = amdgpu.extract_slice %208 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %216 = arith.subf %195, %215 : tensor<128x8xf32, #mma>
    %217 = amdgpu.extract_slice %208 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %218 = arith.subf %197, %217 : tensor<128x8xf32, #mma>
    %219 = amdgpu.extract_slice %208 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %220 = arith.subf %199, %219 : tensor<128x8xf32, #mma>
    %221 = amdgpu.extract_slice %208 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %222 = arith.subf %201, %221 : tensor<128x8xf32, #mma>
    %223 = amdgpu.extract_slice %208 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %224 = arith.subf %203, %223 : tensor<128x8xf32, #mma>
    %225 = math.exp2 %210 : tensor<128x8xf32, #mma>
    %226 = math.exp2 %212 : tensor<128x8xf32, #mma>
    %227 = math.exp2 %214 : tensor<128x8xf32, #mma>
    %228 = math.exp2 %216 : tensor<128x8xf32, #mma>
    %229 = math.exp2 %218 : tensor<128x8xf32, #mma>
    %230 = math.exp2 %220 : tensor<128x8xf32, #mma>
    %231 = math.exp2 %222 : tensor<128x8xf32, #mma>
    %232 = math.exp2 %224 : tensor<128x8xf32, #mma>
    %233 = amdgpu.concat %225, %226, %227, %228, %229, %230, %231, %232 [1, 8] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x64xf32, #mma>
    %234 = "tt.reduce"(%233) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %235 = arith.subf %154, %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %236 = math.exp2 %235 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %237 = tt.expand_dims %236 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %238 = tt.broadcast %237 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %239 = amdgpu.extract_slice %152 [0, 0] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %240 = amdgpu.extract_slice %238 [0, 0] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %241 = arith.mulf %239, %240 : tensor<128x8xf32, #mma>
    %242 = amdgpu.extract_slice %152 [0, 8] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %243 = amdgpu.extract_slice %238 [0, 8] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %244 = arith.mulf %242, %243 : tensor<128x8xf32, #mma>
    %245 = amdgpu.extract_slice %152 [0, 16] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %246 = amdgpu.extract_slice %238 [0, 16] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %247 = arith.mulf %245, %246 : tensor<128x8xf32, #mma>
    %248 = amdgpu.extract_slice %152 [0, 24] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %249 = amdgpu.extract_slice %238 [0, 24] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %250 = arith.mulf %248, %249 : tensor<128x8xf32, #mma>
    %251 = amdgpu.extract_slice %152 [0, 32] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %252 = amdgpu.extract_slice %238 [0, 32] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %253 = arith.mulf %251, %252 : tensor<128x8xf32, #mma>
    %254 = amdgpu.extract_slice %152 [0, 40] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %255 = amdgpu.extract_slice %238 [0, 40] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %256 = arith.mulf %254, %255 : tensor<128x8xf32, #mma>
    %257 = amdgpu.extract_slice %152 [0, 48] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %258 = amdgpu.extract_slice %238 [0, 48] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %259 = arith.mulf %257, %258 : tensor<128x8xf32, #mma>
    %260 = amdgpu.extract_slice %152 [0, 56] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %261 = amdgpu.extract_slice %238 [0, 56] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %262 = arith.mulf %260, %261 : tensor<128x8xf32, #mma>
    %263 = amdgpu.extract_slice %152 [0, 64] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %264 = amdgpu.extract_slice %238 [0, 64] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %265 = arith.mulf %263, %264 : tensor<128x8xf32, #mma>
    %266 = amdgpu.extract_slice %152 [0, 72] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %267 = amdgpu.extract_slice %238 [0, 72] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %268 = arith.mulf %266, %267 : tensor<128x8xf32, #mma>
    %269 = amdgpu.extract_slice %152 [0, 80] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %270 = amdgpu.extract_slice %238 [0, 80] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %271 = arith.mulf %269, %270 : tensor<128x8xf32, #mma>
    %272 = amdgpu.extract_slice %152 [0, 88] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %273 = amdgpu.extract_slice %238 [0, 88] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %274 = arith.mulf %272, %273 : tensor<128x8xf32, #mma>
    %275 = amdgpu.extract_slice %152 [0, 96] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %276 = amdgpu.extract_slice %238 [0, 96] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %277 = arith.mulf %275, %276 : tensor<128x8xf32, #mma>
    %278 = amdgpu.extract_slice %152 [0, 104] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %279 = amdgpu.extract_slice %238 [0, 104] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %280 = arith.mulf %278, %279 : tensor<128x8xf32, #mma>
    %281 = amdgpu.extract_slice %152 [0, 112] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %282 = amdgpu.extract_slice %238 [0, 112] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %283 = arith.mulf %281, %282 : tensor<128x8xf32, #mma>
    %284 = amdgpu.extract_slice %152 [0, 120] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %285 = amdgpu.extract_slice %238 [0, 120] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %286 = arith.mulf %284, %285 : tensor<128x8xf32, #mma>
    %287 = amdgpu.concat %241, %244, %247, %250, %253, %256, %259, %262, %265, %268, %271, %274, %277, %280, %283, %286 [1, 16] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x128xf32, #mma>
    %288 = arith.mulf %153, %236 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %289 = arith.addf %288, %234 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %290 = arith.truncf %225 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %291 = arith.truncf %226 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %292 = arith.truncf %227 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %293 = arith.truncf %228 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %294 = arith.truncf %229 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %295 = arith.truncf %230 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %296 = arith.truncf %231 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %297 = arith.truncf %232 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %298 = amdgpu.concat %290, %291, %292, %293, %294, %295, %296, %297 [1, 8] : tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma>, tensor<128x8xf16, #mma> -> tensor<128x64xf16, #mma>
    %299 = ttg.convert_layout %298 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %300 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %301 = ttg.local_load %300 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %302 = tt.dot %299, %301, %287, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    %303 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    gpu.barrier
    %304 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %305 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%305, %302, %289, %206, %159, %303, %304 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb13:  // pred: ^bb11
    %306 = arith.addi %141, %c63_i32 : i32
    %307 = arith.divsi %306, %c64_i32 : i32
    %308 = arith.cmpi sge, %307, %c1_i32 : i32
    gpu.barrier
    %309 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %310 = tt.splat %308 : i1 -> tensor<64x128xi1, #blocked>
    %311 = tt.load %156, %310 : tensor<64x128x!tt.ptr<f16>, #blocked>
    cf.cond_br %308, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb14:  // pred: ^bb13
    %312 = tt.dot %129, %309, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    cf.br ^bb15(%312 : tensor<128x64xf32, #mma>)
  ^bb15(%313: tensor<128x64xf32, #mma>):  // 2 preds: ^bb13, ^bb14
    %314 = arith.mulf %313, %cst_3 : tensor<128x64xf32, #mma>
    %315 = arith.addf %314, %cst_2 : tensor<128x64xf32, #mma>
    %316 = "tt.reduce"(%315) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %317 = arith.maxnumf %154, %316 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %318 = tt.expand_dims %317 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %319 = tt.broadcast %318 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %320 = arith.subf %315, %319 : tensor<128x64xf32, #mma>
    %321 = math.exp2 %320 : tensor<128x64xf32, #mma>
    %322 = "tt.reduce"(%321) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %323 = arith.subf %154, %317 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %324 = math.exp2 %323 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %325 = tt.expand_dims %324 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %326 = tt.broadcast %325 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %327 = arith.mulf %152, %326 : tensor<128x128xf32, #mma>
    %328 = arith.mulf %153, %324 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %329 = arith.addf %328, %322 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %330 = arith.truncf %321 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %331 = ttg.convert_layout %330 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %332 = ttg.local_alloc %311 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %333 = ttg.local_load %332 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    cf.cond_br %308, ^bb16, ^bb17(%327 : tensor<128x128xf32, #mma>)
  ^bb16:  // pred: ^bb15
    %334 = tt.dot %331, %333, %327, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    cf.br ^bb17(%334 : tensor<128x128xf32, #mma>)
  ^bb17(%335: tensor<128x128xf32, #mma>):  // 2 preds: ^bb15, ^bb16
    %336 = arith.select %308, %335, %152 : tensor<128x128xf32, #mma>
    %337 = arith.select %308, %329, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %338 = arith.select %308, %317, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    cf.br ^bb18(%338, %337, %336, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32)
  ^bb18(%339: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %340: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %341: tensor<128x128xf32, #mma>, %342: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %343 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %343, ^bb19, ^bb31(%339, %340, %341 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>)
  ^bb19:  // pred: ^bb18
    %344 = arith.subi %24, %28 : i32
    %345 = tt.splat %344 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %346 = arith.addi %14, %345 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %347 = arith.muli %138, %c64_i32 : i32
    %348 = arith.muli %347, %arg10 : i32
    %349 = tt.splat %348 : i32 -> tensor<128x64xi32, #blocked1>
    %350 = tt.addptr %106, %349 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %351 = arith.muli %347, %arg13 : i32
    %352 = tt.splat %351 : i32 -> tensor<64x128xi32, #blocked>
    %353 = tt.addptr %120, %352 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %354 = tt.splat %28 : i32 -> tensor<1x64xi32, #blocked1>
    %355 = tt.splat %28 : i32 -> tensor<1x64xi32, #mma>
    %356 = tt.splat %28 : i32 -> tensor<64x1xi32, #blocked>
    %357 = arith.cmpi ne, %69, %c0_i32 : i32
    %358 = tt.broadcast %76 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
    %359 = arith.muli %arg10, %c64_i32 : i32
    %360 = tt.splat %359 : i32 -> tensor<128x64xi32, #blocked1>
    %361 = arith.muli %arg13, %c64_i32 : i32
    %362 = tt.splat %361 : i32 -> tensor<64x128xi32, #blocked>
    %363 = arith.cmpi slt, %342, %139 : i32
    %364 = tt.splat %342 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %365 = arith.addi %364, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %366 = tt.expand_dims %365 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %367 = arith.cmpi slt, %366, %354 : tensor<1x64xi32, #blocked1>
    %368 = tt.broadcast %367 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %369 = tt.splat %363 : i1 -> tensor<128x64xi1, #blocked1>
    %370 = arith.andi %369, %368 : tensor<128x64xi1, #blocked1>
    %371 = tt.load %350, %370, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %372 = ttg.local_alloc %371 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %373 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%342, %341, %340, %339, %350, %353, %372 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb20(%374: i32, %375: tensor<128x128xf32, #mma>, %376: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %377: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %378: tensor<128x64x!tt.ptr<f16>, #blocked1>, %379: tensor<64x128x!tt.ptr<f16>, #blocked>, %380: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>):  // 2 preds: ^bb19, ^bb23
    %381 = arith.cmpi slt, %374, %373 : i32
    cf.cond_br %381, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %382 = tt.addptr %378, %360 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %383 = arith.addi %374, %c64_i32 : i32
    %384 = tt.splat %383 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %385 = tt.splat %374 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %386 = tt.splat %374 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %387 = amdgpu.extract_slice %384 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %388 = amdgpu.extract_slice %13 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %389 = arith.addi %387, %388 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %390 = amdgpu.extract_slice %384 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %391 = amdgpu.extract_slice %13 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %392 = arith.addi %390, %391 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %393 = amdgpu.extract_slice %384 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %394 = amdgpu.extract_slice %13 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %395 = arith.addi %393, %394 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %396 = amdgpu.extract_slice %384 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %397 = amdgpu.extract_slice %13 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %398 = arith.addi %396, %397 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %399 = amdgpu.concat %389, %392, %395, %398 [4] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %400 = amdgpu.extract_slice %386 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %401 = amdgpu.extract_slice %15 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %402 = arith.addi %400, %401 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %403 = amdgpu.extract_slice %386 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %404 = amdgpu.extract_slice %15 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %405 = arith.addi %403, %404 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %406 = amdgpu.extract_slice %386 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %407 = amdgpu.extract_slice %15 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %408 = arith.addi %406, %407 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %409 = amdgpu.extract_slice %386 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %410 = amdgpu.extract_slice %15 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %411 = arith.addi %409, %410 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %412 = amdgpu.concat %402, %405, %408, %411 [4] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %413 = tt.expand_dims %399 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %414 = arith.cmpi slt, %413, %354 : tensor<1x64xi32, #blocked1>
    %415 = tt.broadcast %414 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %416 = tt.load %382, %415, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    gpu.barrier
    %417 = ttg.local_load %380 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %418 = tt.expand_dims %412 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %419 = arith.cmpi slt, %418, %356 : tensor<64x1xi32, #blocked>
    %420 = tt.broadcast %419 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %421 = tt.load %379, %420, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %422 = arith.cmpi eq, %383, %139 : i32
    %423 = arith.andi %422, %357 : i1
    cf.cond_br %423, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb22:  // pred: ^bb21
    %424 = tt.splat %374 : i32 -> tensor<1x64xi32, #mma>
    %425 = arith.addi %424, %101 : tensor<1x64xi32, #mma>
    %426 = arith.cmpi slt, %425, %355 : tensor<1x64xi32, #mma>
    %427 = arith.select %426, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
    %428 = tt.broadcast %427 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
    cf.br ^bb23(%428 : tensor<128x64xf32, #mma>)
  ^bb23(%429: tensor<128x64xf32, #mma>):  // 2 preds: ^bb21, ^bb22
    %430 = arith.addi %385, %346 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %431 = tt.expand_dims %430 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %432 = tt.broadcast %431 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
    %433 = arith.cmpi sge, %358, %432 : tensor<128x64xi32, #mma>
    %434 = arith.select %433, %429, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
    %435 = tt.dot %129, %417, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    %436 = arith.mulf %435, %cst_3 : tensor<128x64xf32, #mma>
    %437 = arith.addf %434, %436 : tensor<128x64xf32, #mma>
    %438 = "tt.reduce"(%437) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %439 = arith.maxnumf %377, %438 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %440 = tt.expand_dims %439 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %441 = tt.broadcast %440 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %442 = arith.subf %437, %441 : tensor<128x64xf32, #mma>
    %443 = math.exp2 %442 : tensor<128x64xf32, #mma>
    %444 = "tt.reduce"(%443) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %445 = arith.subf %377, %439 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %446 = math.exp2 %445 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %447 = tt.expand_dims %446 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %448 = tt.broadcast %447 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %449 = arith.mulf %375, %448 : tensor<128x128xf32, #mma>
    %450 = arith.mulf %376, %446 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %451 = arith.addf %450, %444 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %452 = arith.truncf %443 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %453 = ttg.convert_layout %452 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %454 = ttg.local_alloc %421 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %455 = ttg.local_load %454 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %456 = tt.dot %453, %455, %449, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    %457 = tt.addptr %379, %362 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    gpu.barrier
    %458 = ttg.local_alloc %416 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %459 = arith.addi %374, %c64_i32 : i32
    cf.br ^bb20(%459, %456, %451, %439, %382, %457, %458 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb24:  // pred: ^bb20
    %460 = arith.subi %139, %342 : i32
    %461 = arith.addi %460, %c63_i32 : i32
    %462 = arith.divsi %461, %c64_i32 : i32
    %463 = arith.subi %462, %c1_i32 : i32
    %464 = arith.maxsi %463, %c0_i32 : i32
    %465 = arith.muli %464, %c64_i32 : i32
    %466 = arith.addi %342, %465 : i32
    %467 = arith.cmpi sge, %462, %c1_i32 : i32
    %468 = tt.splat %466 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %469 = tt.splat %466 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %470 = arith.addi %469, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    gpu.barrier
    %471 = ttg.local_load %380 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %472 = tt.expand_dims %470 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %473 = arith.cmpi slt, %472, %356 : tensor<64x1xi32, #blocked>
    %474 = tt.broadcast %473 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %475 = tt.splat %467 : i1 -> tensor<64x128xi1, #blocked>
    %476 = arith.andi %475, %474 : tensor<64x128xi1, #blocked>
    %477 = tt.load %379, %476, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %478 = arith.addi %466, %c64_i32 : i32
    %479 = arith.cmpi eq, %478, %139 : i32
    %480 = arith.andi %479, %357 : i1
    cf.cond_br %480, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb25:  // pred: ^bb24
    %481 = tt.splat %466 : i32 -> tensor<1x64xi32, #mma>
    %482 = arith.addi %481, %101 : tensor<1x64xi32, #mma>
    %483 = arith.cmpi slt, %482, %355 : tensor<1x64xi32, #mma>
    %484 = arith.select %483, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
    %485 = tt.broadcast %484 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
    cf.br ^bb26(%485 : tensor<128x64xf32, #mma>)
  ^bb26(%486: tensor<128x64xf32, #mma>):  // 2 preds: ^bb24, ^bb25
    %487 = arith.addi %468, %346 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %488 = tt.expand_dims %487 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %489 = tt.broadcast %488 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
    %490 = arith.cmpi sge, %358, %489 : tensor<128x64xi32, #mma>
    %491 = arith.select %490, %486, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
    cf.cond_br %467, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb27:  // pred: ^bb26
    %492 = tt.dot %129, %471, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    cf.br ^bb28(%492 : tensor<128x64xf32, #mma>)
  ^bb28(%493: tensor<128x64xf32, #mma>):  // 2 preds: ^bb26, ^bb27
    %494 = arith.mulf %493, %cst_3 : tensor<128x64xf32, #mma>
    %495 = arith.addf %491, %494 : tensor<128x64xf32, #mma>
    %496 = "tt.reduce"(%495) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %497 = arith.maxnumf %377, %496 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %498 = tt.expand_dims %497 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %499 = tt.broadcast %498 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %500 = arith.subf %495, %499 : tensor<128x64xf32, #mma>
    %501 = math.exp2 %500 : tensor<128x64xf32, #mma>
    %502 = "tt.reduce"(%501) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %566 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %566 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %503 = arith.subf %377, %497 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %504 = math.exp2 %503 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %505 = tt.expand_dims %504 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %506 = tt.broadcast %505 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %507 = arith.mulf %375, %506 : tensor<128x128xf32, #mma>
    %508 = arith.mulf %376, %504 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %509 = arith.addf %508, %502 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %510 = arith.truncf %501 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %511 = ttg.convert_layout %510 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %512 = ttg.local_alloc %477 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %513 = ttg.local_load %512 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    cf.cond_br %467, ^bb29, ^bb30(%507 : tensor<128x128xf32, #mma>)
  ^bb29:  // pred: ^bb28
    %514 = tt.dot %511, %513, %507, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    cf.br ^bb30(%514 : tensor<128x128xf32, #mma>)
  ^bb30(%515: tensor<128x128xf32, #mma>):  // 2 preds: ^bb28, ^bb29
    %516 = arith.select %467, %515, %375 : tensor<128x128xf32, #mma>
    %517 = arith.select %467, %509, %376 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %518 = arith.select %467, %497, %377 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    cf.br ^bb31(%518, %517, %516 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>)
  ^bb31(%519: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %520: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %521: tensor<128x128xf32, #mma>):  // 2 preds: ^bb18, ^bb30
    %522 = tt.expand_dims %520 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %523 = arith.divf %cst_6, %522 : tensor<128x1xf32, #mma>
    %524 = tt.broadcast %523 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %525 = arith.mulf %521, %524 : tensor<128x128xf32, #mma>
    %526 = arith.subi %24, %28 : i32
    %527 = arith.truncf %525 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %528 = arith.cmpi sgt, %526, %3 : i32
    %529 = arith.cmpi slt, %526, %32 : i32
    %530 = arith.andi %528, %529 : i1
    cf.cond_br %530, ^bb32, ^bb33(%527 : tensor<128x128xf16, #mma>)
  ^bb32:  // pred: ^bb31
    %531 = tt.splat %526 : i32 -> tensor<128x1xi32, #mma>
    %532 = arith.cmpi sge, %76, %531 : tensor<128x1xi32, #mma>
    %533 = tt.broadcast %532 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %534 = arith.select %533, %527, %cst_5 : tensor<128x128xi1, #mma>, tensor<128x128xf16, #mma>
    cf.br ^bb33(%534 : tensor<128x128xf16, #mma>)
  ^bb33(%535: tensor<128x128xf16, #mma>):  // 2 preds: ^bb31, ^bb32
    %536 = arith.muli %2, %c54848_i32 : i32
    %537 = tt.addptr %arg3, %536 : !tt.ptr<f32>, i32
    %538 = arith.muli %1, %c3428_i32 : i32
    %539 = tt.addptr %537, %538 : !tt.ptr<f32>, i32
    %540 = tt.splat %539 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %541 = tt.addptr %540, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
    %542 = arith.subi %32, %24 : i32
    %543 = arith.cmpi sgt, %542, %c0_i32 : i32
    cf.cond_br %543, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %544 = arith.subi %c128_i32, %542 : i32
    %545 = tt.splat %544 : i32 -> tensor<128xi32, #blocked2>
    %546 = arith.cmpi slt, %6, %545 : tensor<128xi32, #blocked2>
    %547 = math.log2 %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %548 = arith.addf %519, %547 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    gpu.barrier
    %549 = ttg.convert_layout %548 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
    tt.store %541, %549, %546 : tensor<128x!tt.ptr<f32>, #blocked2>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %550 = math.log2 %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %551 = arith.addf %519, %550 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    gpu.barrier
    %552 = ttg.convert_layout %551 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
    tt.store %541, %552 : tensor<128x!tt.ptr<f32>, #blocked2>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %553 = arith.muli %2, %arg14 : i32
    %554 = tt.addptr %arg4, %553 : !tt.ptr<f16>, i32
    %555 = arith.muli %1, %arg15 : i32
    %556 = tt.addptr %554, %555 : !tt.ptr<f16>, i32
    %557 = arith.muli %22, %arg16 : i32
    %558 = tt.addptr %556, %557 : !tt.ptr<f16>, i32
    %559 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
    %560 = arith.muli %76, %559 : tensor<128x1xi32, #mma>
    %561 = tt.splat %558 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %562 = tt.addptr %561, %560 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %563 = tt.broadcast %562 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #mma>
    %564 = tt.addptr %563, %87 : tensor<128x128x!tt.ptr<f16>, #mma>, tensor<128x128xi32, #mma>
    %565 = arith.select %543, %125, %cst_14 : tensor<128x128xi1, #mma>
    tt.store %564, %535, %565 : tensor<128x128x!tt.ptr<f16>, #mma>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb37:  // pred: ^bb1
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, convert-triton-amdgpu-to-llvm{arch=gfx942 ftz=true}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-cf-to-llvm{index-bitwidth=0}, convert-arith-to-llvm{index-bitwidth=0}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, triton-amdgpu-lower-insert-instruction-sched-hints{arch=gfx942 num_stages=2}, enable-line-info, convert-builtin-func-to-llvm{ftz=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:527:0: error: Failures have been detected while processing an MLIR pass pipeline
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:527:0: note: Pipeline failed while executing [`ConvertTritonAMDGPUToLLVM` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
INFO: running a single config given from a file
Traceback (most recent call last):
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 2192, in <module>
    sys.exit(main())
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 2188, in main
    run_benchmark(custom_config, args)
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 2113, in run_benchmark
    bench_flash_attention.run(save_path=".", print_data=True, show_plots=True)
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 388, in run
    result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 335, in _run
    ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 2096, in bench_flash_attention
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 145, in do_bench
    fn()
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 2072, in <lambda>
    fn = lambda: attention(q, k, v, o, input_metadata)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py", line 1221, in forward
    handle = attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, M, o, *q_strides, *k_strides, *v_strides, *o_strides,
  File "/home/dtanner/repos/triton/python/triton/runtime/jit.py", line 368, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/dtanner/repos/triton/python/triton/runtime/autotuner.py", line 206, in run
    ret = self.fn.run(
  File "/home/dtanner/repos/triton/python/triton/runtime/jit.py", line 608, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
  File "/home/dtanner/repos/triton/python/triton/compiler/compiler.py", line 284, in compile
    next_module = compile_ir(module, metadata)
  File "/home/dtanner/repos/triton/python/triton/backends/amd/compiler.py", line 438, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
  File "/home/dtanner/repos/triton/python/triton/backends/amd/compiler.py", line 335, in make_llir
    pm.run(mod)
RuntimeError: PassManager::run failed
