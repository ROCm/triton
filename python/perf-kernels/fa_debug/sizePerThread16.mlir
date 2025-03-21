[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %171 = math.exp2 %170 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %190 = math.exp2 %189 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 64
numReps: 2
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %200 = arith.truncf %187 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %223 = arith.addi %151, %c64_i32 : i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %223 = arith.addi %151, %c64_i32 : i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %460 = arith.maxnumf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %460 = arith.maxnumf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %167 = arith.maxnumf %154, %166 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 64
numReps: 2
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %165 = arith.addf %164, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %490 = arith.addf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %490 = arith.addf %arg29, %arg30 : f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %229 = arith.addf %228, %218 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 64
numReps: 2
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %200 = arith.subf %189, %199 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %243 = arith.subf %154, %197 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 64
numReps: 2
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %164 = arith.mulf %163, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %281 = arith.mulf %152, %280 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x128
srcShapePerCtaTile: 64x16
numReps: 2x8
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %330 = arith.mulf %153, %278 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 64
numReps: 2
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %445 = arith.addi %436, %c64_i32 : i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %445 = arith.addi %436, %c64_i32 : i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %449 = arith.addi %446, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %462 = arith.addi %448, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %485 = arith.andi %484, %419 : i1
[tritonamdgpu-refine-ops]: failed to refine binary op: %485 = arith.andi %484, %419 : i1
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: i32, %arg27: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg28: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_3 = arith.constant dense<0.127517432> : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %cst_8 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
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
    %cst_14 = arith.constant dense<true> : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
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
    %76 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %77 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %78 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %79 = arith.muli %77, %78 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %80 = tt.splat %75 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %81 = tt.addptr %80, %79 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %82 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %84 = tt.expand_dims %82 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %85 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %86 = tt.broadcast %81 : tensor<128x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %87 = tt.broadcast %84 : tensor<1x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
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
    %101 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
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
    %121 = tt.splat %24 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %122 = tt.splat %24 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %123 = arith.cmpi slt, %76, %121 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %124 = arith.cmpi slt, %77, %122 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %125 = tt.broadcast %123 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %126 = tt.broadcast %124 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %127 = tt.load %89, %126, %cst_11 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %128 = ttg.local_alloc %127 {allocation.offset = 0 : i32} : (tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %129 = ttg.local_load %128 : !ttg.memdesc<128x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
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
    cf.cond_br %140, ^bb10, ^bb18(%cst_8, %cst_7, %cst_4, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, i32)
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
    cf.br ^bb11(%c0_i32, %cst_4, %cst_7, %cst_8, %106, %120, %149 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb11(%151: i32, %152: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, %153: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %154: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %155: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %156: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %157: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb10, ^bb12
    %158 = arith.cmpi slt, %151, %150 : i32
    cf.cond_br %158, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %159 = tt.addptr %155, %143 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %160 = tt.load %159 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %161 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %162 = tt.load %156 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %163 = tt.dot %129, %161, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %164 = amdgpu.extract_slice %163 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %165 = amdgpu.extract_slice %cst_3 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %166 = arith.mulf %164, %165 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %167 = amdgpu.extract_slice %163 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %168 = amdgpu.extract_slice %cst_3 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %169 = arith.mulf %167, %168 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %170 = amdgpu.extract_slice %163 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %171 = amdgpu.extract_slice %cst_3 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %172 = arith.mulf %170, %171 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %173 = amdgpu.extract_slice %163 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %174 = amdgpu.extract_slice %cst_3 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %175 = arith.mulf %173, %174 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %176 = amdgpu.extract_slice %163 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %177 = amdgpu.extract_slice %cst_3 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %178 = arith.mulf %176, %177 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %179 = amdgpu.extract_slice %163 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %180 = amdgpu.extract_slice %cst_3 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %181 = arith.mulf %179, %180 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %182 = amdgpu.extract_slice %163 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %183 = amdgpu.extract_slice %cst_3 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %184 = arith.mulf %182, %183 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %185 = amdgpu.extract_slice %163 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %186 = amdgpu.extract_slice %cst_3 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %187 = arith.mulf %185, %186 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %188 = amdgpu.extract_slice %cst_2 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %189 = arith.addf %166, %188 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %190 = amdgpu.extract_slice %cst_2 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %191 = arith.addf %172, %190 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %192 = amdgpu.extract_slice %cst_2 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %193 = arith.addf %178, %192 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %194 = amdgpu.extract_slice %cst_2 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %195 = arith.addf %184, %194 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %196 = amdgpu.extract_slice %cst_2 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %197 = arith.addf %169, %196 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %198 = amdgpu.extract_slice %cst_2 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %199 = arith.addf %175, %198 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %200 = amdgpu.extract_slice %cst_2 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %201 = arith.addf %181, %200 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %202 = amdgpu.extract_slice %cst_2 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %203 = arith.addf %187, %202 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %204 = amdgpu.concat %189, %191, %193, %195, %197, %199, %201, %203 [2, 4] : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %205 = "tt.reduce"(%204) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %206 = amdgpu.extract_slice %154 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %207 = amdgpu.extract_slice %205 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %208 = arith.maxnumf %206, %207 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %209 = amdgpu.extract_slice %154 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %210 = amdgpu.extract_slice %205 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %211 = arith.maxnumf %209, %210 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %212 = amdgpu.concat %208, %211 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %213 = tt.expand_dims %212 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %214 = tt.broadcast %213 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %215 = amdgpu.extract_slice %214 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %216 = arith.subf %189, %215 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %217 = amdgpu.extract_slice %214 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %218 = arith.subf %193, %217 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %219 = amdgpu.extract_slice %214 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %220 = arith.subf %197, %219 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %221 = amdgpu.extract_slice %214 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %222 = arith.subf %201, %221 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %223 = amdgpu.extract_slice %214 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %224 = arith.subf %191, %223 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %225 = amdgpu.extract_slice %214 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %226 = arith.subf %195, %225 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %227 = amdgpu.extract_slice %214 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %228 = arith.subf %199, %227 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %229 = amdgpu.extract_slice %214 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %230 = arith.subf %203, %229 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %231 = math.exp2 %216 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %232 = math.exp2 %220 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %233 = math.exp2 %224 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %234 = math.exp2 %228 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %235 = math.exp2 %218 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %236 = math.exp2 %222 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %237 = math.exp2 %226 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %238 = math.exp2 %230 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %239 = amdgpu.concat %231, %232, %233, %234, %235, %236, %237, %238 [2, 4] : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %240 = "tt.reduce"(%239) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %241 = amdgpu.extract_slice %154 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %242 = arith.subf %241, %208 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %243 = amdgpu.extract_slice %154 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %244 = arith.subf %243, %211 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %245 = math.exp2 %242 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %246 = math.exp2 %244 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %247 = amdgpu.concat %245, %246 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %248 = tt.expand_dims %247 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %249 = tt.broadcast %248 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %250 = amdgpu.extract_slice %152 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %251 = amdgpu.extract_slice %249 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %252 = arith.mulf %250, %251 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %253 = amdgpu.extract_slice %152 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %254 = amdgpu.extract_slice %249 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %255 = arith.mulf %253, %254 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %256 = amdgpu.extract_slice %152 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %257 = amdgpu.extract_slice %249 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %258 = arith.mulf %256, %257 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %259 = amdgpu.extract_slice %152 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %260 = amdgpu.extract_slice %249 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %261 = arith.mulf %259, %260 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %262 = amdgpu.extract_slice %152 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %263 = amdgpu.extract_slice %249 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %264 = arith.mulf %262, %263 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %265 = amdgpu.extract_slice %152 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %266 = amdgpu.extract_slice %249 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %267 = arith.mulf %265, %266 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %268 = amdgpu.extract_slice %152 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %269 = amdgpu.extract_slice %249 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %270 = arith.mulf %268, %269 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %271 = amdgpu.extract_slice %152 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %272 = amdgpu.extract_slice %249 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %273 = arith.mulf %271, %272 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %274 = amdgpu.extract_slice %152 [64, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %275 = amdgpu.extract_slice %249 [64, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %276 = arith.mulf %274, %275 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %277 = amdgpu.extract_slice %152 [64, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %278 = amdgpu.extract_slice %249 [64, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %279 = arith.mulf %277, %278 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %280 = amdgpu.extract_slice %152 [64, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %281 = amdgpu.extract_slice %249 [64, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %282 = arith.mulf %280, %281 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %283 = amdgpu.extract_slice %152 [64, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %284 = amdgpu.extract_slice %249 [64, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %285 = arith.mulf %283, %284 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %286 = amdgpu.extract_slice %152 [64, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %287 = amdgpu.extract_slice %249 [64, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %288 = arith.mulf %286, %287 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %289 = amdgpu.extract_slice %152 [64, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %290 = amdgpu.extract_slice %249 [64, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %291 = arith.mulf %289, %290 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %292 = amdgpu.extract_slice %152 [64, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %293 = amdgpu.extract_slice %249 [64, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %294 = arith.mulf %292, %293 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %295 = amdgpu.extract_slice %152 [64, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %296 = amdgpu.extract_slice %249 [64, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %297 = arith.mulf %295, %296 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %298 = amdgpu.concat %252, %255, %258, %261, %264, %267, %270, %273, %276, %279, %282, %285, %288, %291, %294, %297 [2, 8] : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %299 = amdgpu.extract_slice %153 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %300 = arith.mulf %299, %245 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %301 = amdgpu.extract_slice %153 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %302 = arith.mulf %301, %246 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %303 = amdgpu.extract_slice %240 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %304 = arith.addf %300, %303 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %305 = amdgpu.extract_slice %240 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %306 = arith.addf %302, %305 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %307 = amdgpu.concat %304, %306 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %308 = arith.truncf %231 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %309 = arith.truncf %233 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %310 = arith.truncf %235 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %311 = arith.truncf %237 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %312 = arith.truncf %232 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %313 = arith.truncf %234 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %314 = arith.truncf %236 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %315 = arith.truncf %238 : tensor<64x16xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %316 = amdgpu.concat %308, %309, %310, %311, %312, %313, %314, %315 [2, 4] : tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<64x16xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %317 = ttg.convert_layout %316 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %318 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %319 = ttg.local_load %318 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    %320 = tt.dot %317, %319, %298, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %321 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %322 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %323 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%323, %320, %307, %212, %159, %321, %322 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb13:  // pred: ^bb11
    %324 = arith.addi %141, %c63_i32 : i32
    %325 = arith.divsi %324, %c64_i32 : i32
    %326 = arith.cmpi sge, %325, %c1_i32 : i32
    gpu.barrier
    %327 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %328 = tt.splat %326 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %329 = tt.load %156, %328 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    cf.cond_br %326, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb14:  // pred: ^bb13
    %330 = tt.dot %129, %327, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb15(%330 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb15(%331: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb13, ^bb14
    %332 = arith.mulf %331, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %333 = arith.addf %332, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %334 = "tt.reduce"(%333) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %335 = arith.maxnumf %154, %334 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %336 = tt.expand_dims %335 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %337 = tt.broadcast %336 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %338 = arith.subf %333, %337 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %339 = math.exp2 %338 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %340 = "tt.reduce"(%339) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %341 = arith.subf %154, %335 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %342 = math.exp2 %341 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %343 = tt.expand_dims %342 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %344 = tt.broadcast %343 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %345 = arith.mulf %152, %344 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %346 = arith.mulf %153, %342 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %347 = arith.addf %346, %340 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %348 = arith.truncf %339 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %349 = ttg.convert_layout %348 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %350 = ttg.local_alloc %329 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %351 = ttg.local_load %350 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %326, ^bb16, ^bb17(%345 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb16:  // pred: ^bb15
    %352 = tt.dot %349, %351, %345, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb17(%352 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb17(%353: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb15, ^bb16
    %354 = arith.select %326, %353, %152 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %355 = arith.select %326, %347, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %356 = arith.select %326, %335, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    cf.br ^bb18(%356, %355, %354, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, i32)
  ^bb18(%357: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %358: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %359: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, %360: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %361 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %361, ^bb19, ^bb31(%357, %358, %359 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb19:  // pred: ^bb18
    %362 = arith.subi %24, %28 : i32
    %363 = tt.splat %362 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %364 = arith.addi %14, %363 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %365 = arith.muli %138, %c64_i32 : i32
    %366 = arith.muli %365, %arg10 : i32
    %367 = tt.splat %366 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %368 = tt.addptr %106, %367 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %369 = arith.muli %365, %arg13 : i32
    %370 = tt.splat %369 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %371 = tt.addptr %120, %370 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %372 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %373 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %374 = tt.splat %28 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %375 = arith.cmpi ne, %69, %c0_i32 : i32
    %376 = tt.broadcast %76 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %377 = arith.muli %arg10, %c64_i32 : i32
    %378 = tt.splat %377 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %379 = arith.muli %arg13, %c64_i32 : i32
    %380 = tt.splat %379 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %381 = arith.cmpi slt, %360, %139 : i32
    %382 = tt.splat %360 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %383 = arith.addi %382, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %384 = tt.expand_dims %383 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %385 = arith.cmpi slt, %384, %372 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %386 = tt.broadcast %385 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %387 = tt.splat %381 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %388 = arith.andi %387, %386 : tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %389 = tt.load %368, %388, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %390 = ttg.local_alloc %389 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %391 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%360, %359, %358, %357, %368, %371, %390 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb20(%392: i32, %393: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, %394: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %395: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %396: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %397: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %398: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb19, ^bb23
    %399 = arith.cmpi slt, %392, %391 : i32
    cf.cond_br %399, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %400 = tt.addptr %396, %378 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %401 = arith.addi %392, %c64_i32 : i32
    %402 = tt.splat %401 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %403 = tt.splat %392 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %404 = tt.splat %392 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %405 = amdgpu.extract_slice %402 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %406 = amdgpu.extract_slice %13 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %407 = arith.addi %405, %406 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %408 = amdgpu.extract_slice %402 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %409 = amdgpu.extract_slice %13 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %410 = arith.addi %408, %409 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %411 = amdgpu.extract_slice %402 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %412 = amdgpu.extract_slice %13 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %413 = arith.addi %411, %412 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %414 = amdgpu.extract_slice %402 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %415 = amdgpu.extract_slice %13 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %416 = arith.addi %414, %415 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %417 = amdgpu.concat %407, %410, %413, %416 [4] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %418 = amdgpu.extract_slice %404 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %419 = amdgpu.extract_slice %15 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %420 = arith.addi %418, %419 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %421 = amdgpu.extract_slice %404 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %422 = amdgpu.extract_slice %15 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %423 = arith.addi %421, %422 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %424 = amdgpu.extract_slice %404 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %425 = amdgpu.extract_slice %15 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %426 = arith.addi %424, %425 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %427 = amdgpu.extract_slice %404 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %428 = amdgpu.extract_slice %15 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %429 = arith.addi %427, %428 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %430 = amdgpu.concat %420, %423, %426, %429 [4] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %431 = tt.expand_dims %417 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %432 = arith.cmpi slt, %431, %372 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %433 = tt.broadcast %432 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %434 = tt.load %400, %433, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %435 = ttg.local_load %398 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %436 = tt.expand_dims %430 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %437 = arith.cmpi slt, %436, %374 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %438 = tt.broadcast %437 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %439 = tt.load %397, %438, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %440 = arith.cmpi eq, %401, %139 : i32
    %441 = arith.andi %440, %375 : i1
    cf.cond_br %441, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb22:  // pred: ^bb21
    %442 = tt.splat %392 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %443 = arith.addi %442, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %444 = arith.cmpi slt, %443, %373 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %445 = arith.select %444, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %446 = tt.broadcast %445 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb23(%446 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb23(%447: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb21, ^bb22
    %448 = arith.addi %403, %364 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %449 = tt.expand_dims %448 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %450 = tt.broadcast %449 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %451 = arith.cmpi sge, %376, %450 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %452 = arith.select %451, %447, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %453 = tt.dot %129, %435, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %454 = arith.mulf %453, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %455 = arith.addf %452, %454 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %456 = "tt.reduce"(%455) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %457 = arith.maxnumf %395, %456 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %458 = tt.expand_dims %457 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %459 = tt.broadcast %458 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %460 = arith.subf %455, %459 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %461 = math.exp2 %460 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %462 = "tt.reduce"(%461) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %463 = arith.subf %395, %457 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %464 = math.exp2 %463 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %465 = tt.expand_dims %464 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %466 = tt.broadcast %465 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %467 = arith.mulf %393, %466 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %468 = arith.mulf %394, %464 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %469 = arith.addf %468, %462 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %470 = arith.truncf %461 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %471 = ttg.convert_layout %470 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %472 = ttg.local_alloc %439 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %473 = ttg.local_load %472 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    %474 = tt.dot %471, %473, %467, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %475 = tt.addptr %397, %380 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %476 = ttg.local_alloc %434 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %477 = arith.addi %392, %c64_i32 : i32
    cf.br ^bb20(%477, %474, %469, %457, %400, %475, %476 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb24:  // pred: ^bb20
    %478 = arith.subi %139, %360 : i32
    %479 = arith.addi %478, %c63_i32 : i32
    %480 = arith.divsi %479, %c64_i32 : i32
    %481 = arith.subi %480, %c1_i32 : i32
    %482 = arith.maxsi %481, %c0_i32 : i32
    %483 = arith.muli %482, %c64_i32 : i32
    %484 = arith.addi %360, %483 : i32
    %485 = arith.cmpi sge, %480, %c1_i32 : i32
    %486 = tt.splat %484 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %487 = tt.splat %484 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %488 = arith.addi %487, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    gpu.barrier
    %489 = ttg.local_load %398 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %490 = tt.expand_dims %488 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %491 = arith.cmpi slt, %490, %374 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %492 = tt.broadcast %491 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %493 = tt.splat %485 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %494 = arith.andi %493, %492 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %495 = tt.load %397, %494, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %496 = arith.addi %484, %c64_i32 : i32
    %497 = arith.cmpi eq, %496, %139 : i32
    %498 = arith.andi %497, %375 : i1
    cf.cond_br %498, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb25:  // pred: ^bb24
    %499 = tt.splat %484 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %500 = arith.addi %499, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %501 = arith.cmpi slt, %500, %373 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %502 = arith.select %501, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %503 = tt.broadcast %502 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb26(%503 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb26(%504: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb24, ^bb25
    %505 = arith.addi %486, %364 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %506 = tt.expand_dims %505 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %507 = tt.broadcast %506 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %508 = arith.cmpi sge, %376, %507 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %509 = arith.select %508, %504, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.cond_br %485, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb27:  // pred: ^bb26
    %510 = tt.dot %129, %489, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb28(%510 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb28(%511: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb26, ^bb27
    %512 = arith.mulf %511, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %513 = arith.addf %509, %512 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %514 = "tt.reduce"(%513) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %515 = arith.maxnumf %395, %514 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %516 = tt.expand_dims %515 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %517 = tt.broadcast %516 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %518 = arith.subf %513, %517 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %519 = math.exp2 %518 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %520 = "tt.reduce"(%519) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %521 = arith.subf %395, %515 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %522 = math.exp2 %521 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %523 = tt.expand_dims %522 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %524 = tt.broadcast %523 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %525 = arith.mulf %393, %524 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %526 = arith.mulf %394, %522 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %527 = arith.addf %526, %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %528 = arith.truncf %519 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %529 = ttg.convert_layout %528 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %530 = ttg.local_alloc %495 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %531 = ttg.local_load %530 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %485, ^bb29, ^bb30(%525 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb29:  // pred: ^bb28
    %532 = tt.dot %529, %531, %525, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb30(%532 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb30(%533: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb28, ^bb29
    %534 = arith.select %485, %533, %393 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %535 = arith.select %485, %527, %394 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %536 = arith.select %485, %515, %395 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    cf.br ^bb31(%536, %535, %534 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb31(%537: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %538: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %539: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb18, ^bb30
    %540 = tt.expand_dims %538 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %541 = arith.divf %cst_6, %540 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %542 = tt.broadcast %541 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %543 = arith.mulf %539, %542 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %544 = arith.subi %24, %28 : i32
    %545 = arith.truncf %543 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %546 = arith.cmpi sgt, %544, %3 : i32
    %547 = arith.cmpi slt, %544, %32 : i32
    %548 = arith.andi %546, %547 : i1
    cf.cond_br %548, ^bb32, ^bb33(%545 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb32:  // pred: ^bb31
    %549 = tt.splat %544 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %550 = arith.cmpi sge, %76, %549 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %551 = tt.broadcast %550 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %552 = arith.select %551, %545, %cst_5 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb33(%552 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb33(%553: tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb31, ^bb32
    %554 = arith.muli %2, %c54848_i32 : i32
    %555 = tt.addptr %arg3, %554 : !tt.ptr<f32>, i32
    %556 = arith.muli %1, %c3428_i32 : i32
    %557 = tt.addptr %555, %556 : !tt.ptr<f32>, i32
    %558 = tt.splat %557 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %559 = tt.addptr %558, %12 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %560 = arith.subi %32, %24 : i32
    %561 = arith.cmpi sgt, %560, %c0_i32 : i32
    cf.cond_br %561, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %562 = arith.subi %c128_i32, %560 : i32
    %563 = tt.splat %562 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %564 = arith.cmpi slt, %6, %563 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %565 = math.log2 %538 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %566 = arith.addf %537, %565 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    gpu.barrier
    %567 = ttg.convert_layout %566 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %559, %567, %564 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %568 = math.log2 %538 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %569 = arith.addf %537, %568 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    gpu.barrier
    %570 = ttg.convert_layout %569 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %559, %570 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %571 = arith.muli %2, %arg14 : i32
    %572 = tt.addptr %arg4, %571 : !tt.ptr<f16>, i32
    %573 = arith.muli %1, %arg15 : i32
    %574 = tt.addptr %572, %573 : !tt.ptr<f16>, i32
    %575 = arith.muli %22, %arg16 : i32
    %576 = tt.addptr %574, %575 : !tt.ptr<f16>, i32
    %577 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %578 = arith.muli %76, %577 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %579 = tt.splat %576 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %580 = tt.addptr %579, %578 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %581 = tt.broadcast %580 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %582 = tt.addptr %581, %87 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %583 = arith.select %561, %125, %cst_14 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    tt.store %582, %553, %583 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb37:  // pred: ^bb1
    tt.return
  }
}
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register=1 -> (64)
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (0)
   lane=32 -> (0)
 - warp=1 -> (16)
   warp=2 -> (32)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
sizePerThread: 1
basesPerDim: 2
	numElementsPerThread: 1
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
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
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x16
	numElementsPerThread: 4
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
	numElementsPerThread: 4
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x16
	numElementsPerThread: 4
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x16
packLLElements()
tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32x%8 = "llvm.bitcast"(%7) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 1x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
 - lane=1 -> (0, 0)
   lane=2 -> (0, 0)
   lane=4 -> (0, 0)
   lane=8 -> (0, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (0, 0)
   warp=2 -> (0, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 1), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 1x16
packLLElements()
tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32x%16 = "llvm.bitcast"(%15) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x16
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32x%24 = "llvm.bitcast"(%23) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x16
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32x%32 = "llvm.bitcast"(%31) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x64
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x16
packLLElements()
tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32x%41 = "llvm.bitcast"(%40) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
packLLElements()
tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32xf32xf32xf32
resultVals: %49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32x%49 = "llvm.bitcast"(%48) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
packLLElements()
tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f16xf16xf16xf16
resultVals: %58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16x%58 = "llvm.bitcast"(%57) : (f16) -> f16
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x1
ll: 
 - register=1 -> (0, 0)
   register=2 -> (0, 0)
   register=4 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 0)
   lane=32 -> (0, 0)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 1)]
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x1
basesPerDim: 8x1
	numElementsPerThread: 1
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x1
ll: 
 - register=1 -> (0, 0)
   register=2 -> (0, 0)
   register=4 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 0)
   lane=32 -> (0, 0)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 1)]
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x1
basesPerDim: 8x1
packLLElements()
tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: f32
resultVals: %66 = "llvm.bitcast"(%65) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register=1 -> (64)
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (0)
   lane=32 -> (0)
 - warp=1 -> (16)
   warp=2 -> (32)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
sizePerThread: 1
basesPerDim: 2
packLLElements()
tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
elementTypes: f32
resultVals: %71 = "llvm.bitcast"(%70) : (f32) -> f32
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register=1 -> (64)
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (0)
   lane=32 -> (0)
 - warp=1 -> (16)
   warp=2 -> (32)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
sizePerThread: 1
basesPerDim: 2
packLLElements()
tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
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
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
	numElementsPerThread: 4
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128x128
ll: 
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 16)
   register=8 -> (0, 32)
   register=16 -> (0, 64)
   register=32 -> (64, 0)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (16, 0)
   warp=2 -> (32, 0)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
sizePerThread: 1x4
basesPerDim: 2x32
packLLElements()
tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
elementTypes: i1xi1xi1xi1
resultVals: %147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1x%147 = "llvm.bitcast"(%146) : (i1) -> i1
convertTritonTensorType()
LinearEncodingAttr::getElemsPerThread() determining size of LL Vector
shape: 128
ll: 
 - register=1 -> (64)
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (0)
   lane=32 -> (0)
 - warp=1 -> (16)
   warp=2 -> (32)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128)]
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
sizePerThread: 1
basesPerDim: 2
	numElementsPerThread: 1
packLLElements()
tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
elementTypes: i32
resultVals: %223 = "llvm.add"(%220, %168) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32x%224 = "llvm.add"(%222, %168) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:569:50: error:  size mismatch when packing elements for LLVM struct expected 1 but got 2
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                                                 ^
python3: /home/dtanner/.triton/llvm/llvm-2619c2ed-ubuntu-x64/include/llvm/ADT/ArrayRef.h:260: const T &llvm::ArrayRef<mlir::Type>::operator[](size_t) const [T = mlir::Type]: Assertion `Index < Length && "Invalid index!"' failed.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
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
    %164 = amdgpu.extract_slice %163 [0, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %165 = amdgpu.extract_slice %cst_3 [0, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %166 = arith.mulf %164, %165 : tensor<64x16xf32, #mma>
    %167 = amdgpu.extract_slice %163 [0, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %168 = amdgpu.extract_slice %cst_3 [0, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %169 = arith.mulf %167, %168 : tensor<64x16xf32, #mma>
    %170 = amdgpu.extract_slice %163 [0, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %171 = amdgpu.extract_slice %cst_3 [0, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %172 = arith.mulf %170, %171 : tensor<64x16xf32, #mma>
    %173 = amdgpu.extract_slice %163 [0, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %174 = amdgpu.extract_slice %cst_3 [0, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %175 = arith.mulf %173, %174 : tensor<64x16xf32, #mma>
    %176 = amdgpu.extract_slice %163 [64, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %177 = amdgpu.extract_slice %cst_3 [64, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %178 = arith.mulf %176, %177 : tensor<64x16xf32, #mma>
    %179 = amdgpu.extract_slice %163 [64, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %180 = amdgpu.extract_slice %cst_3 [64, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %181 = arith.mulf %179, %180 : tensor<64x16xf32, #mma>
    %182 = amdgpu.extract_slice %163 [64, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %183 = amdgpu.extract_slice %cst_3 [64, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %184 = arith.mulf %182, %183 : tensor<64x16xf32, #mma>
    %185 = amdgpu.extract_slice %163 [64, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %186 = amdgpu.extract_slice %cst_3 [64, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %187 = arith.mulf %185, %186 : tensor<64x16xf32, #mma>
    %188 = amdgpu.extract_slice %cst_2 [0, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %189 = arith.addf %166, %188 : tensor<64x16xf32, #mma>
    %190 = amdgpu.extract_slice %cst_2 [0, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %191 = arith.addf %172, %190 : tensor<64x16xf32, #mma>
    %192 = amdgpu.extract_slice %cst_2 [0, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %193 = arith.addf %178, %192 : tensor<64x16xf32, #mma>
    %194 = amdgpu.extract_slice %cst_2 [0, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %195 = arith.addf %184, %194 : tensor<64x16xf32, #mma>
    %196 = amdgpu.extract_slice %cst_2 [64, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %197 = arith.addf %169, %196 : tensor<64x16xf32, #mma>
    %198 = amdgpu.extract_slice %cst_2 [64, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %199 = arith.addf %175, %198 : tensor<64x16xf32, #mma>
    %200 = amdgpu.extract_slice %cst_2 [64, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %201 = arith.addf %181, %200 : tensor<64x16xf32, #mma>
    %202 = amdgpu.extract_slice %cst_2 [64, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %203 = arith.addf %187, %202 : tensor<64x16xf32, #mma>
    %204 = amdgpu.concat %189, %191, %193, %195, %197, %199, %201, %203 [2, 4] : tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma> -> tensor<128x64xf32, #mma>
    %205 = "tt.reduce"(%204) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %206 = amdgpu.extract_slice %154 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %207 = amdgpu.extract_slice %205 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %208 = arith.maxnumf %206, %207 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %209 = amdgpu.extract_slice %154 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %210 = amdgpu.extract_slice %205 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %211 = arith.maxnumf %209, %210 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %212 = amdgpu.concat %208, %211 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %213 = tt.expand_dims %212 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %214 = tt.broadcast %213 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %215 = amdgpu.extract_slice %214 [0, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %216 = arith.subf %189, %215 : tensor<64x16xf32, #mma>
    %217 = amdgpu.extract_slice %214 [0, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %218 = arith.subf %193, %217 : tensor<64x16xf32, #mma>
    %219 = amdgpu.extract_slice %214 [0, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %220 = arith.subf %197, %219 : tensor<64x16xf32, #mma>
    %221 = amdgpu.extract_slice %214 [0, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %222 = arith.subf %201, %221 : tensor<64x16xf32, #mma>
    %223 = amdgpu.extract_slice %214 [64, 0] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %224 = arith.subf %191, %223 : tensor<64x16xf32, #mma>
    %225 = amdgpu.extract_slice %214 [64, 16] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %226 = arith.subf %195, %225 : tensor<64x16xf32, #mma>
    %227 = amdgpu.extract_slice %214 [64, 32] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %228 = arith.subf %199, %227 : tensor<64x16xf32, #mma>
    %229 = amdgpu.extract_slice %214 [64, 48] : tensor<128x64xf32, #mma> to tensor<64x16xf32, #mma>
    %230 = arith.subf %203, %229 : tensor<64x16xf32, #mma>
    %231 = math.exp2 %216 : tensor<64x16xf32, #mma>
    %232 = math.exp2 %220 : tensor<64x16xf32, #mma>
    %233 = math.exp2 %224 : tensor<64x16xf32, #mma>
    %234 = math.exp2 %228 : tensor<64x16xf32, #mma>
    %235 = math.exp2 %218 : tensor<64x16xf32, #mma>
    %236 = math.exp2 %222 : tensor<64x16xf32, #mma>
    %237 = math.exp2 %226 : tensor<64x16xf32, #mma>
    %238 = math.exp2 %230 : tensor<64x16xf32, #mma>
    %239 = amdgpu.concat %231, %232, %233, %234, %235, %236, %237, %238 [2, 4] : tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma> -> tensor<128x64xf32, #mma>
    %240 = "tt.reduce"(%239) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %241 = amdgpu.extract_slice %154 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %242 = arith.subf %241, %208 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %243 = amdgpu.extract_slice %154 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %244 = arith.subf %243, %211 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %245 = math.exp2 %242 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %246 = math.exp2 %244 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %247 = amdgpu.concat %245, %246 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %248 = tt.expand_dims %247 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %249 = tt.broadcast %248 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %250 = amdgpu.extract_slice %152 [0, 0] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %251 = amdgpu.extract_slice %249 [0, 0] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %252 = arith.mulf %250, %251 : tensor<64x16xf32, #mma>
    %253 = amdgpu.extract_slice %152 [0, 16] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %254 = amdgpu.extract_slice %249 [0, 16] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %255 = arith.mulf %253, %254 : tensor<64x16xf32, #mma>
    %256 = amdgpu.extract_slice %152 [0, 32] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %257 = amdgpu.extract_slice %249 [0, 32] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %258 = arith.mulf %256, %257 : tensor<64x16xf32, #mma>
    %259 = amdgpu.extract_slice %152 [0, 48] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %260 = amdgpu.extract_slice %249 [0, 48] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %261 = arith.mulf %259, %260 : tensor<64x16xf32, #mma>
    %262 = amdgpu.extract_slice %152 [0, 64] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %263 = amdgpu.extract_slice %249 [0, 64] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %264 = arith.mulf %262, %263 : tensor<64x16xf32, #mma>
    %265 = amdgpu.extract_slice %152 [0, 80] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %266 = amdgpu.extract_slice %249 [0, 80] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %267 = arith.mulf %265, %266 : tensor<64x16xf32, #mma>
    %268 = amdgpu.extract_slice %152 [0, 96] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %269 = amdgpu.extract_slice %249 [0, 96] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %270 = arith.mulf %268, %269 : tensor<64x16xf32, #mma>
    %271 = amdgpu.extract_slice %152 [0, 112] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %272 = amdgpu.extract_slice %249 [0, 112] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %273 = arith.mulf %271, %272 : tensor<64x16xf32, #mma>
    %274 = amdgpu.extract_slice %152 [64, 0] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %275 = amdgpu.extract_slice %249 [64, 0] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %276 = arith.mulf %274, %275 : tensor<64x16xf32, #mma>
    %277 = amdgpu.extract_slice %152 [64, 16] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %278 = amdgpu.extract_slice %249 [64, 16] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %279 = arith.mulf %277, %278 : tensor<64x16xf32, #mma>
    %280 = amdgpu.extract_slice %152 [64, 32] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %281 = amdgpu.extract_slice %249 [64, 32] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %282 = arith.mulf %280, %281 : tensor<64x16xf32, #mma>
    %283 = amdgpu.extract_slice %152 [64, 48] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %284 = amdgpu.extract_slice %249 [64, 48] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %285 = arith.mulf %283, %284 : tensor<64x16xf32, #mma>
    %286 = amdgpu.extract_slice %152 [64, 64] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %287 = amdgpu.extract_slice %249 [64, 64] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %288 = arith.mulf %286, %287 : tensor<64x16xf32, #mma>
    %289 = amdgpu.extract_slice %152 [64, 80] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %290 = amdgpu.extract_slice %249 [64, 80] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %291 = arith.mulf %289, %290 : tensor<64x16xf32, #mma>
    %292 = amdgpu.extract_slice %152 [64, 96] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %293 = amdgpu.extract_slice %249 [64, 96] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %294 = arith.mulf %292, %293 : tensor<64x16xf32, #mma>
    %295 = amdgpu.extract_slice %152 [64, 112] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %296 = amdgpu.extract_slice %249 [64, 112] : tensor<128x128xf32, #mma> to tensor<64x16xf32, #mma>
    %297 = arith.mulf %295, %296 : tensor<64x16xf32, #mma>
    %298 = amdgpu.concat %252, %255, %258, %261, %264, %267, %270, %273, %276, %279, %282, %285, %288, %291, %294, %297 [2, 8] : tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma>, tensor<64x16xf32, #mma> -> tensor<128x128xf32, #mma>
    %299 = amdgpu.extract_slice %153 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %300 = arith.mulf %299, %245 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %301 = amdgpu.extract_slice %153 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %302 = arith.mulf %301, %246 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %303 = amdgpu.extract_slice %240 [0] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %304 = arith.addf %300, %303 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %305 = amdgpu.extract_slice %240 [64] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %306 = arith.addf %302, %305 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %307 = amdgpu.concat %304, %306 [2] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %308 = arith.truncf %231 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %309 = arith.truncf %233 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %310 = arith.truncf %235 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %311 = arith.truncf %237 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %312 = arith.truncf %232 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %313 = arith.truncf %234 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %314 = arith.truncf %236 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %315 = arith.truncf %238 : tensor<64x16xf32, #mma> to tensor<64x16xf16, #mma>
    %316 = amdgpu.concat %308, %309, %310, %311, %312, %313, %314, %315 [2, 4] : tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma>, tensor<64x16xf16, #mma> -> tensor<128x64xf16, #mma>
    %317 = ttg.convert_layout %316 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %318 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %319 = ttg.local_load %318 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %320 = tt.dot %317, %319, %298, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    %321 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    gpu.barrier
    %322 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %323 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%323, %320, %307, %212, %159, %321, %322 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb13:  // pred: ^bb11
    %324 = arith.addi %141, %c63_i32 : i32
    %325 = arith.divsi %324, %c64_i32 : i32
    %326 = arith.cmpi sge, %325, %c1_i32 : i32
    gpu.barrier
    %327 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %328 = tt.splat %326 : i1 -> tensor<64x128xi1, #blocked>
    %329 = tt.load %156, %328 : tensor<64x128x!tt.ptr<f16>, #blocked>
    cf.cond_br %326, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb14:  // pred: ^bb13
    %330 = tt.dot %129, %327, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    cf.br ^bb15(%330 : tensor<128x64xf32, #mma>)
  ^bb15(%331: tensor<128x64xf32, #mma>):  // 2 preds: ^bb13, ^bb14
    %332 = arith.mulf %331, %cst_3 : tensor<128x64xf32, #mma>
    %333 = arith.addf %332, %cst_2 : tensor<128x64xf32, #mma>
    %334 = "tt.reduce"(%333) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %335 = arith.maxnumf %154, %334 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %336 = tt.expand_dims %335 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %337 = tt.broadcast %336 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %338 = arith.subf %333, %337 : tensor<128x64xf32, #mma>
    %339 = math.exp2 %338 : tensor<128x64xf32, #mma>
    %340 = "tt.reduce"(%339) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %341 = arith.subf %154, %335 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %342 = math.exp2 %341 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %343 = tt.expand_dims %342 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %344 = tt.broadcast %343 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %345 = arith.mulf %152, %344 : tensor<128x128xf32, #mma>
    %346 = arith.mulf %153, %342 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %347 = arith.addf %346, %340 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %348 = arith.truncf %339 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %349 = ttg.convert_layout %348 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %350 = ttg.local_alloc %329 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %351 = ttg.local_load %350 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    cf.cond_br %326, ^bb16, ^bb17(%345 : tensor<128x128xf32, #mma>)
  ^bb16:  // pred: ^bb15
    %352 = tt.dot %349, %351, %345, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    cf.br ^bb17(%352 : tensor<128x128xf32, #mma>)
  ^bb17(%353: tensor<128x128xf32, #mma>):  // 2 preds: ^bb15, ^bb16
    %354 = arith.select %326, %353, %152 : tensor<128x128xf32, #mma>
    %355 = arith.select %326, %347, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %356 = arith.select %326, %335, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    cf.br ^bb18(%356, %355, %354, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32)
  ^bb18(%357: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %358: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %359: tensor<128x128xf32, #mma>, %360: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %361 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %361, ^bb19, ^bb31(%357, %358, %359 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>)
  ^bb19:  // pred: ^bb18
    %362 = arith.subi %24, %28 : i32
    %363 = tt.splat %362 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %364 = arith.addi %14, %363 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %365 = arith.muli %138, %c64_i32 : i32
    %366 = arith.muli %365, %arg10 : i32
    %367 = tt.splat %366 : i32 -> tensor<128x64xi32, #blocked1>
    %368 = tt.addptr %106, %367 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %369 = arith.muli %365, %arg13 : i32
    %370 = tt.splat %369 : i32 -> tensor<64x128xi32, #blocked>
    %371 = tt.addptr %120, %370 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %372 = tt.splat %28 : i32 -> tensor<1x64xi32, #blocked1>
    %373 = tt.splat %28 : i32 -> tensor<1x64xi32, #mma>
    %374 = tt.splat %28 : i32 -> tensor<64x1xi32, #blocked>
    %375 = arith.cmpi ne, %69, %c0_i32 : i32
    %376 = tt.broadcast %76 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
    %377 = arith.muli %arg10, %c64_i32 : i32
    %378 = tt.splat %377 : i32 -> tensor<128x64xi32, #blocked1>
    %379 = arith.muli %arg13, %c64_i32 : i32
    %380 = tt.splat %379 : i32 -> tensor<64x128xi32, #blocked>
    %381 = arith.cmpi slt, %360, %139 : i32
    %382 = tt.splat %360 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %383 = arith.addi %382, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %384 = tt.expand_dims %383 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %385 = arith.cmpi slt, %384, %372 : tensor<1x64xi32, #blocked1>
    %386 = tt.broadcast %385 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %387 = tt.splat %381 : i1 -> tensor<128x64xi1, #blocked1>
    %388 = arith.andi %387, %386 : tensor<128x64xi1, #blocked1>
    %389 = tt.load %368, %388, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %390 = ttg.local_alloc %389 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %391 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%360, %359, %358, %357, %368, %371, %390 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb20(%392: i32, %393: tensor<128x128xf32, #mma>, %394: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %395: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %396: tensor<128x64x!tt.ptr<f16>, #blocked1>, %397: tensor<64x128x!tt.ptr<f16>, #blocked>, %398: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>):  // 2 preds: ^bb19, ^bb23
    %399 = arith.cmpi slt, %392, %391 : i32
    cf.cond_br %399, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %400 = tt.addptr %396, %378 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %401 = arith.addi %392, %c64_i32 : i32
    %402 = tt.splat %401 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %403 = tt.splat %392 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %404 = tt.splat %392 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %405 = amdgpu.extract_slice %402 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %406 = amdgpu.extract_slice %13 [0] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %407 = arith.addi %405, %406 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %408 = amdgpu.extract_slice %402 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %409 = amdgpu.extract_slice %13 [16] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %410 = arith.addi %408, %409 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %411 = amdgpu.extract_slice %402 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %412 = amdgpu.extract_slice %13 [32] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %413 = arith.addi %411, %412 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %414 = amdgpu.extract_slice %402 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %415 = amdgpu.extract_slice %13 [48] : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %416 = arith.addi %414, %415 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %417 = amdgpu.concat %407, %410, %413, %416 [4] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %418 = amdgpu.extract_slice %404 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %419 = amdgpu.extract_slice %15 [0] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %420 = arith.addi %418, %419 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %421 = amdgpu.extract_slice %404 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %422 = amdgpu.extract_slice %15 [16] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %423 = arith.addi %421, %422 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %424 = amdgpu.extract_slice %404 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %425 = amdgpu.extract_slice %15 [32] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %426 = arith.addi %424, %425 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %427 = amdgpu.extract_slice %404 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %428 = amdgpu.extract_slice %15 [48] : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %429 = arith.addi %427, %428 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %430 = amdgpu.concat %420, %423, %426, %429 [4] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %431 = tt.expand_dims %417 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %432 = arith.cmpi slt, %431, %372 : tensor<1x64xi32, #blocked1>
    %433 = tt.broadcast %432 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %434 = tt.load %400, %433, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    gpu.barrier
    %435 = ttg.local_load %398 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %436 = tt.expand_dims %430 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %437 = arith.cmpi slt, %436, %374 : tensor<64x1xi32, #blocked>
    %438 = tt.broadcast %437 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %439 = tt.load %397, %438, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %440 = arith.cmpi eq, %401, %139 : i32
    %441 = arith.andi %440, %375 : i1
    cf.cond_br %441, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb22:  // pred: ^bb21
    %442 = tt.splat %392 : i32 -> tensor<1x64xi32, #mma>
    %443 = arith.addi %442, %101 : tensor<1x64xi32, #mma>
    %444 = arith.cmpi slt, %443, %373 : tensor<1x64xi32, #mma>
    %445 = arith.select %444, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
    %446 = tt.broadcast %445 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
    cf.br ^bb23(%446 : tensor<128x64xf32, #mma>)
  ^bb23(%447: tensor<128x64xf32, #mma>):  // 2 preds: ^bb21, ^bb22
    %448 = arith.addi %403, %364 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %449 = tt.expand_dims %448 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %450 = tt.broadcast %449 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
    %451 = arith.cmpi sge, %376, %450 : tensor<128x64xi32, #mma>
    %452 = arith.select %451, %447, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
    %453 = tt.dot %129, %435, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    %454 = arith.mulf %453, %cst_3 : tensor<128x64xf32, #mma>
    %455 = arith.addf %452, %454 : tensor<128x64xf32, #mma>
    %456 = "tt.reduce"(%455) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %457 = arith.maxnumf %395, %456 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %458 = tt.expand_dims %457 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %459 = tt.broadcast %458 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %460 = arith.subf %455, %459 : tensor<128x64xf32, #mma>
    %461 = math.exp2 %460 : tensor<128x64xf32, #mma>
    %462 = "tt.reduce"(%461) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %463 = arith.subf %395, %457 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %464 = math.exp2 %463 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %465 = tt.expand_dims %464 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %466 = tt.broadcast %465 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %467 = arith.mulf %393, %466 : tensor<128x128xf32, #mma>
    %468 = arith.mulf %394, %464 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %469 = arith.addf %468, %462 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %470 = arith.truncf %461 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %471 = ttg.convert_layout %470 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %472 = ttg.local_alloc %439 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %473 = ttg.local_load %472 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %474 = tt.dot %471, %473, %467, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    %475 = tt.addptr %397, %380 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    gpu.barrier
    %476 = ttg.local_alloc %434 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %477 = arith.addi %392, %c64_i32 : i32
    cf.br ^bb20(%477, %474, %469, %457, %400, %475, %476 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb24:  // pred: ^bb20
    %478 = arith.subi %139, %360 : i32
    %479 = arith.addi %478, %c63_i32 : i32
    %480 = arith.divsi %479, %c64_i32 : i32
    %481 = arith.subi %480, %c1_i32 : i32
    %482 = arith.maxsi %481, %c0_i32 : i32
    %483 = arith.muli %482, %c64_i32 : i32
    %484 = arith.addi %360, %483 : i32
    %485 = arith.cmpi sge, %480, %c1_i32 : i32
    %486 = tt.splat %484 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %487 = tt.splat %484 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %488 = arith.addi %487, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    gpu.barrier
    %489 = ttg.local_load %398 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %490 = tt.expand_dims %488 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %491 = arith.cmpi slt, %490, %374 : tensor<64x1xi32, #blocked>
    %492 = tt.broadcast %491 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %493 = tt.splat %485 : i1 -> tensor<64x128xi1, #blocked>
    %494 = arith.andi %493, %492 : tensor<64x128xi1, #blocked>
    %495 = tt.load %397, %494, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %496 = arith.addi %484, %c64_i32 : i32
    %497 = arith.cmpi eq, %496, %139 : i32
    %498 = arith.andi %497, %375 : i1
    cf.cond_br %498, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb25:  // pred: ^bb24
    %499 = tt.splat %484 : i32 -> tensor<1x64xi32, #mma>
    %500 = arith.addi %499, %101 : tensor<1x64xi32, #mma>
    %501 = arith.cmpi slt, %500, %373 : tensor<1x64xi32, #mma>
    %502 = arith.select %501, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
    %503 = tt.broadcast %502 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
    cf.br ^bb26(%503 : tensor<128x64xf32, #mma>)
  ^bb26(%504: tensor<128x64xf32, #mma>):  // 2 preds: ^bb24, ^bb25
    %505 = arith.addi %486, %364 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %506 = tt.expand_dims %505 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %507 = tt.broadcast %506 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
    %508 = arith.cmpi sge, %376, %507 : tensor<128x64xi32, #mma>
    %509 = arith.select %508, %504, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
    cf.cond_br %485, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #mma>)
  ^bb27:  // pred: ^bb26
    %510 = tt.dot %129, %489, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    cf.br ^bb28(%510 : tensor<128x64xf32, #mma>)
  ^bb28(%511: tensor<128x64xf32, #mma>):  // 2 preds: ^bb26, ^bb27
    %512 = arith.mulf %511, %cst_3 : tensor<128x64xf32, #mma>
    %513 = arith.addf %509, %512 : tensor<128x64xf32, #mma>
    %514 = "tt.reduce"(%513) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %515 = arith.maxnumf %395, %514 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %516 = tt.expand_dims %515 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %517 = tt.broadcast %516 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %518 = arith.subf %513, %517 : tensor<128x64xf32, #mma>
    %519 = math.exp2 %518 : tensor<128x64xf32, #mma>
    %520 = "tt.reduce"(%519) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %584 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %584 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %521 = arith.subf %395, %515 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %522 = math.exp2 %521 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %523 = tt.expand_dims %522 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %524 = tt.broadcast %523 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %525 = arith.mulf %393, %524 : tensor<128x128xf32, #mma>
    %526 = arith.mulf %394, %522 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %527 = arith.addf %526, %520 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %528 = arith.truncf %519 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %529 = ttg.convert_layout %528 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %530 = ttg.local_alloc %495 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %531 = ttg.local_load %530 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    cf.cond_br %485, ^bb29, ^bb30(%525 : tensor<128x128xf32, #mma>)
  ^bb29:  // pred: ^bb28
    %532 = tt.dot %529, %531, %525, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    cf.br ^bb30(%532 : tensor<128x128xf32, #mma>)
  ^bb30(%533: tensor<128x128xf32, #mma>):  // 2 preds: ^bb28, ^bb29
    %534 = arith.select %485, %533, %393 : tensor<128x128xf32, #mma>
    %535 = arith.select %485, %527, %394 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %536 = arith.select %485, %515, %395 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    cf.br ^bb31(%536, %535, %534 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>)
  ^bb31(%537: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %538: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %539: tensor<128x128xf32, #mma>):  // 2 preds: ^bb18, ^bb30
    %540 = tt.expand_dims %538 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %541 = arith.divf %cst_6, %540 : tensor<128x1xf32, #mma>
    %542 = tt.broadcast %541 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %543 = arith.mulf %539, %542 : tensor<128x128xf32, #mma>
    %544 = arith.subi %24, %28 : i32
    %545 = arith.truncf %543 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %546 = arith.cmpi sgt, %544, %3 : i32
    %547 = arith.cmpi slt, %544, %32 : i32
    %548 = arith.andi %546, %547 : i1
    cf.cond_br %548, ^bb32, ^bb33(%545 : tensor<128x128xf16, #mma>)
  ^bb32:  // pred: ^bb31
    %549 = tt.splat %544 : i32 -> tensor<128x1xi32, #mma>
    %550 = arith.cmpi sge, %76, %549 : tensor<128x1xi32, #mma>
    %551 = tt.broadcast %550 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %552 = arith.select %551, %545, %cst_5 : tensor<128x128xi1, #mma>, tensor<128x128xf16, #mma>
    cf.br ^bb33(%552 : tensor<128x128xf16, #mma>)
  ^bb33(%553: tensor<128x128xf16, #mma>):  // 2 preds: ^bb31, ^bb32
    %554 = arith.muli %2, %c54848_i32 : i32
    %555 = tt.addptr %arg3, %554 : !tt.ptr<f32>, i32
    %556 = arith.muli %1, %c3428_i32 : i32
    %557 = tt.addptr %555, %556 : !tt.ptr<f32>, i32
    %558 = tt.splat %557 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %559 = tt.addptr %558, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
    %560 = arith.subi %32, %24 : i32
    %561 = arith.cmpi sgt, %560, %c0_i32 : i32
    cf.cond_br %561, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %562 = arith.subi %c128_i32, %560 : i32
    %563 = tt.splat %562 : i32 -> tensor<128xi32, #blocked2>
    %564 = arith.cmpi slt, %6, %563 : tensor<128xi32, #blocked2>
    %565 = math.log2 %538 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %566 = arith.addf %537, %565 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    gpu.barrier
    %567 = ttg.convert_layout %566 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
    tt.store %559, %567, %564 : tensor<128x!tt.ptr<f32>, #blocked2>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %568 = math.log2 %538 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %569 = arith.addf %537, %568 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    gpu.barrier
    %570 = ttg.convert_layout %569 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
    tt.store %559, %570 : tensor<128x!tt.ptr<f32>, #blocked2>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %571 = arith.muli %2, %arg14 : i32
    %572 = tt.addptr %arg4, %571 : !tt.ptr<f16>, i32
    %573 = arith.muli %1, %arg15 : i32
    %574 = tt.addptr %572, %573 : !tt.ptr<f16>, i32
    %575 = arith.muli %22, %arg16 : i32
    %576 = tt.addptr %574, %575 : !tt.ptr<f16>, i32
    %577 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
    %578 = arith.muli %76, %577 : tensor<128x1xi32, #mma>
    %579 = tt.splat %576 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %580 = tt.addptr %579, %578 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %581 = tt.broadcast %580 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #mma>
    %582 = tt.addptr %581, %87 : tensor<128x128x!tt.ptr<f16>, #mma>, tensor<128x128xi32, #mma>
    %583 = arith.select %561, %125, %cst_14 : tensor<128x128xi1, #mma>
    tt.store %582, %553, %583 : tensor<128x128x!tt.ptr<f16>, #mma>
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
