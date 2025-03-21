AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x64
regBases: 16
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 32), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x1
regBases: 2
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 8), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 8), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 8), dim0 (size 128)]
[tritonamdgpu-refine-ops]: failed to refine binary op: %164 = arith.mulf %163, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x128
regBases: 16
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 32), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %177 = arith.mulf %152, %176 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
srcShapePerCtaTile: 128x8
numReps: 1x16
[tritonamdgpu-refine-ops]: rewriteElementWiseOp() - SUCCESS %177 = arith.mulf %152, %176 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: failed to refine binary op: %226 = arith.mulf %153, %174 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
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
    %164 = arith.mulf %163, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %165 = arith.addf %164, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %166 = "tt.reduce"(%165) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %167 = arith.maxnumf %154, %166 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %168 = tt.expand_dims %167 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %169 = tt.broadcast %168 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %170 = arith.subf %165, %169 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %171 = math.exp2 %170 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %172 = "tt.reduce"(%171) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %173 = arith.subf %154, %167 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %174 = math.exp2 %173 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %175 = tt.expand_dims %174 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %176 = tt.broadcast %175 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %177 = amdgpu.extract_slice %152 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %178 = amdgpu.extract_slice %176 [0, 0] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %179 = arith.mulf %177, %178 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %180 = amdgpu.extract_slice %152 [0, 8] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %181 = amdgpu.extract_slice %176 [0, 8] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %182 = arith.mulf %180, %181 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %183 = amdgpu.extract_slice %152 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %184 = amdgpu.extract_slice %176 [0, 16] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %185 = arith.mulf %183, %184 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %186 = amdgpu.extract_slice %152 [0, 24] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %187 = amdgpu.extract_slice %176 [0, 24] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %188 = arith.mulf %186, %187 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %189 = amdgpu.extract_slice %152 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %190 = amdgpu.extract_slice %176 [0, 32] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %191 = arith.mulf %189, %190 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %192 = amdgpu.extract_slice %152 [0, 40] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %193 = amdgpu.extract_slice %176 [0, 40] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %194 = arith.mulf %192, %193 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %195 = amdgpu.extract_slice %152 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %196 = amdgpu.extract_slice %176 [0, 48] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %197 = arith.mulf %195, %196 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %198 = amdgpu.extract_slice %152 [0, 56] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %199 = amdgpu.extract_slice %176 [0, 56] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %200 = arith.mulf %198, %199 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %201 = amdgpu.extract_slice %152 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %202 = amdgpu.extract_slice %176 [0, 64] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %203 = arith.mulf %201, %202 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %204 = amdgpu.extract_slice %152 [0, 72] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %205 = amdgpu.extract_slice %176 [0, 72] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %206 = arith.mulf %204, %205 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %207 = amdgpu.extract_slice %152 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %208 = amdgpu.extract_slice %176 [0, 80] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %209 = arith.mulf %207, %208 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %210 = amdgpu.extract_slice %152 [0, 88] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %211 = amdgpu.extract_slice %176 [0, 88] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %212 = arith.mulf %210, %211 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %213 = amdgpu.extract_slice %152 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %214 = amdgpu.extract_slice %176 [0, 96] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %215 = arith.mulf %213, %214 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %216 = amdgpu.extract_slice %152 [0, 104] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %217 = amdgpu.extract_slice %176 [0, 104] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %218 = arith.mulf %216, %217 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %219 = amdgpu.extract_slice %152 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %220 = amdgpu.extract_slice %176 [0, 112] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %221 = arith.mulf %219, %220 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %222 = amdgpu.extract_slice %152 [0, 120] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %223 = amdgpu.extract_slice %176 [0, 120] : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %224 = arith.mulf %222, %223 : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %225 = amdgpu.concat %179, %182, %185, %188, %191, %194, %197, %200, %203, %206, %209, %212, %215, %218, %221, %224 [1, 16] : tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %226 = arith.mulf %153, %174 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %227 = arith.addf %226, %172 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %228 = arith.truncf %171 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %229 = ttg.convert_layout %228 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %230 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %231 = ttg.local_load %230 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    %232 = tt.dot %229, %231, %225, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %233 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %234 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %235 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%235, %232, %227, %167, %159, %233, %234 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb13:  // pred: ^bb11
    %236 = arith.addi %141, %c63_i32 : i32
    %237 = arith.divsi %236, %c64_i32 : i32
    %238 = arith.cmpi sge, %237, %c1_i32 : i32
    gpu.barrier
    %239 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %240 = tt.splat %238 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %241 = tt.load %156, %240 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    cf.cond_br %238, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb14:  // pred: ^bb13
    %242 = tt.dot %129, %239, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb15(%242 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb15(%243: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb13, ^bb14
    %244 = arith.mulf %243, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %245 = arith.addf %244, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %246 = "tt.reduce"(%245) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %247 = arith.maxnumf %154, %246 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %248 = tt.expand_dims %247 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %249 = tt.broadcast %248 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %250 = arith.subf %245, %249 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %251 = math.exp2 %250 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %252 = "tt.reduce"(%251) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %253 = arith.subf %154, %247 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %254 = math.exp2 %253 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %255 = tt.expand_dims %254 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %256 = tt.broadcast %255 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %257 = arith.mulf %152, %256 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %258 = arith.mulf %153, %254 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %259 = arith.addf %258, %252 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %260 = arith.truncf %251 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %261 = ttg.convert_layout %260 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %262 = ttg.local_alloc %241 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %263 = ttg.local_load %262 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %238, ^bb16, ^bb17(%257 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb16:  // pred: ^bb15
    %264 = tt.dot %261, %263, %257, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb17(%264 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb17(%265: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb15, ^bb16
    %266 = arith.select %238, %265, %152 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %267 = arith.select %238, %259, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %268 = arith.select %238, %247, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    cf.br ^bb18(%268, %267, %266, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, i32)
  ^bb18(%269: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %270: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %271: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, %272: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %273 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %273, ^bb19, ^bb31(%269, %270, %271 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb19:  // pred: ^bb18
    %274 = arith.subi %24, %28 : i32
    %275 = tt.splat %274 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %276 = arith.addi %14, %275 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %277 = arith.muli %138, %c64_i32 : i32
    %278 = arith.muli %277, %arg10 : i32
    %279 = tt.splat %278 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %280 = tt.addptr %106, %279 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %281 = arith.muli %277, %arg13 : i32
    %282 = tt.splat %281 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %283 = tt.addptr %120, %282 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %284 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %285 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %286 = tt.splat %28 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %287 = arith.cmpi ne, %69, %c0_i32 : i32
    %288 = tt.broadcast %76 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %289 = arith.muli %arg10, %c64_i32 : i32
    %290 = tt.splat %289 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %291 = arith.muli %arg13, %c64_i32 : i32
    %292 = tt.splat %291 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %293 = arith.cmpi slt, %272, %139 : i32
    %294 = tt.splat %272 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %295 = arith.addi %294, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %296 = tt.expand_dims %295 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %297 = arith.cmpi slt, %296, %284 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %298 = tt.broadcast %297 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %299 = tt.splat %293 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %300 = arith.andi %299, %298 : tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %301 = tt.load %280, %300, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %302 = ttg.local_alloc %301 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %303 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%272, %271, %270, %269, %280, %283, %302 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb20(%304: i32, %305: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, %306: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %307: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %308: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %309: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %310: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb19, ^bb23
    %311 = arith.cmpi slt, %304, %303 : i32
    cf.cond_br %311, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %312 = tt.addptr %308, %290 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %313 = arith.addi %304, %c64_i32 : i32
    %314 = tt.splat %313 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %315 = tt.splat %304 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %316 = tt.splat %304 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %317 = arith.addi %314, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %318 = arith.addi %316, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %319 = tt.expand_dims %317 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %320 = arith.cmpi slt, %319, %284 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %321 = tt.broadcast %320 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %322 = tt.load %312, %321, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %323 = ttg.local_load %310 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %324 = tt.expand_dims %318 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %325 = arith.cmpi slt, %324, %286 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %326 = tt.broadcast %325 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %327 = tt.load %309, %326, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %328 = arith.cmpi eq, %313, %139 : i32
    %329 = arith.andi %328, %287 : i1
    cf.cond_br %329, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb22:  // pred: ^bb21
    %330 = tt.splat %304 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %331 = arith.addi %330, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %332 = arith.cmpi slt, %331, %285 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %333 = arith.select %332, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %334 = tt.broadcast %333 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb23(%334 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb23(%335: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb21, ^bb22
    %336 = arith.addi %315, %276 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %337 = tt.expand_dims %336 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %338 = tt.broadcast %337 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %339 = arith.cmpi sge, %288, %338 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %340 = arith.select %339, %335, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %341 = tt.dot %129, %323, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %342 = arith.mulf %341, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %343 = arith.addf %340, %342 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %344 = "tt.reduce"(%343) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %345 = arith.maxnumf %307, %344 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %346 = tt.expand_dims %345 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %347 = tt.broadcast %346 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %348 = arith.subf %343, %347 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %349 = math.exp2 %348 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %350 = "tt.reduce"(%349) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %351 = arith.subf %307, %345 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %352 = math.exp2 %351 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %353 = tt.expand_dims %352 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %354 = tt.broadcast %353 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %355 = arith.mulf %305, %354 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %356 = arith.mulf %306, %352 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %357 = arith.addf %356, %350 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %358 = arith.truncf %349 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %359 = ttg.convert_layout %358 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %360 = ttg.local_alloc %327 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %361 = ttg.local_load %360 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    %362 = tt.dot %359, %361, %355, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %363 = tt.addptr %309, %292 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %364 = ttg.local_alloc %322 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %365 = arith.addi %304, %c64_i32 : i32
    cf.br ^bb20(%365, %362, %357, %345, %312, %363, %364 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb24:  // pred: ^bb20
    %366 = arith.subi %139, %272 : i32
    %367 = arith.addi %366, %c63_i32 : i32
    %368 = arith.divsi %367, %c64_i32 : i32
    %369 = arith.subi %368, %c1_i32 : i32
    %370 = arith.maxsi %369, %c0_i32 : i32
    %371 = arith.muli %370, %c64_i32 : i32
    %372 = arith.addi %272, %371 : i32
    %373 = arith.cmpi sge, %368, %c1_i32 : i32
    %374 = tt.splat %372 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %375 = tt.splat %372 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %376 = arith.addi %375, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    gpu.barrier
    %377 = ttg.local_load %310 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
    %378 = tt.expand_dims %376 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %379 = arith.cmpi slt, %378, %286 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %380 = tt.broadcast %379 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %381 = tt.splat %373 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %382 = arith.andi %381, %380 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %383 = tt.load %309, %382, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %384 = arith.addi %372, %c64_i32 : i32
    %385 = arith.cmpi eq, %384, %139 : i32
    %386 = arith.andi %385, %287 : i1
    cf.cond_br %386, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb25:  // pred: ^bb24
    %387 = tt.splat %372 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %388 = arith.addi %387, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %389 = arith.cmpi slt, %388, %285 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %390 = arith.select %389, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %391 = tt.broadcast %390 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb26(%391 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb26(%392: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb24, ^bb25
    %393 = arith.addi %374, %276 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %394 = tt.expand_dims %393 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %395 = tt.broadcast %394 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %396 = arith.cmpi sge, %288, %395 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %397 = arith.select %396, %392, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.cond_br %373, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb27:  // pred: ^bb26
    %398 = tt.dot %129, %377, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb28(%398 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb28(%399: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb26, ^bb27
    %400 = arith.mulf %399, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %401 = arith.addf %397, %400 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %402 = "tt.reduce"(%401) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %403 = arith.maxnumf %307, %402 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %404 = tt.expand_dims %403 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %405 = tt.broadcast %404 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %406 = arith.subf %401, %405 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %407 = math.exp2 %406 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %408 = "tt.reduce"(%407) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %472 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %472 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %409 = arith.subf %307, %403 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %410 = math.exp2 %409 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %411 = tt.expand_dims %410 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %412 = tt.broadcast %411 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %413 = arith.mulf %305, %412 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %414 = arith.mulf %306, %410 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %415 = arith.addf %414, %408 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %416 = arith.truncf %407 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %417 = ttg.convert_layout %416 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %418 = ttg.local_alloc %383 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %419 = ttg.local_load %418 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %373, ^bb29, ^bb30(%413 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb29:  // pred: ^bb28
    %420 = tt.dot %417, %419, %413, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb30(%420 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb30(%421: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb28, ^bb29
    %422 = arith.select %373, %421, %305 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %423 = arith.select %373, %415, %306 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %424 = arith.select %373, %403, %307 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    cf.br ^bb31(%424, %423, %422 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb31(%425: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %426: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, %427: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb18, ^bb30
    %428 = tt.expand_dims %426 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %429 = arith.divf %cst_6, %428 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %430 = tt.broadcast %429 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %431 = arith.mulf %427, %430 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %432 = arith.subi %24, %28 : i32
    %433 = arith.truncf %431 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %434 = arith.cmpi sgt, %432, %3 : i32
    %435 = arith.cmpi slt, %432, %32 : i32
    %436 = arith.andi %434, %435 : i1
    cf.cond_br %436, ^bb32, ^bb33(%433 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb32:  // pred: ^bb31
    %437 = tt.splat %432 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %438 = arith.cmpi sge, %76, %437 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %439 = tt.broadcast %438 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %440 = arith.select %439, %433, %cst_5 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb33(%440 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>)
  ^bb33(%441: tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>):  // 2 preds: ^bb31, ^bb32
    %442 = arith.muli %2, %c54848_i32 : i32
    %443 = tt.addptr %arg3, %442 : !tt.ptr<f32>, i32
    %444 = arith.muli %1, %c3428_i32 : i32
    %445 = tt.addptr %443, %444 : !tt.ptr<f32>, i32
    %446 = tt.splat %445 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %447 = tt.addptr %446, %12 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %448 = arith.subi %32, %24 : i32
    %449 = arith.cmpi sgt, %448, %c0_i32 : i32
    cf.cond_br %449, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %450 = arith.subi %c128_i32, %448 : i32
    %451 = tt.splat %450 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %452 = arith.cmpi slt, %6, %451 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %453 = math.log2 %426 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %454 = arith.addf %425, %453 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    gpu.barrier
    %455 = ttg.convert_layout %454 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %447, %455, %452 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %456 = math.log2 %426 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    %457 = arith.addf %425, %456 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
    gpu.barrier
    %458 = ttg.convert_layout %457 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %447, %458 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %459 = arith.muli %2, %arg14 : i32
    %460 = tt.addptr %arg4, %459 : !tt.ptr<f16>, i32
    %461 = arith.muli %1, %arg15 : i32
    %462 = tt.addptr %460, %461 : !tt.ptr<f16>, i32
    %463 = arith.muli %22, %arg16 : i32
    %464 = tt.addptr %462, %463 : !tt.ptr<f16>, i32
    %465 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %466 = arith.muli %76, %465 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %467 = tt.splat %464 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %468 = tt.addptr %467, %466 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %469 = tt.broadcast %468 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %470 = tt.addptr %469, %87 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    %471 = arith.select %449, %125, %cst_14 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    tt.store %470, %441, %471 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb37:  // pred: ^bb1
    tt.return
  }
}
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 1x64
regBases: 16
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 32), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [4]], lane = [[8], [16], [32], [64], [0], [0]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 1x128
regBases: 16
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 32), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 32), dim0 (size 128)]
#ttg.linear<{register = [[1], [2], [8], [16], [32], [64]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [4]], lane = [[8], [16], [32], [64], [0], [0]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [16, 0], [32, 0], [64, 0], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14436 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x0
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x0
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x8
regBases: 2
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
where out dims are: [dim1 (size 8), dim0 (size 32)]
warpLayout: 
 - warp=1 -> (0, 1)
   warp=2 -> (0, 2)
where out dims are: [dim1 (size 1), dim0 (size 4)]
ctaLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 8), dim0 (size 128)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (0, 32)
   warp=2 -> (0, 64)
where out dims are: [dim1 (size 8), dim0 (size 128)]
#ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14506 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x0
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x0
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14594 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x8
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x1
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14664 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x8
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x1
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14752 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x16
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x2
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14822 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x16
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x2
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14910 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x24
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x3
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14980 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x24
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x3
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15068 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x32
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x4
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15138 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x32
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x4
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15226 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x40
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x5
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15296 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x40
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x5
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15384 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x48
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x6
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15454 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x48
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x6
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15542 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x56
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x7
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15612 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x56
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x7
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15700 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 64>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x64
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x8
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 36
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15770 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 64>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x64
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x8
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 36
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15858 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 72>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x72
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x9
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 36
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 40
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15928 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 72>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x72
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x9
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 36
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 40
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16016 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 80>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x80
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x10
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 40
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 44
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16086 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 80>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x80
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x10
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 40
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 44
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16174 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 88>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x88
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x11
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 44
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16244 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 88>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x88
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x11
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 44
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16332 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 96>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x96
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x12
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 52
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16402 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 96>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x96
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x12
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 52
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16490 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 104>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x104
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x13
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 52
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 56
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16560 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 104>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x104
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x13
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 52
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 56
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16648 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 112>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x112
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x14
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 56
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 60
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16718 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 112>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x112
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x14
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 56
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 60
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16806 = "amdgpu.extract_slice"(%10200) <{static_offsets = array<i64: 0, 120>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x120
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x15
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 60
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=3
resultTy.getShape(): 128x8
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16876 = "amdgpu.extract_slice"(%14371) <{static_offsets = array<i64: 0, 120>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x128
elemsPerThread: 1x64
contigPerThread: 1x4
offsets: 0x120
vals.size(): 64
shapePerCTATile: 128x8
shapePerCTATile: 128x8
sizes: 128x8
CTAOffsets: 0x15
CTASizes: 1x1
CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 60
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=3
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [8, 0], [16, 0], [32, 0], [0, 32], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [8], [16], [32]], lane = [[0], [0], [0], [0], [0], [4]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [0]], warp = [[32], [64]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:2091: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /var/lib/jenkins/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:517.)
  fn = lambda: torch.nn.functional.scaled_dot_product_attention(
INFO: running a single config given from a file
fused-attention-fwd-d128-layoutthd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K     triton      torch
0    2.0  16.0  16.0   8192.0   8192.0  78.366418  14.808346
