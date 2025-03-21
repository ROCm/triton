AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x64
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x1
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %164 = arith.mulf %163, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 64x16
numReps: 2x4
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 64x16
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 128x128
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
[tritonamdgpu-refine-ops]: failed to refine binary op: %201 = arith.mulf %152, %200 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
[tritonamdgpu-refine-ops]: failed to refine binary op: %202 = arith.mulf %153, %198 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
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
    %164 = amdgpu.extract_slice %163 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %165 = amdgpu.extract_slice %cst_3 [0, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %166 = arith.mulf %164, %165 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %167 = amdgpu.extract_slice %163 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %168 = amdgpu.extract_slice %cst_3 [0, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %169 = arith.mulf %167, %168 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %170 = amdgpu.extract_slice %163 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %171 = amdgpu.extract_slice %cst_3 [0, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %172 = arith.mulf %170, %171 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %173 = amdgpu.extract_slice %163 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %174 = amdgpu.extract_slice %cst_3 [0, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %175 = arith.mulf %173, %174 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %176 = amdgpu.extract_slice %163 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %177 = amdgpu.extract_slice %cst_3 [64, 0] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %178 = arith.mulf %176, %177 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %179 = amdgpu.extract_slice %163 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %180 = amdgpu.extract_slice %cst_3 [64, 16] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %181 = arith.mulf %179, %180 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %182 = amdgpu.extract_slice %163 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %183 = amdgpu.extract_slice %cst_3 [64, 32] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %184 = arith.mulf %182, %183 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %185 = amdgpu.extract_slice %163 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %186 = amdgpu.extract_slice %cst_3 [64, 48] : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %187 = arith.mulf %185, %186 : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
    %188 = amdgpu.concat %166, %169, %172, %175, %178, %181, %184, %187 [2, 4] : tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>, tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %189 = arith.addf %188, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %190 = "tt.reduce"(%189) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %191 = arith.maxnumf %154, %190 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %192 = tt.expand_dims %191 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %193 = tt.broadcast %192 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %194 = arith.subf %189, %193 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %195 = math.exp2 %194 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %196 = "tt.reduce"(%195) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %197 = arith.subf %154, %191 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %198 = math.exp2 %197 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %199 = tt.expand_dims %198 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %200 = tt.broadcast %199 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %201 = arith.mulf %152, %200 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %202 = arith.mulf %153, %198 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %203 = arith.addf %202, %196 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %204 = arith.truncf %195 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %205 = ttg.convert_layout %204 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %206 = ttg.local_alloc %162 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %207 = ttg.local_load %206 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    %208 = tt.dot %205, %207, %201, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %209 = tt.addptr %156, %145 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %210 = ttg.local_alloc %160 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %211 = arith.addi %151, %c64_i32 : i32
    cf.br ^bb11(%211, %208, %203, %191, %159, %209, %210 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb13:  // pred: ^bb11
    %212 = arith.addi %141, %c63_i32 : i32
    %213 = arith.divsi %212, %c64_i32 : i32
    %214 = arith.cmpi sge, %213, %c1_i32 : i32
    gpu.barrier
    %215 = ttg.local_load %157 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %216 = tt.splat %214 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %217 = tt.load %156, %216 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    cf.cond_br %214, ^bb14, ^bb15(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb14:  // pred: ^bb13
    %218 = tt.dot %129, %215, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb15(%218 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb15(%219: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb13, ^bb14
    %220 = arith.mulf %219, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %221 = arith.addf %220, %cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %222 = "tt.reduce"(%221) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %223 = arith.maxnumf %154, %222 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %224 = tt.expand_dims %223 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %225 = tt.broadcast %224 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %226 = arith.subf %221, %225 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %227 = math.exp2 %226 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %228 = "tt.reduce"(%227) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %229 = arith.subf %154, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %230 = math.exp2 %229 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %231 = tt.expand_dims %230 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %232 = tt.broadcast %231 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %233 = arith.mulf %152, %232 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %234 = arith.mulf %153, %230 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %235 = arith.addf %234, %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %236 = arith.truncf %227 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %237 = ttg.convert_layout %236 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %238 = ttg.local_alloc %217 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %239 = ttg.local_load %238 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %214, ^bb16, ^bb17(%233 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb16:  // pred: ^bb15
    %240 = tt.dot %237, %239, %233, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb17(%240 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb17(%241: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb15, ^bb16
    %242 = arith.select %214, %241, %152 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %243 = arith.select %214, %235, %153 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %244 = arith.select %214, %223, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    cf.br ^bb18(%244, %243, %242, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, i32)
  ^bb18(%245: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %246: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %247: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, %248: i32):  // 2 preds: ^bb9, ^bb17
    gpu.barrier
    %249 = arith.cmpi sgt, %137, %c0_i32 : i32
    cf.cond_br %249, ^bb19, ^bb31(%245, %246, %247 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb19:  // pred: ^bb18
    %250 = arith.subi %24, %28 : i32
    %251 = tt.splat %250 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %252 = arith.addi %14, %251 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %253 = arith.muli %138, %c64_i32 : i32
    %254 = arith.muli %253, %arg10 : i32
    %255 = tt.splat %254 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %256 = tt.addptr %106, %255 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %257 = arith.muli %253, %arg13 : i32
    %258 = tt.splat %257 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %259 = tt.addptr %120, %258 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %260 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %261 = tt.splat %28 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %262 = tt.splat %28 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %263 = arith.cmpi ne, %69, %c0_i32 : i32
    %264 = tt.broadcast %76 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %265 = arith.muli %arg10, %c64_i32 : i32
    %266 = tt.splat %265 : i32 -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %267 = arith.muli %arg13, %c64_i32 : i32
    %268 = tt.splat %267 : i32 -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %269 = arith.cmpi slt, %248, %139 : i32
    %270 = tt.splat %248 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %271 = arith.addi %270, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %272 = tt.expand_dims %271 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %273 = arith.cmpi slt, %272, %260 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %274 = tt.broadcast %273 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %275 = tt.splat %269 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %276 = arith.andi %275, %274 : tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %277 = tt.load %256, %276, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %278 = ttg.local_alloc %277 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %279 = arith.subi %139, %c64_i32 : i32
    cf.br ^bb20(%248, %247, %246, %245, %256, %259, %278 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb20(%280: i32, %281: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, %282: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %283: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %284: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, %285: tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, %286: !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>):  // 2 preds: ^bb19, ^bb23
    %287 = arith.cmpi slt, %280, %279 : i32
    cf.cond_br %287, ^bb21, ^bb24
  ^bb21:  // pred: ^bb20
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %288 = tt.addptr %284, %266 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %289 = arith.addi %280, %c64_i32 : i32
    %290 = tt.splat %289 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %291 = tt.splat %280 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %292 = tt.splat %280 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %293 = arith.addi %290, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %294 = arith.addi %292, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %295 = tt.expand_dims %293 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %296 = arith.cmpi slt, %295, %260 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %297 = tt.broadcast %296 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %298 = tt.load %288, %297, %cst_10 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
    gpu.barrier
    %299 = ttg.local_load %286 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %300 = tt.expand_dims %294 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %301 = arith.cmpi slt, %300, %262 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %302 = tt.broadcast %301 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %303 = tt.load %285, %302, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %304 = arith.cmpi eq, %289, %139 : i32
    %305 = arith.andi %304, %263 : i1
    cf.cond_br %305, ^bb22, ^bb23(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb22:  // pred: ^bb21
    %306 = tt.splat %280 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %307 = arith.addi %306, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %308 = arith.cmpi slt, %307, %261 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %309 = arith.select %308, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %310 = tt.broadcast %309 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb23(%310 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb23(%311: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb21, ^bb22
    %312 = arith.addi %291, %252 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %313 = tt.expand_dims %312 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %314 = tt.broadcast %313 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %315 = arith.cmpi sge, %264, %314 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %316 = arith.select %315, %311, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %317 = tt.dot %129, %299, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %318 = arith.mulf %317, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %319 = arith.addf %316, %318 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %320 = "tt.reduce"(%319) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %321 = arith.maxnumf %283, %320 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %322 = tt.expand_dims %321 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %323 = tt.broadcast %322 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %324 = arith.subf %319, %323 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %325 = math.exp2 %324 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %326 = "tt.reduce"(%325) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %327 = arith.subf %283, %321 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %328 = math.exp2 %327 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %329 = tt.expand_dims %328 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %330 = tt.broadcast %329 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %331 = arith.mulf %281, %330 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %332 = arith.mulf %282, %328 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %333 = arith.addf %332, %326 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %334 = arith.truncf %325 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %335 = ttg.convert_layout %334 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %336 = ttg.local_alloc %303 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %337 = ttg.local_load %336 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    %338 = tt.dot %335, %337, %331, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %339 = tt.addptr %285, %268 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    gpu.barrier
    %340 = ttg.local_alloc %298 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %341 = arith.addi %280, %c64_i32 : i32
    cf.br ^bb20(%341, %338, %333, %321, %288, %339, %340 : i32, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>)
  ^bb24:  // pred: ^bb20
    %342 = arith.subi %139, %248 : i32
    %343 = arith.addi %342, %c63_i32 : i32
    %344 = arith.divsi %343, %c64_i32 : i32
    %345 = arith.subi %344, %c1_i32 : i32
    %346 = arith.maxsi %345, %c0_i32 : i32
    %347 = arith.muli %346, %c64_i32 : i32
    %348 = arith.addi %248, %347 : i32
    %349 = arith.cmpi sge, %344, %c1_i32 : i32
    %350 = tt.splat %348 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %351 = tt.splat %348 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %352 = arith.addi %351, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    gpu.barrier
    %353 = ttg.local_load %286 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>>
    %354 = tt.expand_dims %352 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %355 = arith.cmpi slt, %354, %262 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %356 = tt.broadcast %355 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %357 = tt.splat %349 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %358 = arith.andi %357, %356 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %359 = tt.load %285, %358, %cst_9 : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %360 = arith.addi %348, %c64_i32 : i32
    %361 = arith.cmpi eq, %360, %139 : i32
    %362 = arith.andi %361, %263 : i1
    cf.cond_br %362, ^bb25, ^bb26(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb25:  // pred: ^bb24
    %363 = tt.splat %348 : i32 -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %364 = arith.addi %363, %101 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %365 = arith.cmpi slt, %364, %261 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %366 = arith.select %365, %cst_0, %cst : tensor<1x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %367 = tt.broadcast %366 : tensor<1x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb26(%367 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb26(%368: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb24, ^bb25
    %369 = arith.addi %350, %252 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %370 = tt.expand_dims %369 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %371 = tt.broadcast %370 : tensor<1x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %372 = arith.cmpi sge, %264, %371 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %373 = arith.select %372, %368, %cst_1 : tensor<128x64xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.cond_br %349, ^bb27, ^bb28(%cst_2 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb27:  // pred: ^bb26
    %374 = tt.dot %129, %353, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb28(%374 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb28(%375: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb26, ^bb27
    %376 = arith.mulf %375, %cst_3 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %377 = arith.addf %373, %376 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %378 = "tt.reduce"(%377) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.maxnumf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %379 = arith.maxnumf %283, %378 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %380 = tt.expand_dims %379 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %381 = tt.broadcast %380 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %382 = arith.subf %377, %381 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %383 = math.exp2 %382 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %384 = "tt.reduce"(%383) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %448 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %448 : f32
    }) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %385 = arith.subf %283, %379 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %386 = math.exp2 %385 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %387 = tt.expand_dims %386 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %388 = tt.broadcast %387 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %389 = arith.mulf %281, %388 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %390 = arith.mulf %282, %386 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %391 = arith.addf %390, %384 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %392 = arith.truncf %383 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %393 = ttg.convert_layout %392 : tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    gpu.barrier
    %394 = ttg.local_alloc %359 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory>
    gpu.barrier
    %395 = ttg.local_load %394 : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>>
    cf.cond_br %349, ^bb29, ^bb30(%389 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb29:  // pred: ^bb28
    %396 = tt.dot %393, %395, %389, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>, kWidth = 4}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb30(%396 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb30(%397: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb28, ^bb29
    %398 = arith.select %349, %397, %281 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %399 = arith.select %349, %391, %282 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %400 = arith.select %349, %379, %283 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    cf.br ^bb31(%400, %399, %398 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb31(%401: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %402: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>, %403: tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb18, ^bb30
    %404 = tt.expand_dims %402 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %405 = arith.divf %cst_6, %404 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %406 = tt.broadcast %405 : tensor<128x1xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %407 = arith.mulf %403, %406 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %408 = arith.subi %24, %28 : i32
    %409 = arith.truncf %407 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> to tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %410 = arith.cmpi sgt, %408, %3 : i32
    %411 = arith.cmpi slt, %408, %32 : i32
    %412 = arith.andi %410, %411 : i1
    cf.cond_br %412, ^bb32, ^bb33(%409 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb32:  // pred: ^bb31
    %413 = tt.splat %408 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %414 = arith.cmpi sge, %76, %413 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %415 = tt.broadcast %414 : tensor<128x1xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %416 = arith.select %415, %409, %cst_5 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb33(%416 : tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>)
  ^bb33(%417: tensor<128x128xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>):  // 2 preds: ^bb31, ^bb32
    %418 = arith.muli %2, %c54848_i32 : i32
    %419 = tt.addptr %arg3, %418 : !tt.ptr<f32>, i32
    %420 = arith.muli %1, %c3428_i32 : i32
    %421 = tt.addptr %419, %420 : !tt.ptr<f32>, i32
    %422 = tt.splat %421 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %423 = tt.addptr %422, %12 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %424 = arith.subi %32, %24 : i32
    %425 = arith.cmpi sgt, %424, %c0_i32 : i32
    cf.cond_br %425, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %426 = arith.subi %c128_i32, %424 : i32
    %427 = tt.splat %426 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %428 = arith.cmpi slt, %6, %427 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %429 = math.log2 %402 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %430 = arith.addf %401, %429 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    gpu.barrier
    %431 = ttg.convert_layout %430 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %423, %431, %428 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb35:  // pred: ^bb33
    %432 = math.log2 %402 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    %433 = arith.addf %401, %432 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>>
    gpu.barrier
    %434 = ttg.convert_layout %433 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %423, %434 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    cf.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    %435 = arith.muli %2, %arg14 : i32
    %436 = tt.addptr %arg4, %435 : !tt.ptr<f16>, i32
    %437 = arith.muli %1, %arg15 : i32
    %438 = tt.addptr %436, %437 : !tt.ptr<f16>, i32
    %439 = arith.muli %22, %arg16 : i32
    %440 = tt.addptr %438, %439 : !tt.ptr<f16>, i32
    %441 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %442 = arith.muli %76, %441 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %443 = tt.splat %440 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %444 = tt.addptr %443, %442 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %445 = tt.broadcast %444 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>> -> tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %446 = tt.addptr %445, %87 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>, tensor<128x128xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    %447 = arith.select %425, %125, %cst_14 : tensor<128x128xi1, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    tt.store %446, %417, %447 : tensor<128x128x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb37:  // pred: ^bb1
    tt.return
  }
}
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 1x64
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
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
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[16], [32], [64]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
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
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
AMDMfmaEncodingAttr::toLinearLayout: trans=1, shape: 1x128
tileLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
where out dims are: [dim1 (size 16), dim0 (size 16)]
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
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
finalLinearLayout: 
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (4, 0)
   lane=32 -> (8, 0)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 16), dim0 (size 64)]
#ttg.linear<{register = [[1], [2], [16], [32], [64]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [4]], lane = [[8], [16], [32], [64], [0], [0]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
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
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
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
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [32, 0], [64, 0], [0, 16], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12757 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x0
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x0
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
#ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12795 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x0
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x0
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12851 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x16
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x1
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12889 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x16
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x1
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12945 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x32
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x2
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12983 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x32
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x2
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13039 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x48
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x3
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13077 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 0x48
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 0x3
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13133 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 64, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x0
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x0
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13171 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 64, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x0
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x0
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13227 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 64, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x16
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x1
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13265 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 64, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x16
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x1
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13321 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 64, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x32
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x2
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13359 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 64, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x32
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x2
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13415 = "amdgpu.extract_slice"(%12724) <{static_offsets = array<i64: 64, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x48
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x3
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
resultTy.getShape(): 64x16
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13453 = "amdgpu.extract_slice"(%162) <{static_offsets = array<i64: 64, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>>) -> tensor<64x16xf32, #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>>
srcShape: 128x64
elemsPerThread: 2x16
contigPerThread: 1x4
offsets: 64x48
vals.size(): 32
shapePerCTATile: 64x16
shapePerCTATile: 64x16
sizes: 64x16
CTAOffsets: 1x3
CTASizes: 1x1
CTAPerShape: 2x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 12
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
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
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [16, 0], [32, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#ttg.linear<{register = [[1], [2], [16], [32]], lane = [[0], [0], [0], [0], [4], [8]], warp = [[0], [0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 0], [0, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [0], [0]], warp = [[16], [32]], block = []}>
#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [0]], block = []}>
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:2091: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /var/lib/jenkins/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:517.)
  fn = lambda: torch.nn.functional.scaled_dot_product_attention(
INFO: running a single config given from a file
fused-attention-fwd-d128-layoutthd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K     triton      torch
0    2.0  16.0  16.0   8192.0   8192.0  60.829747  14.970682
