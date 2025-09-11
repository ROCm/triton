// RUN: triton-opt %s -split-input-file --tritonamdgpu-aggregate-load=factor=-1 | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-aggregate-load=factor=2 | FileCheck %s --check-prefix=FACTOR2

// CHECK-LABEL: kernel_no_k_stride
// CHECK: local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_no_k_stride(
      %arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32},
      %arg8: i32 {tt.divisibility = 16 : i32},
      %arg9: i32 {tt.divisibility = 16 : i32},
      %arg10: i32 {tt.divisibility = 16 : i32},
      %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg12: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x256xf8E4M3FN, #blocked>
    %cst_1 = arith.constant dense<256> : tensor<16x256xi32, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<128x256xi32, #blocked1>
    %cst_3 = arith.constant dense<8> : tensor<256x8xi32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_4 = arith.constant dense<8> : tensor<16x8xi32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %7 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %8 = arith.muli %3, %c16_i32 : i32
    %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %11 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.splat %8 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.splat %8 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %15 = tt.splat %8 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %16 = tt.splat %8 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = arith.addi %13, %9 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.addi %14, %10 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %19 = arith.addi %15, %11 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %20 = arith.addi %16, %12 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.splat %arg11 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = tt.splat %arg11 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %23 = tt.splat %arg11 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #linear}>>
    %24 = tt.splat %arg11 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = tt.addptr %21, %17 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %26 = tt.addptr %22, %18 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked3}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %27 = tt.addptr %23, %19 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %28 = tt.addptr %24, %20 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %29 = tt.load %25 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %30 = tt.load %26 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %31 = tt.load %27 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #linear}>>
    %32 = tt.load %28 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %33 = tt.splat %arg12 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %34 = tt.splat %arg12 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %35 = arith.cmpi slt, %32, %33 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %36 = arith.cmpi slt, %30, %34 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %37 = arith.muli %4, %c256_i32 : i32
    %38 = tt.splat %37 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %39 = tt.splat %37 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %40 = tt.splat %37 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %41 = arith.addi %38, %5 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %42 = arith.addi %39, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %43 = arith.addi %40, %7 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %44 = tt.expand_dims %29 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %45 = tt.expand_dims %30 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %46 = tt.expand_dims %31 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<16x1xi32, #linear>
    %47 = tt.splat %arg8 : i32 -> tensor<16x1xi32, #blocked>
    %48 = arith.muli %44, %47 : tensor<16x1xi32, #blocked>
    %49 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<16x1x!tt.ptr<f8E4M3FN>, #blocked>
    %50 = tt.addptr %49, %48 : tensor<16x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<16x1xi32, #blocked>
    %51 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %53 = tt.broadcast %50 : tensor<16x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>
    %54 = tt.broadcast %52 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %55 = tt.addptr %53, %54 : tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<16x256xi32, #blocked>
    %56 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %57 = tt.expand_dims %56 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %58 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked1>
    %59 = tt.addptr %58, %57 : tensor<128x1x!tt.ptr<i8>, #blocked1>, tensor<128x1xi32, #blocked1>
    %60 = tt.expand_dims %41 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %61 = tt.expand_dims %42 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %62 = tt.splat %arg9 : i32 -> tensor<1x256xi32, #blocked1>
    %63 = arith.muli %60, %62 : tensor<1x256xi32, #blocked1>
    %64 = tt.broadcast %59 : tensor<128x1x!tt.ptr<i8>, #blocked1> -> tensor<128x256x!tt.ptr<i8>, #blocked1>
    %65 = tt.broadcast %63 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %66 = tt.addptr %64, %65 : tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<128x256xi32, #blocked1>
    %67 = tt.splat %arg7 : i32 -> tensor<16x1xi32, #linear>
    %68 = arith.muli %46, %67 : tensor<16x1xi32, #linear>
    %69 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #linear>
    %70 = tt.addptr %69, %68 : tensor<16x1x!tt.ptr<i8>, #linear>, tensor<16x1xi32, #linear>
    %71 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %72 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %73 = tt.expand_dims %71 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %74 = tt.expand_dims %72 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x8xi32, #linear>
    %75 = tt.broadcast %70 : tensor<16x1x!tt.ptr<i8>, #linear> -> tensor<16x8x!tt.ptr<i8>, #linear>
    %76 = tt.broadcast %74 : tensor<1x8xi32, #linear> -> tensor<16x8xi32, #linear>
    %77 = tt.addptr %75, %76 : tensor<16x8x!tt.ptr<i8>, #linear>, tensor<16x8xi32, #linear>
    %78 = tt.expand_dims %43 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<256x1xi32, #blocked2>
    %79 = tt.splat %arg7 : i32 -> tensor<256x1xi32, #blocked2>
    %80 = arith.muli %78, %79 : tensor<256x1xi32, #blocked2>
    %81 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked2>
    %82 = tt.addptr %81, %80 : tensor<256x1x!tt.ptr<i8>, #blocked2>, tensor<256x1xi32, #blocked2>
    %83 = tt.broadcast %82 : tensor<256x1x!tt.ptr<i8>, #blocked2> -> tensor<256x8x!tt.ptr<i8>, #blocked2>
    %84 = tt.broadcast %73 : tensor<1x8xi32, #blocked2> -> tensor<256x8xi32, #blocked2>
    %85 = tt.addptr %83, %84 : tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<256x8xi32, #blocked2>
    %86 = tt.expand_dims %35 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi1, #blocked>
    %87 = tt.broadcast %86 : tensor<16x1xi1, #blocked> -> tensor<16x256xi1, #blocked>
    %88:5 = scf.for %arg13 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg14 = %cst, %arg15 = %55, %arg16 = %66, %arg17 = %85, %arg18 = %77) -> (tensor<16x256xf32, #mma>, tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<16x8x!tt.ptr<i8>, #linear>)  : i32 {
      %104 = tt.load %arg15, %87, %cst_0 : tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>
      %105 = tt.load %arg16 : tensor<128x256x!tt.ptr<i8>, #blocked1>
      %106 = tt.load %arg18 : tensor<16x8x!tt.ptr<i8>, #linear>
      %107 = tt.load %arg17 : tensor<256x8x!tt.ptr<i8>, #blocked2>
      %108 = ttg.convert_layout %104 : tensor<16x256xf8E4M3FN, #blocked> -> tensor<16x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %109 = ttg.convert_layout %105 : tensor<128x256xi8, #blocked1> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %110 = ttg.convert_layout %107 : tensor<256x8xi8, #blocked2> -> tensor<256x8xi8, #linear1>
      %111 = tt.dot_scaled %108 scale %106, %109 scale %110, %arg14 lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<16x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<16x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<16x256xf32, #mma>
      %112 = tt.addptr %arg15, %cst_1 : tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<16x256xi32, #blocked>
      %113 = tt.addptr %arg16, %cst_2 : tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<128x256xi32, #blocked1>
      %114 = tt.addptr %arg18, %cst_4 : tensor<16x8x!tt.ptr<i8>, #linear>, tensor<16x8xi32, #linear>
      %115 = tt.addptr %arg17, %cst_3 : tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<256x8xi32, #blocked2>
      scf.yield %111, %112, %113, %115, %114 : tensor<16x256xf32, #mma>, tensor<16x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<16x8x!tt.ptr<i8>, #linear>
    }
    %89 = tt.splat %arg10 : i32 -> tensor<16x1xi32, #blocked3>
    %90 = arith.muli %89, %45 : tensor<16x1xi32, #blocked3>
    %91 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked3>
    %92 = tt.addptr %91, %90 : tensor<16x1x!tt.ptr<f32>, #blocked3>, tensor<16x1xi32, #blocked3>
    %93 = tt.broadcast %92 : tensor<16x1x!tt.ptr<f32>, #blocked3> -> tensor<16x256x!tt.ptr<f32>, #blocked3>
    %94 = tt.broadcast %61 : tensor<1x256xi32, #blocked3> -> tensor<16x256xi32, #blocked3>
    %95 = tt.addptr %93, %94 : tensor<16x256x!tt.ptr<f32>, #blocked3>, tensor<16x256xi32, #blocked3>
    %96 = tt.expand_dims %36 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi1, #blocked3>
    %97 = tt.splat %arg6 : i32 -> tensor<1x256xi32, #blocked3>
    %98 = arith.cmpi slt, %61, %97 : tensor<1x256xi32, #blocked3>
    %99 = tt.broadcast %96 : tensor<16x1xi1, #blocked3> -> tensor<16x256xi1, #blocked3>
    %100 = tt.broadcast %98 : tensor<1x256xi1, #blocked3> -> tensor<16x256xi1, #blocked3>
    %101 = arith.andi %99, %100 : tensor<16x256xi1, #blocked3>
    %102 = ttg.convert_layout %95 : tensor<16x256x!tt.ptr<f32>, #blocked3> -> tensor<16x256x!tt.ptr<f32>, #mma>
    %103 = ttg.convert_layout %101 : tensor<16x256xi1, #blocked3> -> tensor<16x256xi1, #mma>
    tt.store %102, %88#0, %103 : tensor<16x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: kernel_strided_k
// CHECK: local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_strided_k(
      %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<8192> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<8192> : tensor<1x64xi32, #blocked1>
    %cst_1 = arith.constant dense<32> : tensor<1x8xi32, #blocked2>
    %cst_2 = arith.constant dense<53248> : tensor<1x8xi32, #blocked3>
    %cst_3 = arith.constant dense<128> : tensor<32x128xi32, #blocked>
    %cst_4 = arith.constant dense<128> : tensor<128x64xi32, #blocked1>
    %cst_5 = arith.constant dense<256> : tensor<32x8xi32, #blocked2>
    %cst_6 = arith.constant dense<425984> : tensor<64x8xi32, #blocked3>
    %cst_7 = arith.constant dense<53248> : tensor<32x1xi64, #blocked4>
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.cmpi sgt, %3, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = arith.cmpi sgt, %4, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.muli %3, %c32_i32 : i32
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %11 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = arith.addi %11, %8 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.addi %12, %9 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %15 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = arith.remsi %13, %15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.remsi %14, %16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = arith.muli %4, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %23 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %25 = arith.addi %23, %20 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %24, %22 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %27 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %29 = arith.remsi %25, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %30 = arith.remsi %26, %28 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %31 = tt.expand_dims %17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %32 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %33 = arith.muli %31, %cst : tensor<32x1xi32, #blocked>
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %36 = tt.broadcast %33 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %38 = arith.addi %36, %37 : tensor<32x128xi32, #blocked>
    %39 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>, #blocked>
    %40 = tt.addptr %39, %38 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
    %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %43 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %44 = arith.muli %43, %cst_0 : tensor<1x64xi32, #blocked1>
    %45 = tt.broadcast %42 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %47 = arith.addi %45, %46 : tensor<128x64xi32, #blocked1>
    %48 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked1>
    %49 = tt.addptr %48, %47 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
    %50 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked2>
    %51 = tt.addptr %50, %32 : tensor<32x1x!tt.ptr<i8>, #blocked2>, tensor<32x1xi32, #blocked2>
    %52 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %53 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %54 = tt.expand_dims %52 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %55 = tt.expand_dims %53 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x8xi32, #blocked3>
    %56 = arith.muli %54, %cst_1 : tensor<1x8xi32, #blocked2>
    %57 = tt.broadcast %51 : tensor<32x1x!tt.ptr<i8>, #blocked2> -> tensor<32x8x!tt.ptr<i8>, #blocked2>
    %58 = tt.broadcast %56 : tensor<1x8xi32, #blocked2> -> tensor<32x8xi32, #blocked2>
    %59 = tt.addptr %57, %58 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
    %60 = tt.expand_dims %30 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %61 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<64x1x!tt.ptr<i8>, #blocked3>
    %62 = tt.addptr %61, %60 : tensor<64x1x!tt.ptr<i8>, #blocked3>, tensor<64x1xi32, #blocked3>
    %63 = arith.muli %55, %cst_2 : tensor<1x8xi32, #blocked3>
    %64 = tt.broadcast %62 : tensor<64x1x!tt.ptr<i8>, #blocked3> -> tensor<64x8x!tt.ptr<i8>, #blocked3>
    %65 = tt.broadcast %63 : tensor<1x8xi32, #blocked3> -> tensor<64x8xi32, #blocked3>
    %66 = tt.addptr %64, %65 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
    %67:5 = scf.for %arg7 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg8 = %cst_8, %arg9 = %59, %arg10 = %66, %arg11 = %40, %arg12 = %49) -> (tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>)  : i32 {
      %96 = tt.load %arg9 : tensor<32x8x!tt.ptr<i8>, #blocked2>
      %97 = tt.load %arg10 : tensor<64x8x!tt.ptr<i8>, #blocked3>
      %98 = tt.load %arg11 : tensor<32x128x!tt.ptr<i8>, #blocked>
      %99 = tt.load %arg12 cacheModifier = cg : tensor<128x64x!tt.ptr<i8>, #blocked1>
      %100 = ttg.convert_layout %98 : tensor<32x128xi8, #blocked> -> tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %101 = ttg.convert_layout %99 : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %102 = ttg.convert_layout %96 : tensor<32x8xi8, #blocked2> -> tensor<32x8xi8, #linear>
      %103 = ttg.convert_layout %97 : tensor<64x8xi8, #blocked3> -> tensor<64x8xi8, #linear1>
      %104 = tt.dot_scaled %100 scale %102, %101 scale %103, %cst_8 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<64x8xi8, #linear1> -> tensor<32x64xf32, #mma>
      %105 = arith.addf %arg8, %104 : tensor<32x64xf32, #mma>
      %106 = tt.addptr %arg11, %cst_3 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
      %107 = tt.addptr %arg12, %cst_4 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
      %108 = tt.addptr %arg9, %cst_5 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
      %109 = tt.addptr %arg10, %cst_6 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
      scf.yield %105, %108, %109, %106, %107 : tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>
    }
    %68 = arith.truncf %67#0 : tensor<32x64xf32, #mma> to tensor<32x64xbf16, #mma>
    %69 = arith.extsi %10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %70 = arith.extsi %7 : i32 to i64
    %71 = tt.splat %70 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %72 = arith.addi %71, %69 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %73 = arith.extsi %21 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %74 = arith.extsi %19 : i32 to i64
    %75 = tt.splat %74 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %76 = arith.addi %75, %73 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %77 = tt.expand_dims %72 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi64, #blocked4>
    %78 = arith.muli %77, %cst_7 : tensor<32x1xi64, #blocked4>
    %79 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked4>
    %80 = tt.addptr %79, %78 : tensor<32x1x!tt.ptr<bf16>, #blocked4>, tensor<32x1xi64, #blocked4>
    %81 = tt.expand_dims %76 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi64, #blocked4>
    %82 = tt.broadcast %80 : tensor<32x1x!tt.ptr<bf16>, #blocked4> -> tensor<32x64x!tt.ptr<bf16>, #blocked4>
    %83 = tt.broadcast %81 : tensor<1x64xi64, #blocked4> -> tensor<32x64xi64, #blocked4>
    %84 = tt.addptr %82, %83 : tensor<32x64x!tt.ptr<bf16>, #blocked4>, tensor<32x64xi64, #blocked4>
    %85 = arith.extsi %arg5 : i32 to i64
    %86 = tt.splat %85 : i64 -> tensor<32x1xi64, #blocked4>
    %87 = arith.cmpi slt, %77, %86 : tensor<32x1xi64, #blocked4>
    %88 = arith.extsi %arg6 : i32 to i64
    %89 = tt.splat %88 : i64 -> tensor<1x64xi64, #blocked4>
    %90 = arith.cmpi slt, %81, %89 : tensor<1x64xi64, #blocked4>
    %91 = tt.broadcast %87 : tensor<32x1xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %92 = tt.broadcast %90 : tensor<1x64xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %93 = arith.andi %91, %92 : tensor<32x64xi1, #blocked4>
    %94 = ttg.convert_layout %84 : tensor<32x64x!tt.ptr<bf16>, #blocked4> -> tensor<32x64x!tt.ptr<bf16>, #mma>
    %95 = ttg.convert_layout %93 : tensor<32x64xi1, #blocked4> -> tensor<32x64xi1, #mma>
    tt.store %94, %68, %95 : tensor<32x64x!tt.ptr<bf16>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: kernel_m_equal_1
// CHECK: local_load

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_m_equal_1(
      %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<8192> : tensor<1x64xi32, #blocked>
    %cst_0 = arith.constant dense<53248> : tensor<1x8xi32, #blocked1>
    %cst_1 = arith.constant dense<128> : tensor<32x128xi32, #blocked2>
    %cst_2 = arith.constant dense<128> : tensor<128x64xi32, #blocked>
    %cst_3 = arith.constant dense<425984> : tensor<64x8xi32, #blocked1>
    %cst_4 = arith.constant dense<53248> : tensor<32x1xi64, #blocked3>
    %cst_5 = arith.constant dense<1> : tensor<32x1xi64, #blocked3>
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_6 = arith.constant dense<8> : tensor<32x8xi32, #blocked4>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg5, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.cmpi sgt, %3, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = arith.cmpi sgt, %4, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.muli %4, %c64_i32 : i32
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %11 = tt.splat %7 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.splat %7 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %13 = arith.addi %11, %8 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = arith.addi %12, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.splat %arg5 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %16 = tt.splat %arg5 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = arith.remsi %13, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = arith.remsi %14, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %21 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x128x!tt.ptr<i8>, #blocked2>
    %22 = tt.addptr %21, %20 : tensor<1x128x!tt.ptr<i8>, #blocked2>, tensor<1x128xi32, #blocked2>
    %23 = tt.broadcast %22 : tensor<1x128x!tt.ptr<i8>, #blocked2> -> tensor<32x128x!tt.ptr<i8>, #blocked2>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %26 = tt.expand_dims %17 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %27 = arith.muli %26, %cst : tensor<1x64xi32, #blocked>
    %28 = tt.broadcast %25 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %29 = tt.broadcast %27 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %30 = arith.addi %28, %29 : tensor<128x64xi32, #blocked>
    %31 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %32 = tt.addptr %31, %30 : tensor<128x64x!tt.ptr<i8>, #blocked>, tensor<128x64xi32, #blocked>
    %33 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %34 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %35 = tt.expand_dims %33 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %36 = tt.expand_dims %34 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x8xi32, #blocked4>
    %37 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<1x8x!tt.ptr<i8>, #blocked4>
    %38 = tt.addptr %37, %36 : tensor<1x8x!tt.ptr<i8>, #blocked4>, tensor<1x8xi32, #blocked4>
    %39 = tt.broadcast %38 : tensor<1x8x!tt.ptr<i8>, #blocked4> -> tensor<32x8x!tt.ptr<i8>, #blocked4>
    %40 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %41 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<64x1x!tt.ptr<i8>, #blocked1>
    %42 = tt.addptr %41, %40 : tensor<64x1x!tt.ptr<i8>, #blocked1>, tensor<64x1xi32, #blocked1>
    %43 = arith.muli %35, %cst_0 : tensor<1x8xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<64x1x!tt.ptr<i8>, #blocked1> -> tensor<64x8x!tt.ptr<i8>, #blocked1>
    %45 = tt.broadcast %43 : tensor<1x8xi32, #blocked1> -> tensor<64x8xi32, #blocked1>
    %46 = tt.addptr %44, %45 : tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<64x8xi32, #blocked1>
    %47:5 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg7 = %cst_7, %arg8 = %39, %arg9 = %46, %arg10 = %23, %arg11 = %32) -> (tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked4>, tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<32x128x!tt.ptr<i8>, #blocked2>, tensor<128x64x!tt.ptr<i8>, #blocked>)  : i32 {
      %76 = tt.load %arg8 : tensor<32x8x!tt.ptr<i8>, #blocked4>
      %77 = tt.load %arg9 : tensor<64x8x!tt.ptr<i8>, #blocked1>
      %78 = tt.load %arg10 : tensor<32x128x!tt.ptr<i8>, #blocked2>
      %79 = tt.load %arg11 cacheModifier = cg : tensor<128x64x!tt.ptr<i8>, #blocked>
      %80 = ttg.convert_layout %78 : tensor<32x128xi8, #blocked2> -> tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %81 = ttg.convert_layout %79 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %82 = ttg.convert_layout %76 : tensor<32x8xi8, #blocked4> -> tensor<32x8xi8, #linear>
      %83 = ttg.convert_layout %77 : tensor<64x8xi8, #blocked1> -> tensor<64x8xi8, #linear1>
      %84 = tt.dot_scaled %80 scale %82, %81 scale %83, %cst_7 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<64x8xi8, #linear1> -> tensor<32x64xf32, #mma>
      %85 = arith.addf %arg7, %84 : tensor<32x64xf32, #mma>
      %86 = tt.addptr %arg10, %cst_1 : tensor<32x128x!tt.ptr<i8>, #blocked2>, tensor<32x128xi32, #blocked2>
      %87 = tt.addptr %arg11, %cst_2 : tensor<128x64x!tt.ptr<i8>, #blocked>, tensor<128x64xi32, #blocked>
      %88 = tt.addptr %arg8, %cst_6 : tensor<32x8x!tt.ptr<i8>, #blocked4>, tensor<32x8xi32, #blocked4>
      %89 = tt.addptr %arg9, %cst_3 : tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<64x8xi32, #blocked1>
      scf.yield %85, %88, %89, %86, %87 : tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked4>, tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<32x128x!tt.ptr<i8>, #blocked2>, tensor<128x64x!tt.ptr<i8>, #blocked>
    }
    %48 = arith.truncf %47#0 : tensor<32x64xf32, #mma> to tensor<32x64xbf16, #mma>
    %49 = arith.muli %3, %c32_i32 : i32
    %50 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %51 = arith.extsi %50 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %52 = arith.extsi %49 : i32 to i64
    %53 = tt.splat %52 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %54 = arith.addi %53, %51 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %55 = arith.extsi %9 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %56 = arith.extsi %7 : i32 to i64
    %57 = tt.splat %56 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %58 = arith.addi %57, %55 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %59 = tt.expand_dims %54 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi64, #blocked3>
    %60 = arith.muli %59, %cst_4 : tensor<32x1xi64, #blocked3>
    %61 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked3>
    %62 = tt.addptr %61, %60 : tensor<32x1x!tt.ptr<bf16>, #blocked3>, tensor<32x1xi64, #blocked3>
    %63 = tt.expand_dims %58 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi64, #blocked3>
    %64 = tt.broadcast %62 : tensor<32x1x!tt.ptr<bf16>, #blocked3> -> tensor<32x64x!tt.ptr<bf16>, #blocked3>
    %65 = tt.broadcast %63 : tensor<1x64xi64, #blocked3> -> tensor<32x64xi64, #blocked3>
    %66 = tt.addptr %64, %65 : tensor<32x64x!tt.ptr<bf16>, #blocked3>, tensor<32x64xi64, #blocked3>
    %67 = arith.cmpi slt, %59, %cst_5 : tensor<32x1xi64, #blocked3>
    %68 = arith.extsi %arg5 : i32 to i64
    %69 = tt.splat %68 : i64 -> tensor<1x64xi64, #blocked3>
    %70 = arith.cmpi slt, %63, %69 : tensor<1x64xi64, #blocked3>
    %71 = tt.broadcast %67 : tensor<32x1xi1, #blocked3> -> tensor<32x64xi1, #blocked3>
    %72 = tt.broadcast %70 : tensor<1x64xi1, #blocked3> -> tensor<32x64xi1, #blocked3>
    %73 = arith.andi %71, %72 : tensor<32x64xi1, #blocked3>
    %74 = ttg.convert_layout %66 : tensor<32x64x!tt.ptr<bf16>, #blocked3> -> tensor<32x64x!tt.ptr<bf16>, #mma>
    %75 = ttg.convert_layout %73 : tensor<32x64xi1, #blocked3> -> tensor<32x64xi1, #mma>
    tt.store %74, %48, %75 : tensor<32x64x!tt.ptr<bf16>, #mma>
    tt.return
  }
}

// -----

// FACTOR2-LABEL: kernel_with_outer_loop
// FACTOR2: [[LOW_BOUND_2:%.*]] = arith.constant 0 : i32
// FACTOR2: [[LOW_BOUND_1:%.*]] = arith.constant 0 : i32
// FACTOR2-DAG: [[UPPER_BOUND_1:%.*]] = arith.constant 128 : i32
// FACTOR2-DAG: [[UPPER_BOUND_2:%.*]] = arith.constant 2 : i32
// FACTOR2: scf.for {{.*}} [[LOW_BOUND_1]] to [[UPPER_BOUND_1]]
// FACTOR2: scf.for {{.*}} [[LOW_BOUND_2]] to [[UPPER_BOUND_2]]
// FACTOR2: local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_with_outer_loop(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32768> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<32768> : tensor<1x64xi32, #blocked1>
    %cst_1 = arith.constant dense<9728> : tensor<1x8xi32, #blocked2>
    %cst_2 = arith.constant dense<8192> : tensor<1x8xi32, #blocked3>
    %cst_3 = arith.constant dense<128> : tensor<32x128xi32, #blocked>
    %cst_4 = arith.constant dense<128> : tensor<128x64xi32, #blocked1>
    %cst_5 = arith.constant dense<77824> : tensor<32x8xi32, #blocked2>
    %cst_6 = arith.constant dense<65536> : tensor<64x8xi32, #blocked3>
    %cst_7 = arith.constant dense<8192> : tensor<32x1xi64, #blocked4>
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c256_i32 = arith.constant 256 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.cmpi sgt, %3, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = arith.cmpi sgt, %4, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.muli %3, %c32_i32 : i32
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %11 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = arith.addi %11, %8 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.addi %12, %9 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %15 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = arith.remsi %13, %15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.remsi %14, %16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = arith.muli %4, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %23 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %25 = arith.addi %23, %20 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %24, %22 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %27 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %29 = arith.remsi %25, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %30 = arith.remsi %26, %28 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %31 = tt.expand_dims %17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %32 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %33 = arith.muli %31, %cst : tensor<32x1xi32, #blocked>
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %36 = tt.broadcast %33 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %38 = arith.addi %36, %37 : tensor<32x128xi32, #blocked>
    %39 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>, #blocked>
    %40 = tt.addptr %39, %38 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
    %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %43 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %44 = arith.muli %43, %cst_0 : tensor<1x64xi32, #blocked1>
    %45 = tt.broadcast %42 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %47 = arith.addi %45, %46 : tensor<128x64xi32, #blocked1>
    %48 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked1>
    %49 = tt.addptr %48, %47 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
    %50 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked2>
    %51 = tt.addptr %50, %32 : tensor<32x1x!tt.ptr<i8>, #blocked2>, tensor<32x1xi32, #blocked2>
    %52 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %53 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %54 = tt.expand_dims %52 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %55 = tt.expand_dims %53 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x8xi32, #blocked3>
    %56 = arith.muli %54, %cst_1 : tensor<1x8xi32, #blocked2>
    %57 = tt.broadcast %51 : tensor<32x1x!tt.ptr<i8>, #blocked2> -> tensor<32x8x!tt.ptr<i8>, #blocked2>
    %58 = tt.broadcast %56 : tensor<1x8xi32, #blocked2> -> tensor<32x8xi32, #blocked2>
    %59 = tt.addptr %57, %58 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
    %60 = tt.expand_dims %30 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %61 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<64x1x!tt.ptr<i8>, #blocked3>
    %62 = tt.addptr %61, %60 : tensor<64x1x!tt.ptr<i8>, #blocked3>, tensor<64x1xi32, #blocked3>
    %63 = arith.muli %55, %cst_2 : tensor<1x8xi32, #blocked3>
    %64 = tt.broadcast %62 : tensor<64x1x!tt.ptr<i8>, #blocked3> -> tensor<64x8x!tt.ptr<i8>, #blocked3>
    %65 = tt.broadcast %63 : tensor<1x8xi32, #blocked3> -> tensor<64x8xi32, #blocked3>
    %66 = tt.addptr %64, %65 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
    %67:5 = scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32 iter_args(%arg8 = %cst_8, %arg9 = %59, %arg10 = %66, %arg11 = %40, %arg12 = %49) -> (tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>)  : i32 {
      %96 = tt.load %arg9 : tensor<32x8x!tt.ptr<i8>, #blocked2>
      %97 = tt.load %arg10 : tensor<64x8x!tt.ptr<i8>, #blocked3>
      %98 = tt.load %arg11 : tensor<32x128x!tt.ptr<i8>, #blocked>
      %99 = tt.load %arg12 cacheModifier = cg : tensor<128x64x!tt.ptr<i8>, #blocked1>
      %100 = ttg.convert_layout %98 : tensor<32x128xi8, #blocked> -> tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %101 = ttg.convert_layout %99 : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %102 = ttg.convert_layout %96 : tensor<32x8xi8, #blocked2> -> tensor<32x8xi8, #linear>
      %103 = ttg.convert_layout %97 : tensor<64x8xi8, #blocked3> -> tensor<64x8xi8, #linear1>
      %104 = tt.dot_scaled %100 scale %102, %101 scale %103, %cst_8 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<64x8xi8, #linear1> -> tensor<32x64xf32, #mma>
      %105 = arith.addf %arg8, %104 : tensor<32x64xf32, #mma>
      %106 = tt.addptr %arg11, %cst_3 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
      %107 = tt.addptr %arg12, %cst_4 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
      %108 = tt.addptr %arg9, %cst_5 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
      %109 = tt.addptr %arg10, %cst_6 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
      scf.yield %105, %108, %109, %106, %107 : tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>
    }
    %68 = arith.truncf %67#0 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
    %69 = arith.extsi %10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %70 = arith.extsi %7 : i32 to i64
    %71 = tt.splat %70 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %72 = arith.addi %71, %69 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %73 = arith.extsi %21 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %74 = arith.extsi %19 : i32 to i64
    %75 = tt.splat %74 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %76 = arith.addi %75, %73 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %77 = tt.expand_dims %72 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi64, #blocked4>
    %78 = arith.muli %77, %cst_7 : tensor<32x1xi64, #blocked4>
    %79 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked4>
    %80 = tt.addptr %79, %78 : tensor<32x1x!tt.ptr<f16>, #blocked4>, tensor<32x1xi64, #blocked4>
    %81 = tt.expand_dims %76 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi64, #blocked4>
    %82 = tt.broadcast %80 : tensor<32x1x!tt.ptr<f16>, #blocked4> -> tensor<32x64x!tt.ptr<f16>, #blocked4>
    %83 = tt.broadcast %81 : tensor<1x64xi64, #blocked4> -> tensor<32x64xi64, #blocked4>
    %84 = tt.addptr %82, %83 : tensor<32x64x!tt.ptr<f16>, #blocked4>, tensor<32x64xi64, #blocked4>
    %85 = arith.extsi %arg5 : i32 to i64
    %86 = tt.splat %85 : i64 -> tensor<32x1xi64, #blocked4>
    %87 = arith.cmpi slt, %77, %86 : tensor<32x1xi64, #blocked4>
    %88 = arith.extsi %arg6 : i32 to i64
    %89 = tt.splat %88 : i64 -> tensor<1x64xi64, #blocked4>
    %90 = arith.cmpi slt, %81, %89 : tensor<1x64xi64, #blocked4>
    %91 = tt.broadcast %87 : tensor<32x1xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %92 = tt.broadcast %90 : tensor<1x64xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %93 = arith.andi %91, %92 : tensor<32x64xi1, #blocked4>
    %94 = ttg.convert_layout %84 : tensor<32x64x!tt.ptr<f16>, #blocked4> -> tensor<32x64x!tt.ptr<f16>, #mma>
    %95 = ttg.convert_layout %93 : tensor<32x64xi1, #blocked4> -> tensor<32x64xi1, #mma>
    tt.store %94, %68, %95 : tensor<32x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: kernel_upper_bound_nondividable_by_factor_neg
// FACTOR2-LABEL: kernel_upper_bound_nondividable_by_factor_neg
// CHECK-NOT: local_load
// FACTOR2-NOT: local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_upper_bound_nondividable_by_factor_neg(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32768> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<32768> : tensor<1x64xi32, #blocked1>
    %cst_1 = arith.constant dense<9728> : tensor<1x8xi32, #blocked2>
    %cst_2 = arith.constant dense<8192> : tensor<1x8xi32, #blocked3>
    %cst_3 = arith.constant dense<128> : tensor<32x128xi32, #blocked>
    %cst_4 = arith.constant dense<128> : tensor<128x64xi32, #blocked1>
    %cst_5 = arith.constant dense<77824> : tensor<32x8xi32, #blocked2>
    %cst_6 = arith.constant dense<65536> : tensor<64x8xi32, #blocked3>
    %cst_7 = arith.constant dense<8192> : tensor<32x1xi64, #blocked4>
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c256_i32 = arith.constant 256 : i32
    %c3_i32 = arith.constant 3 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.cmpi sgt, %3, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = arith.cmpi sgt, %4, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.muli %3, %c32_i32 : i32
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %11 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.splat %7 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = arith.addi %11, %8 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.addi %12, %9 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %15 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.splat %arg5 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = arith.remsi %13, %15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.remsi %14, %16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = arith.muli %4, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %23 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %25 = arith.addi %23, %20 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %24, %22 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %27 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %29 = arith.remsi %25, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %30 = arith.remsi %26, %28 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %31 = tt.expand_dims %17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %32 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %33 = arith.muli %31, %cst : tensor<32x1xi32, #blocked>
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %36 = tt.broadcast %33 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %38 = arith.addi %36, %37 : tensor<32x128xi32, #blocked>
    %39 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>, #blocked>
    %40 = tt.addptr %39, %38 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
    %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %43 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %44 = arith.muli %43, %cst_0 : tensor<1x64xi32, #blocked1>
    %45 = tt.broadcast %42 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %47 = arith.addi %45, %46 : tensor<128x64xi32, #blocked1>
    %48 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked1>
    %49 = tt.addptr %48, %47 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
    %50 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked2>
    %51 = tt.addptr %50, %32 : tensor<32x1x!tt.ptr<i8>, #blocked2>, tensor<32x1xi32, #blocked2>
    %52 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %53 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %54 = tt.expand_dims %52 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %55 = tt.expand_dims %53 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x8xi32, #blocked3>
    %56 = arith.muli %54, %cst_1 : tensor<1x8xi32, #blocked2>
    %57 = tt.broadcast %51 : tensor<32x1x!tt.ptr<i8>, #blocked2> -> tensor<32x8x!tt.ptr<i8>, #blocked2>
    %58 = tt.broadcast %56 : tensor<1x8xi32, #blocked2> -> tensor<32x8xi32, #blocked2>
    %59 = tt.addptr %57, %58 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
    %60 = tt.expand_dims %30 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %61 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<64x1x!tt.ptr<i8>, #blocked3>
    %62 = tt.addptr %61, %60 : tensor<64x1x!tt.ptr<i8>, #blocked3>, tensor<64x1xi32, #blocked3>
    %63 = arith.muli %55, %cst_2 : tensor<1x8xi32, #blocked3>
    %64 = tt.broadcast %62 : tensor<64x1x!tt.ptr<i8>, #blocked3> -> tensor<64x8x!tt.ptr<i8>, #blocked3>
    %65 = tt.broadcast %63 : tensor<1x8xi32, #blocked3> -> tensor<64x8xi32, #blocked3>
    %66 = tt.addptr %64, %65 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
    %67:5 = scf.for %arg7 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg8 = %cst_8, %arg9 = %59, %arg10 = %66, %arg11 = %40, %arg12 = %49) -> (tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>)  : i32 {
      %96 = tt.load %arg9 : tensor<32x8x!tt.ptr<i8>, #blocked2>
      %97 = tt.load %arg10 : tensor<64x8x!tt.ptr<i8>, #blocked3>
      %98 = tt.load %arg11 : tensor<32x128x!tt.ptr<i8>, #blocked>
      %99 = tt.load %arg12 cacheModifier = cg : tensor<128x64x!tt.ptr<i8>, #blocked1>
      %100 = ttg.convert_layout %98 : tensor<32x128xi8, #blocked> -> tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %101 = ttg.convert_layout %99 : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %102 = ttg.convert_layout %96 : tensor<32x8xi8, #blocked2> -> tensor<32x8xi8, #linear>
      %103 = ttg.convert_layout %97 : tensor<64x8xi8, #blocked3> -> tensor<64x8xi8, #linear1>
      %104 = tt.dot_scaled %100 scale %102, %101 scale %103, %cst_8 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<64x8xi8, #linear1> -> tensor<32x64xf32, #mma>
      %105 = arith.addf %arg8, %104 : tensor<32x64xf32, #mma>
      %106 = tt.addptr %arg11, %cst_3 : tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<32x128xi32, #blocked>
      %107 = tt.addptr %arg12, %cst_4 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
      %108 = tt.addptr %arg9, %cst_5 : tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<32x8xi32, #blocked2>
      %109 = tt.addptr %arg10, %cst_6 : tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<64x8xi32, #blocked3>
      scf.yield %105, %108, %109, %106, %107 : tensor<32x64xf32, #mma>, tensor<32x8x!tt.ptr<i8>, #blocked2>, tensor<64x8x!tt.ptr<i8>, #blocked3>, tensor<32x128x!tt.ptr<i8>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>
    }
    %68 = arith.truncf %67#0 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
    %69 = arith.extsi %10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %70 = arith.extsi %7 : i32 to i64
    %71 = tt.splat %70 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %72 = arith.addi %71, %69 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %73 = arith.extsi %21 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %74 = arith.extsi %19 : i32 to i64
    %75 = tt.splat %74 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %76 = arith.addi %75, %73 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %77 = tt.expand_dims %72 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi64, #blocked4>
    %78 = arith.muli %77, %cst_7 : tensor<32x1xi64, #blocked4>
    %79 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked4>
    %80 = tt.addptr %79, %78 : tensor<32x1x!tt.ptr<f16>, #blocked4>, tensor<32x1xi64, #blocked4>
    %81 = tt.expand_dims %76 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi64, #blocked4>
    %82 = tt.broadcast %80 : tensor<32x1x!tt.ptr<f16>, #blocked4> -> tensor<32x64x!tt.ptr<f16>, #blocked4>
    %83 = tt.broadcast %81 : tensor<1x64xi64, #blocked4> -> tensor<32x64xi64, #blocked4>
    %84 = tt.addptr %82, %83 : tensor<32x64x!tt.ptr<f16>, #blocked4>, tensor<32x64xi64, #blocked4>
    %85 = arith.extsi %arg5 : i32 to i64
    %86 = tt.splat %85 : i64 -> tensor<32x1xi64, #blocked4>
    %87 = arith.cmpi slt, %77, %86 : tensor<32x1xi64, #blocked4>
    %88 = arith.extsi %arg6 : i32 to i64
    %89 = tt.splat %88 : i64 -> tensor<1x64xi64, #blocked4>
    %90 = arith.cmpi slt, %81, %89 : tensor<1x64xi64, #blocked4>
    %91 = tt.broadcast %87 : tensor<32x1xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %92 = tt.broadcast %90 : tensor<1x64xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
    %93 = arith.andi %91, %92 : tensor<32x64xi1, #blocked4>
    %94 = ttg.convert_layout %84 : tensor<32x64x!tt.ptr<f16>, #blocked4> -> tensor<32x64x!tt.ptr<f16>, #mma>
    %95 = ttg.convert_layout %93 : tensor<32x64xi1, #blocked4> -> tensor<32x64xi1, #mma>
    tt.store %94, %68, %95 : tensor<32x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----


// CHECK-LABEL: kernel_non_constant_stride
// FACTOR2-LABEL: kernel_non_constant_stride
// FACTOR2: local_load
// CHECK: local_load

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_non_constant_stride(
      %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32},
      %arg8: i32 {tt.divisibility = 16 : i32},
      %arg9: i32 {tt.divisibility = 16 : i32},
      %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_1 = arith.constant dense<106496> : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %cst_2 = arith.constant dense<106496> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %cst_3 = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %cst_4 = arith.constant dense<128> : tensor<128x256xi32, #blocked3>
    %cst_5 = arith.constant dense<128> : tensor<128x1xi64, #blocked4>
    %cst_6 = arith.constant dense<106496> : tensor<1x256xi64, #blocked4>
    %c416_i32 = arith.constant 416 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %1 = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sgt, %arg9, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    %4 = arith.cmpi sgt, %arg10, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    llvm.intr.assume %true : i1
    %5 = tt.get_program_id x : i32
    %6 = arith.divsi %5, %c416_i32 : i32
    %7 = arith.remsi %5, %c416_i32 : i32
    %8 = arith.cmpi sgt, %6, %c0_i32 : i32
    llvm.intr.assume %8 : i1
    %9 = arith.cmpi sgt, %7, %c0_i32 : i32
    llvm.intr.assume %9 : i1
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %13 = arith.muli %6, %c128_i32 : i32
    %14 = tt.splat %13 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.splat %13 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.addi %14, %10 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = arith.addi %15, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.remsi %16, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.remsi %17, %cst : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.muli %7, %c256_i32 : i32
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %24 = tt.splat %20 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %25 = tt.splat %20 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %26 = arith.addi %24, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %27 = arith.addi %25, %23 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %28 = arith.remsi %26, %cst_2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %29 = arith.remsi %27, %cst_1 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %30 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %31 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %32 = tt.splat %arg5 : i32 -> tensor<128x1xi32, #blocked1>
    %33 = arith.muli %30, %32 : tensor<128x1xi32, #blocked1>
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %36 = tt.broadcast %33 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %38 = arith.addi %36, %37 : tensor<128x128xi32, #blocked1>
    %39 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
    %40 = tt.addptr %39, %38 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
    %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %44 = tt.splat %arg6 : i32 -> tensor<1x256xi32, #blocked3>
    %45 = arith.muli %43, %44 : tensor<1x256xi32, #blocked3>
    %46 = tt.broadcast %42 : tensor<128x1xi32, #blocked3> -> tensor<128x256xi32, #blocked3>
    %47 = tt.broadcast %45 : tensor<1x256xi32, #blocked3> -> tensor<128x256xi32, #blocked3>
    %48 = arith.addi %46, %47 : tensor<128x256xi32, #blocked3>
    %49 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>, #blocked3>
    %50 = tt.addptr %49, %48 : tensor<128x256x!tt.ptr<i8>, #blocked3>, tensor<128x256xi32, #blocked3>
    %51 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
    %52 = tt.addptr %51, %31 : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi32, #blocked>
    %53 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %54 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %55 = tt.expand_dims %53 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %56 = tt.expand_dims %54 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %57 = tt.splat %arg9 : i32 -> tensor<1x8xi32, #blocked>
    %58 = arith.muli %55, %57 : tensor<1x8xi32, #blocked>
    %59 = tt.broadcast %52 : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x8x!tt.ptr<i8>, #blocked>
    %60 = tt.broadcast %58 : tensor<1x8xi32, #blocked> -> tensor<128x8xi32, #blocked>
    %61 = tt.addptr %59, %60 : tensor<128x8x!tt.ptr<i8>, #blocked>, tensor<128x8xi32, #blocked>
    %62 = tt.expand_dims %29 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<256x1xi32, #blocked2>
    %63 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked2>
    %64 = tt.addptr %63, %62 : tensor<256x1x!tt.ptr<i8>, #blocked2>, tensor<256x1xi32, #blocked2>
    %65 = tt.splat %arg10 : i32 -> tensor<1x8xi32, #blocked2>
    %66 = arith.muli %56, %65 : tensor<1x8xi32, #blocked2>
    %67 = tt.broadcast %64 : tensor<256x1x!tt.ptr<i8>, #blocked2> -> tensor<256x8x!tt.ptr<i8>, #blocked2>
    %68 = tt.broadcast %66 : tensor<1x8xi32, #blocked2> -> tensor<256x8xi32, #blocked2>
    %69 = tt.addptr %67, %68 : tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<256x8xi32, #blocked2>
    %70 = arith.muli %arg9, %c8_i32 : i32
    %71 = tt.splat %70 : i32 -> tensor<128x8xi32, #blocked>
    %72 = arith.muli %arg10, %c8_i32 : i32
    %73 = tt.splat %72 : i32 -> tensor<256x8xi32, #blocked2>
    %74:5 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg12 = %cst_7, %arg13 = %61, %arg14 = %69, %arg15 = %40, %arg16 = %50) -> (tensor<128x256xf32, #mma>, tensor<128x8x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x256x!tt.ptr<i8>, #blocked3>)  : i32 {
      %101 = tt.load %arg13 : tensor<128x8x!tt.ptr<i8>, #blocked>
      %102 = tt.load %arg14 : tensor<256x8x!tt.ptr<i8>, #blocked2>
      %103 = tt.load %arg15 : tensor<128x128x!tt.ptr<i8>, #blocked1>
      %104 = tt.load %arg16 cacheModifier = ca : tensor<128x256x!tt.ptr<i8>, #blocked3>
      %105 = ttg.convert_layout %103 : tensor<128x128xi8, #blocked1> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %106 = ttg.convert_layout %104 : tensor<128x256xi8, #blocked3> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %107 = ttg.convert_layout %101 : tensor<128x8xi8, #blocked> -> tensor<128x8xi8, #linear>
      %108 = ttg.convert_layout %102 : tensor<256x8xi8, #blocked2> -> tensor<256x8xi8, #linear1>
      %109 = tt.dot_scaled %105 scale %107, %106 scale %108, %arg12 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<128x256xf32, #mma>
      %110 = tt.addptr %arg15, %cst_3 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
      %111 = tt.addptr %arg16, %cst_4 : tensor<128x256x!tt.ptr<i8>, #blocked3>, tensor<128x256xi32, #blocked3>
      %112 = tt.addptr %arg13, %71 : tensor<128x8x!tt.ptr<i8>, #blocked>, tensor<128x8xi32, #blocked>
      %113 = tt.addptr %arg14, %73 : tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<256x8xi32, #blocked2>
      scf.yield %109, %112, %113, %110, %111 : tensor<128x256xf32, #mma>, tensor<128x8x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked2>, tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x256x!tt.ptr<i8>, #blocked3>
    }
    %75 = arith.truncf %74#0 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
    %76 = arith.extsi %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %77 = arith.extsi %13 : i32 to i64
    %78 = tt.splat %77 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %79 = arith.addi %78, %76 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %80 = arith.extsi %22 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %81 = arith.extsi %20 : i32 to i64
    %82 = tt.splat %81 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %83 = arith.addi %82, %80 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %84 = tt.expand_dims %79 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi64, #blocked4>
    %85 = arith.extsi %arg8 : i32 to i64
    %86 = tt.splat %85 : i64 -> tensor<128x1xi64, #blocked4>
    %87 = arith.muli %86, %84 : tensor<128x1xi64, #blocked4>
    %88 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked4>
    %89 = tt.addptr %88, %87 : tensor<128x1x!tt.ptr<bf16>, #blocked4>, tensor<128x1xi64, #blocked4>
    %90 = tt.expand_dims %83 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x256xi64, #blocked4>
    %91 = tt.broadcast %89 : tensor<128x1x!tt.ptr<bf16>, #blocked4> -> tensor<128x256x!tt.ptr<bf16>, #blocked4>
    %92 = tt.broadcast %90 : tensor<1x256xi64, #blocked4> -> tensor<128x256xi64, #blocked4>
    %93 = tt.addptr %91, %92 : tensor<128x256x!tt.ptr<bf16>, #blocked4>, tensor<128x256xi64, #blocked4>
    %94 = arith.cmpi slt, %84, %cst_5 : tensor<128x1xi64, #blocked4>
    %95 = arith.cmpi slt, %90, %cst_6 : tensor<1x256xi64, #blocked4>
    %96 = tt.broadcast %94 : tensor<128x1xi1, #blocked4> -> tensor<128x256xi1, #blocked4>
    %97 = tt.broadcast %95 : tensor<1x256xi1, #blocked4> -> tensor<128x256xi1, #blocked4>
    %98 = arith.andi %96, %97 : tensor<128x256xi1, #blocked4>
    %99 = ttg.convert_layout %93 : tensor<128x256x!tt.ptr<bf16>, #blocked4> -> tensor<128x256x!tt.ptr<bf16>, #mma>
    %100 = ttg.convert_layout %98 : tensor<128x256xi1, #blocked4> -> tensor<128x256xi1, #mma>
    tt.store %99, %75, %100 : tensor<128x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: preshuffled_kernel
// CHECK: local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [4, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[1, 0], [2, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [128, 0]], lane = [[0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], tilesPerWarp = [2, 2], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @preshuffled_kernel(
      %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32},
      %arg8: i32 {tt.divisibility = 16 : i32},
      %arg9: i32 {tt.divisibility = 16 : i32},
      %arg10: i32 {tt.divisibility = 16 : i32},
      %arg11: i32 {tt.divisibility = 16 : i32},
      %arg12: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x256xi32, #blocked1>
    %cst_1 = arith.constant dense<256> : tensor<4x256xi32, #blocked2>
    %cst_2 = arith.constant dense<256> : tensor<8x256xi32, #blocked3>
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %1 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sgt, %arg10, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sgt, %arg11, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %4 = arith.cmpi sgt, %arg12, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    %5 = tt.get_program_id x : i32
    %6 = arith.addi %arg6, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.divsi %5, %7 : i32
    %9 = arith.remsi %5, %7 : i32
    %10 = arith.cmpi sgt, %8, %c0_i32 : i32
    llvm.intr.assume %10 : i1
    %11 = arith.cmpi sgt, %9, %c0_i32 : i32
    llvm.intr.assume %11 : i1
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %14 = arith.muli %8, %c128_i32 : i32
    %15 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.addi %15, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = arith.remsi %16, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = arith.muli %9, %c256_i32 : i32
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %22 = tt.splat %19 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %23 = arith.addi %22, %20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = tt.splat %arg6 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = arith.remsi %23, %24 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %27 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
    %28 = arith.muli %26, %27 : tensor<128x1xi32, #blocked>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %31 = tt.broadcast %28 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %32 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %33 = arith.addi %31, %32 : tensor<128x128xi32, #blocked>
    %34 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked>
    %35 = tt.addptr %34, %33 : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi32, #blocked>
    %36 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %38 = tt.expand_dims %25 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %39 = tt.splat %arg8 : i32 -> tensor<1x256xi32, #blocked1>
    %40 = arith.muli %38, %39 : tensor<1x256xi32, #blocked1>
    %41 = tt.broadcast %37 : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %42 = tt.broadcast %40 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %43 = arith.addi %41, %42 : tensor<128x256xi32, #blocked1>
    %44 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>, #blocked1>
    %45 = tt.addptr %44, %43 : tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<128x256xi32, #blocked1>
    %46 = arith.muli %9, %c8_i32 : i32
    %47 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %48 = tt.splat %46 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %49 = arith.addi %48, %47 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %50 = tt.splat %arg6 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %51 = arith.remsi %49, %50 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<8x1xi32, #blocked3>
    %53 = tt.splat %arg12 : i32 -> tensor<8x1xi32, #blocked3>
    %54 = arith.muli %52, %53 : tensor<8x1xi32, #blocked3>
    %55 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<8x1x!tt.ptr<i8>, #blocked3>
    %56 = tt.addptr %55, %54 : tensor<8x1x!tt.ptr<i8>, #blocked3>, tensor<8x1xi32, #blocked3>
    %57 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %58 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %59 = tt.expand_dims %57 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %60 = tt.expand_dims %58 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %61 = tt.broadcast %56 : tensor<8x1x!tt.ptr<i8>, #blocked3> -> tensor<8x256x!tt.ptr<i8>, #blocked3>
    %62 = tt.broadcast %60 : tensor<1x256xi32, #blocked3> -> tensor<8x256xi32, #blocked3>
    %63 = tt.addptr %61, %62 : tensor<8x256x!tt.ptr<i8>, #blocked3>, tensor<8x256xi32, #blocked3>
    %64 = arith.muli %8, %c4_i32 : i32
    %65 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %66 = tt.splat %64 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %67 = arith.addi %66, %65 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %68 = tt.splat %arg5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %69 = arith.remsi %67, %68 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %70 = tt.expand_dims %69 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<4x1xi32, #blocked2>
    %71 = tt.splat %arg11 : i32 -> tensor<4x1xi32, #blocked2>
    %72 = arith.muli %70, %71 : tensor<4x1xi32, #blocked2>
    %73 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<4x1x!tt.ptr<i8>, #blocked2>
    %74 = tt.addptr %73, %72 : tensor<4x1x!tt.ptr<i8>, #blocked2>, tensor<4x1xi32, #blocked2>
    %75 = tt.broadcast %74 : tensor<4x1x!tt.ptr<i8>, #blocked2> -> tensor<4x256x!tt.ptr<i8>, #blocked2>
    %76 = tt.broadcast %59 : tensor<1x256xi32, #blocked2> -> tensor<4x256xi32, #blocked2>
    %77 = tt.addptr %75, %76 : tensor<4x256x!tt.ptr<i8>, #blocked2>, tensor<4x256xi32, #blocked2>
    %78:5 = scf.for %arg13 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg14 = %cst_3, %arg15 = %77, %arg16 = %63, %arg17 = %35, %arg18 = %45) -> (tensor<128x256xf32, #mma>, tensor<4x256x!tt.ptr<i8>, #blocked2>, tensor<8x256x!tt.ptr<i8>, #blocked3>, tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked1>)  : i32 {
      %109 = tt.load %arg15 : tensor<4x256x!tt.ptr<i8>, #blocked2>
      %110 = ttg.convert_layout %109 : tensor<4x256xi8, #blocked2> -> tensor<4x256xi8, #linear>
      %111 = tt.load %arg16 : tensor<8x256x!tt.ptr<i8>, #blocked3>
      %112 = ttg.convert_layout %111 : tensor<8x256xi8, #blocked3> -> tensor<8x256xi8, #linear1>
      %113 = tt.reshape %110 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
      %114 = tt.reshape %112 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
      %115 = tt.load %arg17 : tensor<128x128x!tt.ptr<i8>, #blocked>
      %116 = tt.load %arg18 cacheModifier = cg : tensor<128x256x!tt.ptr<i8>, #blocked1>
      %117 = ttg.convert_layout %115 : tensor<128x128xi8, #blocked> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %118 = ttg.convert_layout %116 : tensor<128x256xi8, #blocked1> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %119 = tt.dot_scaled %117 scale %113, %118 scale %114, %arg14 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
      %120 = tt.addptr %arg17, %cst : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi32, #blocked>
      %121 = tt.addptr %arg18, %cst_0 : tensor<128x256x!tt.ptr<i8>, #blocked1>, tensor<128x256xi32, #blocked1>
      %122 = tt.addptr %arg15, %cst_1 : tensor<4x256x!tt.ptr<i8>, #blocked2>, tensor<4x256xi32, #blocked2>
      %123 = tt.addptr %arg16, %cst_2 : tensor<8x256x!tt.ptr<i8>, #blocked3>, tensor<8x256xi32, #blocked3>
      scf.yield %119, %122, %123, %120, %121 : tensor<128x256xf32, #mma>, tensor<4x256x!tt.ptr<i8>, #blocked2>, tensor<8x256x!tt.ptr<i8>, #blocked3>, tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked1>
    }
    %79 = arith.truncf %78#0 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
    %80 = arith.extsi %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %81 = arith.extsi %14 : i32 to i64
    %82 = tt.splat %81 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %83 = arith.addi %82, %80 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %84 = arith.extsi %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %85 = arith.extsi %19 : i32 to i64
    %86 = tt.splat %85 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %87 = arith.addi %86, %84 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %88 = tt.expand_dims %83 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi64, #blocked3>
    %89 = arith.extsi %arg10 : i32 to i64
    %90 = tt.splat %89 : i64 -> tensor<128x1xi64, #blocked3>
    %91 = arith.muli %90, %88 : tensor<128x1xi64, #blocked3>
    %92 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked3>
    %93 = tt.addptr %92, %91 : tensor<128x1x!tt.ptr<bf16>, #blocked3>, tensor<128x1xi64, #blocked3>
    %94 = tt.expand_dims %87 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi64, #blocked3>
    %95 = tt.broadcast %93 : tensor<128x1x!tt.ptr<bf16>, #blocked3> -> tensor<128x256x!tt.ptr<bf16>, #blocked3>
    %96 = tt.broadcast %94 : tensor<1x256xi64, #blocked3> -> tensor<128x256xi64, #blocked3>
    %97 = tt.addptr %95, %96 : tensor<128x256x!tt.ptr<bf16>, #blocked3>, tensor<128x256xi64, #blocked3>
    %98 = arith.extsi %arg5 : i32 to i64
    %99 = tt.splat %98 : i64 -> tensor<128x1xi64, #blocked3>
    %100 = arith.cmpi slt, %88, %99 : tensor<128x1xi64, #blocked3>
    %101 = arith.extsi %arg6 : i32 to i64
    %102 = tt.splat %101 : i64 -> tensor<1x256xi64, #blocked3>
    %103 = arith.cmpi slt, %94, %102 : tensor<1x256xi64, #blocked3>
    %104 = tt.broadcast %100 : tensor<128x1xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %105 = tt.broadcast %103 : tensor<1x256xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %106 = arith.andi %104, %105 : tensor<128x256xi1, #blocked3>
    %107 = ttg.convert_layout %97 : tensor<128x256x!tt.ptr<bf16>, #blocked3> -> tensor<128x256x!tt.ptr<bf16>, #mma>
    %108 = ttg.convert_layout %106 : tensor<128x256xi1, #blocked3> -> tensor<128x256xi1, #mma>
    tt.store %107, %79, %108 : tensor<128x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}
