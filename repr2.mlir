#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_gemm_afp4_wfp4_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<53248> : tensor<32x1xi64, #blocked>
    %cst_0 = arith.constant dense<425984> : tensor<64x8xi32, #blocked1>
    %cst_1 = arith.constant dense<256> : tensor<32x8xi32, #blocked1>
    %cst_2 = arith.constant dense<128> : tensor<128x64xi32, #blocked2>
    %cst_3 = arith.constant dense<128> : tensor<32x128xi32, #blocked3>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked2>
    %cst_5 = arith.constant dense<53248> : tensor<1x8xi32, #blocked1>
    %cst_6 = arith.constant dense<32> : tensor<1x8xi32, #blocked1>
    %cst_7 = arith.constant dense<8192> : tensor<1x64xi32, #blocked2>
    %cst_8 = arith.constant dense<8192> : tensor<32x1xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
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
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked4>
    %8 = arith.muli %3, %c32_i32 : i32
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked4>
    %10 = tt.splat %8 : i32 -> tensor<32xi32, #blocked4>
    %11 = arith.addi %10, %9 : tensor<32xi32, #blocked4>
    %12 = tt.splat %arg5 : i32 -> tensor<32xi32, #blocked4>
    %13 = arith.remsi %11, %12 : tensor<32xi32, #blocked4>
    %14 = arith.muli %4, %c64_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %16 = tt.splat %14 : i32 -> tensor<64xi32, #blocked4>
    %17 = arith.addi %16, %15 : tensor<64xi32, #blocked4>
    %18 = tt.splat %arg6 : i32 -> tensor<64xi32, #blocked4>
    %19 = arith.remsi %17, %18 : tensor<64xi32, #blocked4>
    %20 = ttg.convert_layout %13 : tensor<32xi32, #blocked4> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %22 = ttg.convert_layout %21 : tensor<32x1xi32, #blocked5> -> tensor<32x1xi32, #blocked>
    %23 = arith.muli %22, %cst_8 : tensor<32x1xi32, #blocked>
    %24 = ttg.convert_layout %7 : tensor<128xi32, #blocked4> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x128xi32, #blocked6>
    %26 = ttg.convert_layout %25 : tensor<1x128xi32, #blocked6> -> tensor<1x128xi32, #blocked3>
    %27 = tt.broadcast %23 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %28 = ttg.convert_layout %27 : tensor<32x128xi32, #blocked> -> tensor<32x128xi32, #blocked3>
    %29 = tt.broadcast %26 : tensor<1x128xi32, #blocked3> -> tensor<32x128xi32, #blocked3>
    %30 = arith.addi %28, %29 : tensor<32x128xi32, #blocked3>
    %31 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>, #blocked3>
    %32 = tt.addptr %31, %30 : tensor<32x128x!tt.ptr<i8>, #blocked3>, tensor<32x128xi32, #blocked3>
    %33 = ttg.convert_layout %7 : tensor<128xi32, #blocked4> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<128x1xi32, #blocked5>
    %35 = ttg.convert_layout %34 : tensor<128x1xi32, #blocked5> -> tensor<128x1xi32, #blocked>
    %36 = ttg.convert_layout %19 : tensor<64xi32, #blocked4> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %37 = tt.expand_dims %36 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xi32, #blocked6>
    %38 = ttg.convert_layout %37 : tensor<1x64xi32, #blocked6> -> tensor<1x64xi32, #blocked2>
    %39 = arith.muli %38, %cst_7 : tensor<1x64xi32, #blocked2>
    %40 = tt.broadcast %35 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %41 = ttg.convert_layout %40 : tensor<128x64xi32, #blocked> -> tensor<128x64xi32, #blocked2>
    %42 = tt.broadcast %39 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
    %43 = arith.addi %41, %42 : tensor<128x64xi32, #blocked2>
    %44 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked2>
    %45 = tt.addptr %44, %43 : tensor<128x64x!tt.ptr<i8>, #blocked2>, tensor<128x64xi32, #blocked2>
    %46 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked4>
    %47 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked>
    %48 = tt.addptr %47, %22 : tensor<32x1x!tt.ptr<i8>, #blocked>, tensor<32x1xi32, #blocked>
    %49 = ttg.convert_layout %46 : tensor<8xi32, #blocked4> -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %50 = tt.expand_dims %49 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x8xi32, #blocked6>
    %51 = ttg.convert_layout %50 : tensor<1x8xi32, #blocked6> -> tensor<1x8xi32, #blocked1>
    %52 = arith.muli %51, %cst_6 : tensor<1x8xi32, #blocked1>
    %53 = tt.broadcast %48 : tensor<32x1x!tt.ptr<i8>, #blocked> -> tensor<32x8x!tt.ptr<i8>, #blocked>
    %54 = ttg.convert_layout %53 : tensor<32x8x!tt.ptr<i8>, #blocked> -> tensor<32x8x!tt.ptr<i8>, #blocked1>
    %55 = tt.broadcast %52 : tensor<1x8xi32, #blocked1> -> tensor<32x8xi32, #blocked1>
    %56 = tt.addptr %54, %55 : tensor<32x8x!tt.ptr<i8>, #blocked1>, tensor<32x8xi32, #blocked1>
    %57 = ttg.convert_layout %19 : tensor<64xi32, #blocked4> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %58 = tt.expand_dims %57 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<64x1xi32, #blocked5>
    %59 = ttg.convert_layout %58 : tensor<64x1xi32, #blocked5> -> tensor<64x1xi32, #blocked>
    %60 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<64x1x!tt.ptr<i8>, #blocked>
    %61 = tt.addptr %60, %59 : tensor<64x1x!tt.ptr<i8>, #blocked>, tensor<64x1xi32, #blocked>
    %62 = arith.muli %51, %cst_5 : tensor<1x8xi32, #blocked1>
    %63 = tt.broadcast %61 : tensor<64x1x!tt.ptr<i8>, #blocked> -> tensor<64x8x!tt.ptr<i8>, #blocked>
    %64 = ttg.convert_layout %63 : tensor<64x8x!tt.ptr<i8>, #blocked> -> tensor<64x8x!tt.ptr<i8>, #blocked1>
    %65 = tt.broadcast %62 : tensor<1x8xi32, #blocked1> -> tensor<64x8xi32, #blocked1>
    %66 = tt.addptr %64, %65 : tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<64x8xi32, #blocked1>
    %67:5 = scf.for %arg7 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg8 = %cst_4, %arg9 = %32, %arg10 = %45, %arg11 = %56, %arg12 = %66) -> (tensor<32x64xf32, #blocked2>, tensor<32x128x!tt.ptr<i8>, #blocked3>, tensor<128x64x!tt.ptr<i8>, #blocked2>, tensor<32x8x!tt.ptr<i8>, #blocked1>, tensor<64x8x!tt.ptr<i8>, #blocked1>)  : i32 {
      %100 = tt.load %arg11 : tensor<32x8x!tt.ptr<i8>, #blocked1>
      %101 = tt.load %arg12 : tensor<64x8x!tt.ptr<i8>, #blocked1>
      %102 = tt.load %arg9 : tensor<32x128x!tt.ptr<i8>, #blocked3>
      %103 = tt.load %arg10 cacheModifier = cg : tensor<128x64x!tt.ptr<i8>, #blocked2>
      %104 = tt.dot_scaled %102 scale %100, %103 scale %101, %cst_4 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #blocked3>, tensor<32x8xi8, #blocked1> * tensor<128x64xi8, #blocked2>, tensor<64x8xi8, #blocked1> -> tensor<32x64xf32, #blocked2>
      %105 = arith.addf %arg8, %104 : tensor<32x64xf32, #blocked2>
      %106 = tt.addptr %arg9, %cst_3 : tensor<32x128x!tt.ptr<i8>, #blocked3>, tensor<32x128xi32, #blocked3>
      %107 = tt.addptr %arg10, %cst_2 : tensor<128x64x!tt.ptr<i8>, #blocked2>, tensor<128x64xi32, #blocked2>
      %108 = tt.addptr %arg11, %cst_1 : tensor<32x8x!tt.ptr<i8>, #blocked1>, tensor<32x8xi32, #blocked1>
      %109 = tt.addptr %arg12, %cst_0 : tensor<64x8x!tt.ptr<i8>, #blocked1>, tensor<64x8xi32, #blocked1>
      scf.yield %105, %106, %107, %108, %109 : tensor<32x64xf32, #blocked2>, tensor<32x128x!tt.ptr<i8>, #blocked3>, tensor<128x64x!tt.ptr<i8>, #blocked2>, tensor<32x8x!tt.ptr<i8>, #blocked1>, tensor<64x8x!tt.ptr<i8>, #blocked1>
    }
    %68 = arith.truncf %67#0 : tensor<32x64xf32, #blocked2> to tensor<32x64xbf16, #blocked2>
    %69 = arith.extsi %9 : tensor<32xi32, #blocked4> to tensor<32xi64, #blocked4>
    %70 = arith.extsi %8 : i32 to i64
    %71 = tt.splat %70 : i64 -> tensor<32xi64, #blocked4>
    %72 = arith.addi %71, %69 : tensor<32xi64, #blocked4>
    %73 = arith.extsi %15 : tensor<64xi32, #blocked4> to tensor<64xi64, #blocked4>
    %74 = arith.extsi %14 : i32 to i64
    %75 = tt.splat %74 : i64 -> tensor<64xi64, #blocked4>
    %76 = arith.addi %75, %73 : tensor<64xi64, #blocked4>
    %77 = ttg.convert_layout %72 : tensor<32xi64, #blocked4> -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %78 = tt.expand_dims %77 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi64, #blocked5>
    %79 = ttg.convert_layout %78 : tensor<32x1xi64, #blocked5> -> tensor<32x1xi64, #blocked>
    %80 = arith.muli %79, %cst : tensor<32x1xi64, #blocked>
    %81 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked>
    %82 = tt.addptr %81, %80 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi64, #blocked>
    %83 = ttg.convert_layout %76 : tensor<64xi64, #blocked4> -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xi64, #blocked6>
    %85 = ttg.convert_layout %84 : tensor<1x64xi64, #blocked6> -> tensor<1x64xi64, #blocked2>
    %86 = tt.broadcast %82 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
    %87 = ttg.convert_layout %86 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
    %88 = tt.broadcast %85 : tensor<1x64xi64, #blocked2> -> tensor<32x64xi64, #blocked2>
    %89 = tt.addptr %87, %88 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi64, #blocked2>
    %90 = arith.extsi %arg5 : i32 to i64
    %91 = tt.splat %90 : i64 -> tensor<32x1xi64, #blocked>
    %92 = arith.cmpi slt, %79, %91 : tensor<32x1xi64, #blocked>
    %93 = arith.extsi %arg6 : i32 to i64
    %94 = tt.splat %93 : i64 -> tensor<1x64xi64, #blocked2>
    %95 = arith.cmpi slt, %85, %94 : tensor<1x64xi64, #blocked2>
    %96 = tt.broadcast %92 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
    %97 = ttg.convert_layout %96 : tensor<32x64xi1, #blocked> -> tensor<32x64xi1, #blocked2>
    %98 = tt.broadcast %95 : tensor<1x64xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
    %99 = arith.andi %97, %98 : tensor<32x64xi1, #blocked2>
    tt.store %89, %68, %99 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(tritongpu-coalesce, tritongpu-remove-layout-conversions, tritongpu-optimize-thread-locality, tritonamdgpu-accelerate-matmul{arch-generation-name=gfx950 kPack=1 matrix-instruction-size=16}, tritongpu-remove-layout-conversions, tritonamdgpu-optimize-epilogue, tritonamdgpu-aggregate-load{arch-generation-name=gfx950}, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tt.func(tritonamdgpu-hoist-layout-conversions), tritongpu-fuse-nested-loops, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-licm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritonamdgpu-stream-pipeline{global_prefetch=0 local_prefetch=0 num_stages=2 use_async_copy=true}, tritonamdgpu-coalesce-async-copy{arch-generation-name=gfx950}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tritongpu-remove-layout-conversions, tritongpu-reduce-data-duplication, tritonamdgpu-reorder-instructions, tt.func(tritonamdgpu-canonicalize-pointers), canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritonamdgpu-convert-buffer-ops{arch-generation-name=gfx950}, tritonamdgpu-fold-true-cmpi, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, tritonamdgpu-update-async-wait-count{arch-generation-name=gfx950})",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
