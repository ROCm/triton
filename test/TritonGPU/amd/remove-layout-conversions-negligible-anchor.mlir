// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// A 128x1 load and a dot both reach the reduce, and before accelerate-matmul
// has run neither candidate layout is of the preferred kind. The load governs
// 512 bytes across 4 warps of 32 lanes, exactly the one dword per thread that
// is still counted as negligible, so its layout must not be the one the reduce
// adopts: laying the whole tile out one element per thread would put every
// value in the epilogue through shared memory.

// CHECK-DAG: #[[$DOT:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$NARROW:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#dot = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#narrow = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#generic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @reduce_keeps_dot_layout_over_narrow_load
  tt.func @reduce_keeps_dot_layout_over_narrow_load(
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>>,
      %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>>,
      %pbias: tensor<128x1x!tt.ptr<f32>, #narrow>)
      -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>> {
    // The dot result is left alone and the 128x1 load is converted instead.
    // CHECK:      %[[ACC:.*]] = tt.dot
    // CHECK:      %[[LOAD:.*]] = tt.load {{.*}} : tensor<128x1x!tt.ptr<f32>, #[[$NARROW]]>
    // CHECK:      %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<128x1xf32, #[[$NARROW]]> -> tensor<128x1xf32, #[[$DOT]]>
    // CHECK:      %[[BCAST:.*]] = tt.broadcast %[[CVT]]
    // CHECK:      %[[SUM:.*]] = arith.addf %[[ACC]], %[[BCAST]]
    // CHECK:      "tt.reduce"(%[[SUM]])
    // CHECK:      }) : (tensor<128x64xf32, #[[$DOT]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$DOT]]}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #dot>
    %acc = tt.dot %a, %b, %zero : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>> -> tensor<128x64xf32, #dot>
    %bias = tt.load %pbias : tensor<128x1x!tt.ptr<f32>, #narrow>
    %biasg = ttg.convert_layout %bias : tensor<128x1xf32, #narrow> -> tensor<128x1xf32, #generic>
    %bcast = tt.broadcast %biasg : tensor<128x1xf32, #generic> -> tensor<128x64xf32, #generic>
    %accg = ttg.convert_layout %acc : tensor<128x64xf32, #dot> -> tensor<128x64xf32, #generic>
    %sum = arith.addf %accg, %bcast : tensor<128x64xf32, #generic>
    %red = "tt.reduce"(%sum) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<128x64xf32, #generic>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
    tt.return %red : tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
  }
}

// -----

// A tensor of pointers counts as the data it addresses. The 128x64 pointer
// argument governs a 32 KiB access and so must outrank the 128x1 offset load,
// which governs the one dword per thread that counts as negligible. Measuring
// a !tt.ptr element by its own width instead -- which reports nothing, since
// a pointer is neither an int nor a float -- would score the pointer argument
// zero and invert that, sending the whole pointer tensor through a conversion.

// CHECK-DAG: #[[$PTRS:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$OFFS:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

#ptrs = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#offs = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#generic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @addptr_keeps_pointer_arg_layout
  tt.func @addptr_keeps_pointer_arg_layout(
      %p: tensor<128x64x!tt.ptr<f32>, #ptrs>,
      %poff: tensor<128x1x!tt.ptr<i32>, #offs>)
      -> tensor<128x64x!tt.ptr<f32>, #generic> {
    // Only the 128 offsets are converted; the pointer tensor keeps its layout.
    // CHECK:      %[[OFF:.*]] = tt.load %arg1 : tensor<128x1x!tt.ptr<i32>, #[[$OFFS]]>
    // CHECK:      %[[CVT:.*]] = ttg.convert_layout %[[OFF]] : tensor<128x1xi32, #[[$OFFS]]> -> tensor<128x1xi32, #[[$PTRS]]>
    // CHECK:      %[[BCAST:.*]] = tt.broadcast %[[CVT]] : tensor<128x1xi32, #[[$PTRS]]> -> tensor<128x64xi32, #[[$PTRS]]>
    // CHECK:      tt.addptr %arg0, %[[BCAST]] : tensor<128x64x!tt.ptr<f32>, #[[$PTRS]]>
    %off1 = tt.load %poff : tensor<128x1x!tt.ptr<i32>, #offs>
    %off1g = ttg.convert_layout %off1 : tensor<128x1xi32, #offs> -> tensor<128x1xi32, #generic>
    %off = tt.broadcast %off1g : tensor<128x1xi32, #generic> -> tensor<128x64xi32, #generic>
    %pg = ttg.convert_layout %p : tensor<128x64x!tt.ptr<f32>, #ptrs> -> tensor<128x64x!tt.ptr<f32>, #generic>
    %p2 = tt.addptr %pg, %off : tensor<128x64x!tt.ptr<f32>, #generic>, tensor<128x64xi32, #generic>
    tt.return %p2 : tensor<128x64x!tt.ptr<f32>, #generic>
  }
}

// -----

// Traffic never takes a layout away from the memory access that owns it. This
// atomic governs 512 bytes, the one dword per thread that counts as
// negligible, and the reduction feeding its value governs a whole 128x64 tile.
// Demoting the atomic to that tile's layout would stop it addressing
// contiguous memory to save a conversion of 128 values, so its own encoding --
// the one coalesce picked by measuring this access -- wins regardless.
//
// tt.load cannot reach this case: it carries SameLoadStoreOperandsAndResult-
// Encoding rather than the SameOperandsAndResultEncoding propagateToUsers
// looks for, so nothing ever propagates onto a load result, and tt.store has
// no result to propagate onto. The atomics are the reachable ones.

// CHECK-DAG: #[[$TILE:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$OWN:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

#tile = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#own = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @atomic_keeps_own_layout
  tt.func @atomic_keeps_own_layout(
      %ptile: tensor<128x64x!tt.ptr<f32>, #tile>,
      %pat: tensor<128x1x!tt.ptr<f32>, #own>,
      %mask: tensor<128x1xi1, #own>)
      -> tensor<128x1xf32, #own> {
    // CHECK: tt.atomic_rmw fadd, relaxed, gpu, %arg1, {{.*}}, %arg2 : (tensor<128x1x!tt.ptr<f32>, #[[$OWN]]>, tensor<128x1xf32, #[[$OWN]]>, tensor<128x1xi1, #[[$OWN]]>) -> tensor<128x1xf32, #[[$OWN]]>
    %t = tt.load %ptile : tensor<128x64x!tt.ptr<f32>, #tile>
    %red = "tt.reduce"(%t) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<128x64xf32, #tile>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #tile}>>
    %val = tt.expand_dims %red {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #tile}>> -> tensor<128x1xf32, #tile>
    %valc = ttg.convert_layout %val : tensor<128x1xf32, #tile> -> tensor<128x1xf32, #own>
    %r = tt.atomic_rmw fadd, relaxed, gpu, %pat, %valc, %mask : (tensor<128x1x!tt.ptr<f32>, #own>, tensor<128x1xf32, #own>, tensor<128x1xi1, #own>) -> tensor<128x1xf32, #own>
    tt.return %r : tensor<128x1xf32, #own>
  }
}

// -----

// The same epilogue with a load that is not negligible: it governs a full
// 128x64 tile, so the traffic rule does not fire and the layout it anchors is
// left to win the conflict, as it did before that rule existed.

// CHECK-DAG: #[[$WIDE:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

#dot = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#wide = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#generic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @reduce_keeps_wide_load_layout
  tt.func @reduce_keeps_wide_load_layout(
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>>,
      %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>>,
      %pbias: tensor<128x64x!tt.ptr<f32>, #wide>)
      -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>> {
    // CHECK:      "tt.reduce"
    // CHECK:      }) : (tensor<128x64xf32, #[[$WIDE]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$WIDE]]}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #dot>
    %acc = tt.dot %a, %b, %zero : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>> -> tensor<128x64xf32, #dot>
    %bias = tt.load %pbias : tensor<128x64x!tt.ptr<f32>, #wide>
    %biasg = ttg.convert_layout %bias : tensor<128x64xf32, #wide> -> tensor<128x64xf32, #generic>
    %accg = ttg.convert_layout %acc : tensor<128x64xf32, #dot> -> tensor<128x64xf32, #generic>
    %sum = arith.addf %accg, %biasg : tensor<128x64xf32, #generic>
    %red = "tt.reduce"(%sum) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<128x64xf32, #generic>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
    tt.return %red : tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
  }
}
