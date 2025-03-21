[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %109 = arith.addf %108, %cst : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: srcShape: 128x64
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128x32
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: numReps: 1x2
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %176 = arith.addf %arg27, %arg28 : f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %176 = arith.addf %arg27, %arg28 : f32
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %129 = arith.addf %128, %122 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: srcShape: 128
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: numReps: 1
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %120 = arith.subf %115, %119 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: srcShape: 128x64
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128x32
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: numReps: 1x2
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %129 = arith.subf %98, %117 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: srcShape: 128
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: numReps: 1
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %108 = arith.mulf %107, %cst_0 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: srcShape: 128x64
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128x32
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: numReps: 1x2
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %142 = arith.mulf %96, %141 : tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: srcShape: 128x128
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128x32
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: mfma layout

[tritonamdgpu-refine-ops]: instrShape: 32x32
[tritonamdgpu-refine-ops]: warpsPerCTA: 4x1
[tritonamdgpu-refine-ops]: numReps: 1x4
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %155 = arith.mulf %97, %139 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: srcShape: 128
[tritonamdgpu-refine-ops]: srcShapePerCtaTile: 128
[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: getRefinedShape()

[tritonamdgpu-refine-ops]: numReps: 1
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11236 = "amdgpu.extract_slice"(%81) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11286 = "amdgpu.extract_slice"(%81) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11336 = "amdgpu.extract_slice"(%45) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11386 = "amdgpu.extract_slice"(%45) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11469 = "amdgpu.extract_slice"(%8292) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11551 = "amdgpu.extract_slice"(%8292) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11633 = "amdgpu.extract_slice"(%8292) <{static_offsets = array<i64: 0, 64>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x64
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11715 = "amdgpu.extract_slice"(%8292) <{static_offsets = array<i64: 0, 96>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x96
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x12
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14449 = "amdgpu.extract_slice"(%10931) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14499 = "amdgpu.extract_slice"(%10931) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15286 = "amdgpu.extract_slice"(%15234) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15336 = "amdgpu.extract_slice"(%15234) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 16
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15632 = "amdgpu.extract_slice"(%15435) <{static_offsets = array<i64: 0, 96>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x96
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x12
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15714 = "amdgpu.extract_slice"(%15435) <{static_offsets = array<i64: 0, 64>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x64
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15796 = "amdgpu.extract_slice"(%15435) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=15
[tritonamdgpu-extract-slice-to-llvm]: resultTy.getShape(): 128x32
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15878 = "amdgpu.extract_slice"(%15435) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x32
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x4
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 48
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 16
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=8
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=9
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=10
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=11
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=12
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=13
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=14
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=15
INFO: running a single config given from a file
fused-attention-fwd-d128-layoutbhsd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton       torch
0    2.0  16.0  16.0   8192.0   8192.0  373.962049  261.557999
Refining ops.
TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter()
