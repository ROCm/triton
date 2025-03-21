[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9150 = "amdgpu.extract_slice"(%8306) <{static_offsets = array<i64: 0, 0>}> : (tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<16x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 64x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 4x8
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: sizes: 16x128
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 4x1
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9250 = "amdgpu.extract_slice"(%8306) <{static_offsets = array<i64: 16, 0>}> : (tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<16x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 64x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 4x8
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 16x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: sizes: 16x128
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 1x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 4x1
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9350 = "amdgpu.extract_slice"(%8306) <{static_offsets = array<i64: 32, 0>}> : (tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<16x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 64x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 4x8
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 32x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: sizes: 16x128
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 2x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 4x1
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9392 = "amdgpu.extract_slice"(%8306) <{static_offsets = array<i64: 48, 0>}> : (tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<16x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 64x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 4x8
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 48x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 16x128
[tritonamdgpu-extract-slice-to-llvm]: sizes: 16x128
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 3x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 4x1
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9434 = "amdgpu.extract_slice"(%9116) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> tensor<128x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 8x4
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 8x1
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9476 = "amdgpu.extract_slice"(%9116) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> tensor<128x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 8x4
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 8x1
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9518 = "amdgpu.extract_slice"(%9116) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> tensor<128x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 8x4
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 8x1
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9560 = "amdgpu.extract_slice"(%9116) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> tensor<128x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 8x4
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 8x1
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x4
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 0
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9634 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 0>}> {DotTile = #amdgpu<DotTile<0, -1, 0> [0], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %9676 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
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
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %10101 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x32xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
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
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %10558 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 16>}> {DotTile = #amdgpu<DotTile<0, -1, 1> [2], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %11382 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 32>}> {DotTile = #amdgpu<DotTile<0, -1, 2> [4], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12206 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 48>}> {DotTile = #amdgpu<DotTile<0, -1, 3> [6], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %12914 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 64>}> {DotTile = #amdgpu<DotTile<0, -1, 4> [8], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x64
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 40
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %13622 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 80>}> {DotTile = #amdgpu<DotTile<0, -1, 5> [10], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x80
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x5
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 40
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14330 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 96>}> {DotTile = #amdgpu<DotTile<0, -1, 6> [12], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x96
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x6
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 56
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %14846 = "amdgpu.extract_slice"(%7551) <{static_offsets = array<i64: 0, 112>}> {DotTile = #amdgpu<DotTile<0, -1, 7> [14], <0, -1, 0> [0]>} : (tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>) -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x8
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x112
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x16
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x16
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x7
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 56
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 56
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 8
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=3
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=4
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=5
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=6
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=7
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15330 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15368 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15406 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15444 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x24
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15482 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15520 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x40
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x5
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15558 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x6
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15596 = "amdgpu.extract_slice"(%101) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x56
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x7
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15634 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15672 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15710 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15748 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x24
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15786 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15824 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x40
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x5
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15862 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x6
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15900 = "amdgpu.extract_slice"(%65) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x56
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x7
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %15971 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16041 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16111 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16181 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x24
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16251 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16321 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x40
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x5
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16391 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x6
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16461 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x56
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x7
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16531 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 64>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x64
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 32
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 36
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=32, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=33, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=34, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=35, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16601 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 72>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x72
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x9
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 36
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 40
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=36, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=37, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=38, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=39, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16671 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 80>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x80
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x10
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 40
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 44
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=40, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=41, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=42, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=43, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16741 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 88>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x88
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x11
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 44
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 48
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=44, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=45, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=46, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=47, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16811 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 96>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x96
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x12
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 48
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 52
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=48, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=49, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=50, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=51, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16881 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 104>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x104
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x13
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 52
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 56
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=52, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=53, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=54, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=55, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %16951 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 112>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x112
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x14
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 56
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 60
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=56, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=57, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=58, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=59, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %17021 = "amdgpu.extract_slice"(%8312) <{static_offsets = array<i64: 0, 120>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x128
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x64
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x120
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 64
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x15
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x16
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 60
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 60
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 64
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=60, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=61, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=62, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=63, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:326:20: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
        acc = acc * alpha[:, None]
                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18757 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 56>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x56
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x7
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 28
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 32
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=28, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=29, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=30, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=31, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18795 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 48>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x48
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x6
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 24
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 28
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=24, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=25, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=26, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=27, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18833 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 40>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x40
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x5
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 20
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 24
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=20, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=21, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=22, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=23, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18871 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 32>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x32
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x4
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 16
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 20
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=16, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=17, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=18, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=19, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18909 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 24>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x24
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x3
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 12
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 16
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=12, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=13, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=14, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=15, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18947 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 16>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x16
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x2
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 8
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 12
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=8, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=9, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=10, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=11, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %18985 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 8>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x8
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x1
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 4
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 8
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=4, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=5, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=6, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=7, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
[tritonamdgpu-extract-slice-to-llvm]: 

ExtractSliceOpToLLVM::processLayout()
[tritonamdgpu-extract-slice-to-llvm]: op: %19023 = "amdgpu.extract_slice"(%17158) <{static_offsets = array<i64: 0, 0>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x8xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-extract-slice-to-llvm]: srcShape: 128x64
[tritonamdgpu-extract-slice-to-llvm]: elemsPerThread: 1x32
[tritonamdgpu-extract-slice-to-llvm]: contigPerThread: 1x4
[tritonamdgpu-extract-slice-to-llvm]: offsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: vals.size(): 32
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: shapePerCTATile: 128x8
[tritonamdgpu-extract-slice-to-llvm]: sizes: 128x8
[tritonamdgpu-extract-slice-to-llvm]: CTAOffsets: 0x0
[tritonamdgpu-extract-slice-to-llvm]: CTASizes: 1x1
[tritonamdgpu-extract-slice-to-llvm]: CTAPerShape: 1x8
[tritonamdgpu-extract-slice-to-llvm]: skipElems: 0
[tritonamdgpu-extract-slice-to-llvm]: tensorStride: 28
[tritonamdgpu-extract-slice-to-llvm]: lastIdx: 4
[tritonamdgpu-extract-slice-to-llvm]: numElemsPerVec: 4
[tritonamdgpu-extract-slice-to-llvm]: i=0, j=0
[tritonamdgpu-extract-slice-to-llvm]: i=1, j=1
[tritonamdgpu-extract-slice-to-llvm]: i=2, j=2
[tritonamdgpu-extract-slice-to-llvm]: i=3, j=3
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error:  size mismatch when packing elements for LLVM struct expected 16 but got 4
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:19: error:  size mismatch when packing elements for LLVM struct expected 32 but got 128
            qk += (tl.dot(q, k) * QK_SCALE)
                  ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
python3: /home/dtanner/.triton/llvm/llvm-2619c2ed-ubuntu-x64/include/llvm/ADT/ArrayRef.h:260: const T &llvm::ArrayRef<mlir::Type>::operator[](size_t) const [T = mlir::Type]: Assertion `Index < Length && "Invalid index!"' failed.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: f32, %arg24: i32, %arg25: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg26: i32) attributes {noinline = false} {
    %c56_i32 = arith.constant 56 : i32
    %c40_i32 = arith.constant 40 : i32
    %c24_i32 = arith.constant 24 : i32
    %c8_i32 = arith.constant 8 : i32
    %c112_i32 = arith.constant 112 : i32
    %c96_i32 = arith.constant 96 : i32
    %c80_i32 = arith.constant 80 : i32
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0.127517432> : tensor<128x64xf32, #mma>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #mma>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_4 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %c8320_i32 = arith.constant 8320 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_6 = arith.constant dense<8192> : tensor<128x1xi32, #blocked>
    %c8128_i32 = arith.constant 8128 : i32
    %cst_7 = arith.constant dense<true> : tensor<128x128xi1, #mma>
    %cst_8 = arith.constant dense<8192> : tensor<128x1xi32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #blocked1>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #blocked1>
    %13 = arith.muli %2, %arg5 : i32
    %14 = tt.addptr %arg0, %13 : !tt.ptr<f16>, i32
    %15 = arith.muli %1, %arg6 : i32
    %16 = tt.addptr %14, %15 : !tt.ptr<f16>, i32
    %17 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %18 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
    %20 = arith.muli %18, %19 : tensor<128x1xi32, #blocked>
    %21 = tt.splat %16 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %22 = tt.addptr %21, %20 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25 = tt.expand_dims %23 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %26 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %27 = tt.broadcast %22 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %28 = tt.broadcast %25 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %29 = tt.broadcast %26 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %30 = tt.addptr %27, %29 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %31 = arith.muli %2, %arg8 : i32
    %32 = tt.addptr %arg1, %31 : !tt.ptr<f16>, i32
    %33 = arith.muli %1, %arg9 : i32
    %34 = tt.addptr %32, %33 : !tt.ptr<f16>, i32
    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %36 = tt.expand_dims %35 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %37 = tt.splat %34 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %38 = tt.addptr %37, %36 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
    %39 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %40 = tt.expand_dims %39 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %41 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked2>
    %42 = arith.muli %40, %41 : tensor<1x64xi32, #blocked2>
    %43 = tt.broadcast %38 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
    %44 = tt.broadcast %42 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
    %45 = tt.addptr %43, %44 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
    %46 = arith.muli %2, %arg11 : i32
    %47 = tt.addptr %arg2, %46 : !tt.ptr<f16>, i32
    %48 = arith.muli %1, %arg12 : i32
    %49 = tt.addptr %47, %48 : !tt.ptr<f16>, i32
    %50 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %51 = tt.expand_dims %50 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %52 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked>
    %53 = arith.muli %51, %52 : tensor<64x1xi32, #blocked>
    %54 = tt.splat %49 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %55 = tt.addptr %54, %53 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %56 = tt.broadcast %55 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %57 = tt.broadcast %26 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %58 = tt.addptr %56, %57 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %59 = arith.cmpi slt, %17, %cst_8 : tensor<128x1xi32, #mma>
    %60 = arith.cmpi slt, %18, %cst_6 : tensor<128x1xi32, #blocked>
    %61 = tt.broadcast %59 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %62 = tt.broadcast %60 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %63 = arith.muli %arg10, %c64_i32 : i32
    %64 = tt.splat %63 : i32 -> tensor<128x64xi32, #blocked2>
    %65 = arith.muli %arg13, %c64_i32 : i32
    %66 = tt.splat %65 : i32 -> tensor<64x128xi32, #blocked>
    %67 = arith.addi %0, %c1_i32 : i32
    %68 = arith.muli %67, %c128_i32 : i32
    %69 = arith.muli %2, %c131072_i32 : i32
    %70 = tt.addptr %arg3, %69 : !tt.ptr<f32>, i32
    %71 = arith.muli %1, %c8192_i32 : i32
    %72 = tt.addptr %70, %71 : !tt.ptr<f32>, i32
    %73 = tt.splat %72 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %74 = tt.addptr %73, %12 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %75 = arith.subi %68, %c8192_i32 : i32
    %76 = arith.cmpi sgt, %75, %c0_i32 : i32
    %77 = arith.muli %2, %arg14 : i32
    %78 = tt.addptr %arg4, %77 : !tt.ptr<f16>, i32
    %79 = arith.muli %1, %arg15 : i32
    %80 = tt.addptr %78, %79 : !tt.ptr<f16>, i32
    %81 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
    %82 = arith.muli %17, %81 : tensor<128x1xi32, #mma>
    %83 = tt.splat %80 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %84 = tt.addptr %83, %82 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %85 = tt.broadcast %84 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #mma>
    %86 = tt.addptr %85, %28 : tensor<128x128x!tt.ptr<f16>, #mma>, tensor<128x128xi32, #mma>
    %87 = arith.select %76, %61, %cst_7 : tensor<128x128xi1, #mma>
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%88: i32):  // 2 preds: ^bb0, ^bb8
    %89 = arith.cmpi slt, %88, %c1_i32 : i32
    cf.cond_br %89, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %90 = tt.load %30, %62, %cst_5 : tensor<128x128x!tt.ptr<f16>, #blocked>
    gpu.barrier
    %91 = ttg.local_alloc %90 {allocation.offset = 0 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    gpu.barrier
    %92 = ttg.local_load %91 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %93 = tt.load %45 : tensor<128x64x!tt.ptr<f16>, #blocked2>
    gpu.barrier
    %94 = ttg.local_alloc %93 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    cf.br ^bb3(%c0_i32, %cst_1, %cst_3, %cst_4, %45, %58, %94 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb3(%95: i32, %96: tensor<128x128xf32, #mma>, %97: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %98: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %99: tensor<128x64x!tt.ptr<f16>, #blocked2>, %100: tensor<64x128x!tt.ptr<f16>, #blocked>, %101: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>):  // 2 preds: ^bb2, ^bb4
    %102 = arith.cmpi slt, %95, %c8128_i32 : i32
    cf.cond_br %102, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %103 = tt.addptr %99, %64 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
    %104 = amdgpu.extract_slice %103 [0, 0] : tensor<128x64x!tt.ptr<f16>, #blocked2> to tensor<128x16x!tt.ptr<f16>, #blocked2>
    %105 = tt.load %104 : tensor<128x16x!tt.ptr<f16>, #blocked2>
    %106 = amdgpu.extract_slice %103 [0, 16] : tensor<128x64x!tt.ptr<f16>, #blocked2> to tensor<128x16x!tt.ptr<f16>, #blocked2>
    %107 = tt.load %106 : tensor<128x16x!tt.ptr<f16>, #blocked2>
    %108 = amdgpu.extract_slice %103 [0, 32] : tensor<128x64x!tt.ptr<f16>, #blocked2> to tensor<128x16x!tt.ptr<f16>, #blocked2>
    %109 = tt.load %108 : tensor<128x16x!tt.ptr<f16>, #blocked2>
    %110 = amdgpu.extract_slice %103 [0, 48] : tensor<128x64x!tt.ptr<f16>, #blocked2> to tensor<128x16x!tt.ptr<f16>, #blocked2>
    %111 = tt.load %110 : tensor<128x16x!tt.ptr<f16>, #blocked2>
    %112 = amdgpu.concat %105, %107, %109, %111 [1, 4] : tensor<128x16xf16, #blocked2>, tensor<128x16xf16, #blocked2>, tensor<128x16xf16, #blocked2>, tensor<128x16xf16, #blocked2> -> tensor<128x64xf16, #blocked2>
    gpu.barrier
    %113 = ttg.memdesc_subview %101[%c0_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %114 = ttg.local_load %113 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %115 = ttg.memdesc_subview %101[%c16_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %116 = ttg.local_load %115 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %117 = ttg.memdesc_subview %101[%c32_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %118 = ttg.local_load %117 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %119 = ttg.memdesc_subview %101[%c48_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %120 = ttg.local_load %119 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %121 = ttg.memdesc_subview %101[%c64_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %122 = ttg.local_load %121 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %123 = ttg.memdesc_subview %101[%c80_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %124 = ttg.local_load %123 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %125 = ttg.memdesc_subview %101[%c96_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %126 = ttg.local_load %125 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %127 = ttg.memdesc_subview %101[%c112_i32, %c0_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %128 = ttg.local_load %127 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %129 = ttg.memdesc_subview %101[%c0_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %130 = ttg.local_load %129 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %131 = ttg.memdesc_subview %101[%c16_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %132 = ttg.local_load %131 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %133 = ttg.memdesc_subview %101[%c32_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %134 = ttg.local_load %133 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %135 = ttg.memdesc_subview %101[%c48_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %136 = ttg.local_load %135 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %137 = ttg.memdesc_subview %101[%c64_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %138 = ttg.local_load %137 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %139 = ttg.memdesc_subview %101[%c80_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %140 = ttg.local_load %139 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %141 = ttg.memdesc_subview %101[%c96_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %142 = ttg.local_load %141 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %143 = ttg.memdesc_subview %101[%c112_i32, %c32_i32] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64>
    %144 = ttg.local_load %143 : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable, 128x64> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %145 = amdgpu.extract_slice %100 [0, 0] : tensor<64x128x!tt.ptr<f16>, #blocked> to tensor<16x128x!tt.ptr<f16>, #blocked>
    %146 = tt.load %145 : tensor<16x128x!tt.ptr<f16>, #blocked>
    %147 = amdgpu.extract_slice %100 [16, 0] : tensor<64x128x!tt.ptr<f16>, #blocked> to tensor<16x128x!tt.ptr<f16>, #blocked>
    %148 = tt.load %147 : tensor<16x128x!tt.ptr<f16>, #blocked>
    %149 = amdgpu.extract_slice %100 [32, 0] : tensor<64x128x!tt.ptr<f16>, #blocked> to tensor<16x128x!tt.ptr<f16>, #blocked>
    %150 = tt.load %149 : tensor<16x128x!tt.ptr<f16>, #blocked>
    %151 = amdgpu.extract_slice %100 [48, 0] : tensor<64x128x!tt.ptr<f16>, #blocked> to tensor<16x128x!tt.ptr<f16>, #blocked>
    %152 = tt.load %151 : tensor<16x128x!tt.ptr<f16>, #blocked>
    %153 = amdgpu.concat %146, %148, %150, %152 [4, 1] : tensor<16x128xf16, #blocked>, tensor<16x128xf16, #blocked>, tensor<16x128xf16, #blocked>, tensor<16x128xf16, #blocked> -> tensor<64x128xf16, #blocked>
    %154 = amdgpu.extract_slice %92 [0, 0] {DotTile = #amdgpu<DotTile<0, -1, 0> [0], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %155 = amdgpu.extract_slice %92 [0, 16] {DotTile = #amdgpu<DotTile<0, -1, 1> [2], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %156 = amdgpu.extract_slice %92 [0, 32] {DotTile = #amdgpu<DotTile<0, -1, 2> [4], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %157 = amdgpu.extract_slice %92 [0, 48] {DotTile = #amdgpu<DotTile<0, -1, 3> [6], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %158 = amdgpu.extract_slice %92 [0, 64] {DotTile = #amdgpu<DotTile<0, -1, 4> [8], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %159 = amdgpu.extract_slice %92 [0, 80] {DotTile = #amdgpu<DotTile<0, -1, 5> [10], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %160 = amdgpu.extract_slice %92 [0, 96] {DotTile = #amdgpu<DotTile<0, -1, 6> [12], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %161 = amdgpu.extract_slice %92 [0, 112] {DotTile = #amdgpu<DotTile<0, -1, 7> [14], <0, -1, 0> [0]>} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> to tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %162 = amdgpu.extract_slice %cst [0, 0] : tensor<128x64xf32, #mma> to tensor<128x32xf32, #mma>
    %163 = amdgpu.extract_slice %cst [0, 32] : tensor<128x64xf32, #mma> to tensor<128x32xf32, #mma>
    %164 = tt.dot %154, %114, %162, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 0> [0], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %165 = tt.dot %154, %130, %163, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 0> [1], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %166 = tt.dot %155, %116, %164, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 1> [2], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %167 = tt.dot %155, %132, %165, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 1> [3], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %168 = tt.dot %156, %118, %166, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 2> [4], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %169 = tt.dot %156, %134, %167, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 2> [5], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %170 = tt.dot %157, %120, %168, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 3> [6], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %171 = tt.dot %157, %136, %169, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 3> [7], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %172 = tt.dot %158, %122, %170, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 4> [8], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %173 = tt.dot %158, %138, %171, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 4> [9], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %174 = tt.dot %159, %124, %172, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 5> [10], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %175 = tt.dot %159, %140, %173, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 5> [11], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %176 = tt.dot %160, %126, %174, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 6> [12], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %177 = tt.dot %160, %142, %175, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 6> [13], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %178 = tt.dot %161, %128, %176, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 7> [14], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %179 = tt.dot %161, %144, %177, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 7> [15], <0, 0, 0> [0]>} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma>
    %180 = amdgpu.concat %178, %179 [1, 2] : tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma> -> tensor<128x64xf32, #mma>
    %181 = amdgpu.extract_slice %180 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %182 = amdgpu.extract_slice %cst_0 [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %183 = arith.mulf %181, %182 : tensor<128x8xf32, #mma>
    %184 = amdgpu.extract_slice %180 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %185 = amdgpu.extract_slice %cst_0 [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %186 = arith.mulf %184, %185 : tensor<128x8xf32, #mma>
    %187 = amdgpu.extract_slice %180 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %188 = amdgpu.extract_slice %cst_0 [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %189 = arith.mulf %187, %188 : tensor<128x8xf32, #mma>
    %190 = amdgpu.extract_slice %180 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %191 = amdgpu.extract_slice %cst_0 [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %192 = arith.mulf %190, %191 : tensor<128x8xf32, #mma>
    %193 = amdgpu.extract_slice %180 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %194 = amdgpu.extract_slice %cst_0 [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %195 = arith.mulf %193, %194 : tensor<128x8xf32, #mma>
    %196 = amdgpu.extract_slice %180 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %197 = amdgpu.extract_slice %cst_0 [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %198 = arith.mulf %196, %197 : tensor<128x8xf32, #mma>
    %199 = amdgpu.extract_slice %180 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %200 = amdgpu.extract_slice %cst_0 [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %201 = arith.mulf %199, %200 : tensor<128x8xf32, #mma>
    %202 = amdgpu.extract_slice %180 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %203 = amdgpu.extract_slice %cst_0 [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %204 = arith.mulf %202, %203 : tensor<128x8xf32, #mma>
    %205 = amdgpu.extract_slice %cst [0, 0] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %206 = arith.addf %183, %205 : tensor<128x8xf32, #mma>
    %207 = amdgpu.extract_slice %cst [0, 8] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %208 = arith.addf %186, %207 : tensor<128x8xf32, #mma>
    %209 = amdgpu.extract_slice %cst [0, 16] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %210 = arith.addf %189, %209 : tensor<128x8xf32, #mma>
    %211 = amdgpu.extract_slice %cst [0, 24] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %212 = arith.addf %192, %211 : tensor<128x8xf32, #mma>
    %213 = amdgpu.extract_slice %cst [0, 32] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %214 = arith.addf %195, %213 : tensor<128x8xf32, #mma>
    %215 = amdgpu.extract_slice %cst [0, 40] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %216 = arith.addf %198, %215 : tensor<128x8xf32, #mma>
    %217 = amdgpu.extract_slice %cst [0, 48] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %218 = arith.addf %201, %217 : tensor<128x8xf32, #mma>
    %219 = amdgpu.extract_slice %cst [0, 56] : tensor<128x64xf32, #mma> to tensor<128x8xf32, #mma>
    %220 = arith.addf %204, %219 : tensor<128x8xf32, #mma>
    %221 = amdgpu.concat %206, %208, %210, %212, %214, %216, %218, %220 [1, 8] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x64xf32, #mma>
    %222 = "tt.reduce"(%221) <{axis = 1 : i32}> ({
    ^bb0(%arg27: f32, %arg28: f32):
      %464 = arith.maxnumf %arg27, %arg28 : f32
      tt.reduce.return %464 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %223 = arith.maxnumf %98, %222 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %224 = tt.expand_dims %223 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %225 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %226 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %227 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %228 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %229 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %230 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %231 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %232 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %233 = arith.subf %206, %225 : tensor<128x8xf32, #mma>
    %234 = arith.subf %208, %226 : tensor<128x8xf32, #mma>
    %235 = arith.subf %210, %227 : tensor<128x8xf32, #mma>
    %236 = arith.subf %212, %228 : tensor<128x8xf32, #mma>
    %237 = arith.subf %214, %229 : tensor<128x8xf32, #mma>
    %238 = arith.subf %216, %230 : tensor<128x8xf32, #mma>
    %239 = arith.subf %218, %231 : tensor<128x8xf32, #mma>
    %240 = arith.subf %220, %232 : tensor<128x8xf32, #mma>
    %241 = math.exp2 %233 : tensor<128x8xf32, #mma>
    %242 = math.exp2 %234 : tensor<128x8xf32, #mma>
    %243 = math.exp2 %235 : tensor<128x8xf32, #mma>
    %244 = math.exp2 %236 : tensor<128x8xf32, #mma>
    %245 = math.exp2 %237 : tensor<128x8xf32, #mma>
    %246 = math.exp2 %238 : tensor<128x8xf32, #mma>
    %247 = math.exp2 %239 : tensor<128x8xf32, #mma>
    %248 = math.exp2 %240 : tensor<128x8xf32, #mma>
    %249 = amdgpu.concat %241, %242, %243, %244, %245, %246, %247, %248 [1, 8] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x64xf32, #mma>
    %250 = "tt.reduce"(%249) <{axis = 1 : i32}> ({
    ^bb0(%arg27: f32, %arg28: f32):
      %464 = arith.addf %arg27, %arg28 : f32
      tt.reduce.return %464 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %251 = arith.subf %98, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %252 = math.exp2 %251 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %253 = tt.expand_dims %252 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %254 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %255 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %256 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %257 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %258 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %259 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %260 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %261 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %262 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %263 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %264 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %265 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %266 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %267 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %268 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %269 = tt.broadcast %253 : tensor<128x1xf32, #mma> -> tensor<128x8xf32, #mma>
    %270 = amdgpu.extract_slice %96 [0, 0] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %271 = arith.mulf %270, %254 : tensor<128x8xf32, #mma>
    %272 = amdgpu.extract_slice %96 [0, 8] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %273 = arith.mulf %272, %255 : tensor<128x8xf32, #mma>
    %274 = amdgpu.extract_slice %96 [0, 16] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %275 = arith.mulf %274, %256 : tensor<128x8xf32, #mma>
    %276 = amdgpu.extract_slice %96 [0, 24] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %277 = arith.mulf %276, %257 : tensor<128x8xf32, #mma>
    %278 = amdgpu.extract_slice %96 [0, 32] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %279 = arith.mulf %278, %258 : tensor<128x8xf32, #mma>
    %280 = amdgpu.extract_slice %96 [0, 40] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %281 = arith.mulf %280, %259 : tensor<128x8xf32, #mma>
    %282 = amdgpu.extract_slice %96 [0, 48] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %283 = arith.mulf %282, %260 : tensor<128x8xf32, #mma>
    %284 = amdgpu.extract_slice %96 [0, 56] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %285 = arith.mulf %284, %261 : tensor<128x8xf32, #mma>
    %286 = amdgpu.extract_slice %96 [0, 64] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %287 = arith.mulf %286, %262 : tensor<128x8xf32, #mma>
    %288 = amdgpu.extract_slice %96 [0, 72] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %289 = arith.mulf %288, %263 : tensor<128x8xf32, #mma>
    %290 = amdgpu.extract_slice %96 [0, 80] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %291 = arith.mulf %290, %264 : tensor<128x8xf32, #mma>
    %292 = amdgpu.extract_slice %96 [0, 88] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %293 = arith.mulf %292, %265 : tensor<128x8xf32, #mma>
    %294 = amdgpu.extract_slice %96 [0, 96] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %295 = arith.mulf %294, %266 : tensor<128x8xf32, #mma>
    %296 = amdgpu.extract_slice %96 [0, 104] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %297 = arith.mulf %296, %267 : tensor<128x8xf32, #mma>
    %298 = amdgpu.extract_slice %96 [0, 112] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %299 = arith.mulf %298, %268 : tensor<128x8xf32, #mma>
    %300 = amdgpu.extract_slice %96 [0, 120] : tensor<128x128xf32, #mma> to tensor<128x8xf32, #mma>
    %301 = arith.mulf %300, %269 : tensor<128x8xf32, #mma>
    %302 = amdgpu.concat %271, %273, %275, %277, %279, %281, %283, %285, %287, %289, %291, %293, %295, %297, %299, %301 [1, 16] : tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma>, tensor<128x8xf32, #mma> -> tensor<128x128xf32, #mma>
    %303 = arith.mulf %97, %252 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %304 = arith.addf %303, %250 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %305 = arith.truncf %241 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %306 = arith.truncf %242 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %307 = arith.truncf %243 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %308 = arith.truncf %244 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %309 = arith.truncf %245 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %310 = arith.truncf %246 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %311 = arith.truncf %247 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %312 = arith.truncf %248 : tensor<128x8xf32, #mma> to tensor<128x8xf16, #mma>
    %313 = ttg.convert_layout %305 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %314 = ttg.convert_layout %306 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %315 = ttg.convert_layout %307 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %316 = ttg.convert_layout %308 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %317 = ttg.convert_layout %309 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %318 = ttg.convert_layout %310 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %319 = ttg.convert_layout %311 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %320 = ttg.convert_layout %312 : tensor<128x8xf16, #mma> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %321 = ttg.local_alloc %153 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %322 = ttg.memdesc_subview %321[%c0_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %323 = ttg.local_load %322 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %324 = ttg.memdesc_subview %321[%c8_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %325 = ttg.local_load %324 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %326 = ttg.memdesc_subview %321[%c16_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %327 = ttg.local_load %326 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %328 = ttg.memdesc_subview %321[%c24_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %329 = ttg.local_load %328 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %330 = ttg.memdesc_subview %321[%c32_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %331 = ttg.local_load %330 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %332 = ttg.memdesc_subview %321[%c40_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %333 = ttg.local_load %332 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %334 = ttg.memdesc_subview %321[%c48_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %335 = ttg.local_load %334 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %336 = ttg.memdesc_subview %321[%c56_i32, %c0_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %337 = ttg.local_load %336 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %338 = ttg.memdesc_subview %321[%c0_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %339 = ttg.local_load %338 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %340 = ttg.memdesc_subview %321[%c8_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %341 = ttg.local_load %340 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %342 = ttg.memdesc_subview %321[%c16_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %343 = ttg.local_load %342 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %344 = ttg.memdesc_subview %321[%c24_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %345 = ttg.local_load %344 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %346 = ttg.memdesc_subview %321[%c32_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %347 = ttg.local_load %346 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %348 = ttg.memdesc_subview %321[%c40_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %349 = ttg.local_load %348 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %350 = ttg.memdesc_subview %321[%c48_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %351 = ttg.local_load %350 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %352 = ttg.memdesc_subview %321[%c56_i32, %c32_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %353 = ttg.local_load %352 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %354 = ttg.memdesc_subview %321[%c0_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %355 = ttg.local_load %354 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %356 = ttg.memdesc_subview %321[%c8_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %357 = ttg.local_load %356 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %358 = ttg.memdesc_subview %321[%c16_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %359 = ttg.local_load %358 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %360 = ttg.memdesc_subview %321[%c24_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %361 = ttg.local_load %360 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %362 = ttg.memdesc_subview %321[%c32_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %363 = ttg.local_load %362 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %364 = ttg.memdesc_subview %321[%c40_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %365 = ttg.local_load %364 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %366 = ttg.memdesc_subview %321[%c48_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %367 = ttg.local_load %366 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %368 = ttg.memdesc_subview %321[%c56_i32, %c64_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %369 = ttg.local_load %368 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %370 = ttg.memdesc_subview %321[%c0_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %371 = ttg.local_load %370 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %372 = ttg.memdesc_subview %321[%c8_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %373 = ttg.local_load %372 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %374 = ttg.memdesc_subview %321[%c16_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %375 = ttg.local_load %374 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %376 = ttg.memdesc_subview %321[%c24_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %377 = ttg.local_load %376 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %378 = ttg.memdesc_subview %321[%c32_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %379 = ttg.local_load %378 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %380 = ttg.memdesc_subview %321[%c40_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %381 = ttg.local_load %380 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %382 = ttg.memdesc_subview %321[%c48_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %383 = ttg.local_load %382 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %384 = ttg.memdesc_subview %321[%c56_i32, %c96_i32] : !ttg.memdesc<64x128xf16, #shared2, #smem> -> !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128>
    %385 = ttg.local_load %384 : !ttg.memdesc<8x32xf16, #shared2, #smem, mutable, 64x128> -> tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %386 = amdgpu.extract_slice %302 [0, 0] : tensor<128x128xf32, #mma> to tensor<128x32xf32, #mma>
    %387 = amdgpu.extract_slice %302 [0, 32] : tensor<128x128xf32, #mma> to tensor<128x32xf32, #mma>
    %388 = amdgpu.extract_slice %302 [0, 64] : tensor<128x128xf32, #mma> to tensor<128x32xf32, #mma>
    %389 = amdgpu.extract_slice %302 [0, 96] : tensor<128x128xf32, #mma> to tensor<128x32xf32, #mma>
    %390 = tt.dot %313, %323, %386, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 0> [0], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %391 = tt.dot %313, %339, %387, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 0> [0], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %392 = tt.dot %313, %355, %388, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 0> [1], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %393 = tt.dot %313, %371, %389, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 0> [1], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %394 = tt.dot %314, %325, %390, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 1> [2], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %395 = tt.dot %314, %341, %391, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 1> [2], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %396 = tt.dot %314, %357, %392, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 1> [3], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %397 = tt.dot %314, %373, %393, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 1> [3], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %398 = tt.dot %315, %327, %394, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 2> [4], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %399 = tt.dot %315, %343, %395, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 2> [4], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %400 = tt.dot %315, %359, %396, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 2> [5], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %401 = tt.dot %315, %375, %397, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 2> [5], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %402 = tt.dot %316, %329, %398, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 3> [6], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %403 = tt.dot %316, %345, %399, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 3> [6], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %404 = tt.dot %316, %361, %400, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 3> [7], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %405 = tt.dot %316, %377, %401, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 3> [7], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %406 = tt.dot %317, %331, %402, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 4> [8], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %407 = tt.dot %317, %347, %403, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 4> [8], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %408 = tt.dot %317, %363, %404, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 4> [9], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %409 = tt.dot %317, %379, %405, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 4> [9], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %410 = tt.dot %318, %333, %406, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 5> [10], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %411 = tt.dot %318, %349, %407, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 5> [10], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %412 = tt.dot %318, %365, %408, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 5> [11], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %413 = tt.dot %318, %381, %409, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 5> [11], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %414 = tt.dot %319, %335, %410, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 6> [12], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %415 = tt.dot %319, %351, %411, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 6> [12], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %416 = tt.dot %319, %367, %412, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 6> [13], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %417 = tt.dot %319, %383, %413, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 6> [13], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %418 = tt.dot %320, %337, %414, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 7> [14], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %419 = tt.dot %320, %353, %415, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 0, 7> [14], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %420 = tt.dot %320, %369, %416, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 7> [15], <0, 0, 0> [0]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %421 = tt.dot %320, %385, %417, inputPrecision = tf32 {DotTile = #amdgpu<DotTile<0, 1, 7> [15], <0, 1, 0> [1]>} : tensor<128x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
    %422 = amdgpu.concat %418, %419, %420, %421 [1, 4] : tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma> -> tensor<128x128xf32, #mma>
    %423 = tt.addptr %100, %66 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    gpu.barrier
    %424 = ttg.local_alloc %112 {allocation.offset = 0 : i32} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %425 = arith.addi %95, %c64_i32 : i32
    cf.br ^bb3(%425, %422, %304, %223, %103, %423, %424 : i32, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)
  ^bb5:  // pred: ^bb3
    gpu.barrier
    %426 = ttg.local_load %101 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %427 = tt.load %100 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %428 = tt.dot %92, %426, %cst, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    %429 = arith.mulf %428, %cst_0 : tensor<128x64xf32, #mma>
    %430 = arith.addf %429, %cst : tensor<128x64xf32, #mma>
    %431 = "tt.reduce"(%430) <{axis = 1 : i32}> ({
    ^bb0(%arg27: f32, %arg28: f32):
      %464 = arith.maxnumf %arg27, %arg28 : f32
      tt.reduce.return %464 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %432 = arith.maxnumf %98, %431 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %433 = tt.expand_dims %432 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %434 = tt.broadcast %433 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %435 = arith.subf %430, %434 : tensor<128x64xf32, #mma>
    %436 = math.exp2 %435 : tensor<128x64xf32, #mma>
    %437 = "tt.reduce"(%436) <{axis = 1 : i32}> ({
    ^bb0(%arg27: f32, %arg28: f32):
      %464 = arith.addf %arg27, %arg28 : f32
      tt.reduce.return %464 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %438 = arith.subf %98, %432 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %439 = math.exp2 %438 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %440 = tt.expand_dims %439 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %441 = tt.broadcast %440 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %442 = arith.mulf %96, %441 : tensor<128x128xf32, #mma>
    %443 = arith.mulf %97, %439 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %444 = arith.addf %443, %437 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %445 = arith.truncf %436 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    %446 = ttg.convert_layout %445 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    gpu.barrier
    %447 = ttg.local_alloc %427 {allocation.offset = 0 : i32} : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
    gpu.barrier
    %448 = ttg.local_load %447 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %449 = tt.dot %446, %448, %442, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
    gpu.barrier
    %450 = tt.expand_dims %444 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %451 = arith.divf %cst_2, %450 : tensor<128x1xf32, #mma>
    %452 = tt.broadcast %451 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %453 = arith.mulf %449, %452 : tensor<128x128xf32, #mma>
    %454 = arith.truncf %453 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    cf.cond_br %76, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %455 = arith.subi %c8320_i32, %68 : i32
    %456 = tt.splat %455 : i32 -> tensor<128xi32, #blocked1>
    %457 = arith.cmpi slt, %6, %456 : tensor<128xi32, #blocked1>
    %458 = math.log2 %444 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %459 = arith.addf %432, %458 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %460 = ttg.convert_layout %459 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked1>
    tt.store %74, %460, %457 : tensor<128x!tt.ptr<f32>, #blocked1>
    cf.br ^bb8
  ^bb7:  // pred: ^bb5
    %461 = math.log2 %444 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %462 = arith.addf %432, %461 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %463 = ttg.convert_layout %462 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked1>
    tt.store %74, %463 : tensor<128x!tt.ptr<f32>, #blocked1>
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    tt.store %86, %454, %87 : tensor<128x128x!tt.ptr<f16>, #mma>
    cf.br ^bb1(%c1_i32 : i32)
  ^bb9:  // pred: ^bb1
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-amdgpu-reschedule-ops{arch=gfx942}, convert-triton-amdgpu-to-llvm{arch=gfx942 ftz=true}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-cf-to-llvm{index-bitwidth=0}, convert-arith-to-llvm{index-bitwidth=0}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, triton-amdgpu-lower-insert-instruction-sched-hints{arch=gfx942 num_stages=2}, enable-line-info, convert-builtin-func-to-llvm{ftz=true})",
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
Refining ops.
