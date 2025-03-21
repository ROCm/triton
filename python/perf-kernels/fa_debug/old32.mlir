TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (0, 1)
   register=8 -> (0, 2)
 - lane=1 -> (4, 0)
   lane=2 -> (8, 0)
   lane=4 -> (16, 0)
   lane=8 -> (32, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 64), dim0 (size 64)]
ensureLayoutNotLargerThan
dim1: 128 < 128
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (0, 1)
   register=8 -> (0, 2)
 - lane=1 -> (4, 0)
   lane=2 -> (8, 0)
   lane=4 -> (16, 0)
   lane=8 -> (32, 0)
   lane=16 -> (64, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (0, 8)
   warp=2 -> (0, 16)
where out dims are: [dim1 (size 128), dim0 (size 32)]
ensureLayoutNotLargerThan
dim1: 128 < 128
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 1
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (0, 1)
   register=8 -> (0, 2)
 - lane=1 -> (4, 0)
   lane=2 -> (8, 0)
   lane=4 -> (16, 0)
   lane=8 -> (32, 0)
   lane=16 -> (0, 4)
   lane=32 -> (0, 8)
 - warp=1 -> (0, 16)
   warp=2 -> (0, 32)
where out dims are: [dim1 (size 64), dim0 (size 64)]
ensureLayoutNotLargerThan
dim1: 64 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 3, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 1
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (0, 1)
   register=8 -> (0, 2)
 - lane=1 -> (4, 0)
   lane=2 -> (8, 0)
   lane=4 -> (16, 0)
   lane=8 -> (32, 0)
   lane=16 -> (64, 0)
   lane=32 -> (0, 4)
 - warp=1 -> (0, 8)
   warp=2 -> (0, 16)
where out dims are: [dim1 (size 128), dim0 (size 32)]
ensureLayoutNotLargerThan
dim1: 128 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 4, outValue: 64, actualSize: 128, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 3, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 64
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim0 (size 128), dim1 (size 16)]
ensureLayoutNotLargerThan
dim0: 128 < 128
dim1: 64 < 64
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim1 (size 128), dim0 (size 16)]
ensureLayoutNotLargerThan
dim1: 128 < 128
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (0, 1)
   register=2 -> (0, 2)
   register=4 -> (0, 4)
   register=8 -> (0, 16)
   register=16 -> (0, 32)
   register=32 -> (0, 64)
 - lane=1 -> (1, 0)
   lane=2 -> (2, 0)
   lane=4 -> (4, 0)
   lane=8 -> (8, 0)
   lane=16 -> (16, 0)
   lane=32 -> (0, 8)
 - warp=1 -> (32, 0)
   warp=2 -> (64, 0)
where out dims are: [dim0 (size 128), dim1 (size 128)]
ensureLayoutNotLargerThan
dim0: 128 < 128
dim1: 128 < 128
TritonGPUDialect::toLinearLayout()
AMDMfmaEncodingAttr::toLinearLayout
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
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 64
ctaLayout.toString()
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
ensureLayoutNotLargerThan
dim1: 64 < 64
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 64
ctaLayout.toString()
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
where out dims are: [dim0 (size 128), dim1 (size 64)]
ensureLayoutNotLargerThan
dim0: 128 < 128
dim1: 64 < 64
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 64
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim1 (size 128), dim0 (size 16)]
ensureLayoutNotLargerThan
dim1: 128 < 128
dim0: 64 < 64
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 64
dim1: 1 < 128
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (8, 0)
   register=8 -> (16, 0)
   register=16 -> (32, 0)
 - lane=1 -> (0, 1)
   lane=2 -> (0, 2)
   lane=4 -> (0, 4)
   lane=8 -> (0, 8)
   lane=16 -> (0, 16)
   lane=32 -> (4, 0)
 - warp=1 -> (64, 0)
   warp=2 -> (128, 0)
where out dims are: [dim0 (size 256), dim1 (size 32)]
ensureLayoutNotLargerThan
dim0: 256 < 64
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
		basisIdx6
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: warp, basisIdx: 1, outValue: 128, actualSize: 256, desiredSize: 64
		basis=0
		inDimName: warp, basisIdx: 0, outValue: 64, actualSize: 128, desiredSize: 64
		basis=0
		inDimName: register, basisIdx: 4, outValue: 32, actualSize: 64, desiredSize: 64
		DONE
dim1: 128 < 128
TritonGPUDialect::toLinearLayout()
TritonGPUDialect::toLinearLayout()
AMDMfmaEncodingAttr::toLinearLayout
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
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 1
ctaLayout.toString()
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
ensureLayoutNotLargerThan
dim1: 32 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: register, basisIdx: 3, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 2, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 5, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim0: 128 < 128
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 128
ctaLayout.toString()
 - register is a size 1 dimension
 - lane=1 -> (1)
   lane=2 -> (2)
   lane=4 -> (4)
   lane=8 -> (8)
   lane=16 -> (16)
   lane=32 -> (32)
 - warp=1 -> (64)
   warp=2 -> (128)
where out dims are: [dim0 (size 256)]
ensureLayoutNotLargerThan
dim0: 256 < 128
	inDimName: register
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: warp, basisIdx: 1, outValue: 128, actualSize: 256, desiredSize: 128
		basis=0
		inDimName: warp, basisIdx: 0, outValue: 64, actualSize: 128, desiredSize: 128
		DONE
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %171 = math.exp2 %170 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
TritonGPUDialect::toLinearLayout()
AMDMfmaEncodingAttr::toLinearLayout
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
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 8
ctaLayout.toString()
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
ensureLayoutNotLargerThan
dim1: 32 < 8
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: register, basisIdx: 3, outValue: 16, actualSize: 32, desiredSize: 8
		basis=0
		inDimName: register, basisIdx: 2, outValue: 8, actualSize: 16, desiredSize: 8
		basis=0
		inDimName: lane, basisIdx: 5, outValue: 4, actualSize: 8, desiredSize: 8
		DONE
dim0: 128 < 128
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %215 = "math.exp2"(%214) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %221 = "arith.truncf"(%212) : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %244 = "arith.addi"(%176, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %244 = "arith.addi"(%176, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %488 = "arith.maxnumf"(%arg43, %arg44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %488 = "arith.maxnumf"(%arg43, %arg44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %192 = "arith.maxnumf"(%179, %191) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %190 = "arith.addf"(%189, %3) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %511 = "arith.addf"(%arg41, %arg42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
[tritonamdgpu-refine-ops]: failed to refine binary op: %511 = "arith.addf"(%arg41, %arg42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %244 = "arith.addf"(%243, %237) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %219 = "arith.subf"(%214, %218) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %262 = "arith.subf"(%179, %216) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %189 = "arith.mulf"(%188, %4) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
srcShape: 128x64
srcShapePerCtaTile: 128x8
numReps: 1x8
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %290 = "arith.mulf"(%177, %289) <{fastmath = #arith.fastmath<none>}> : (tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>, tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>) -> tensor<128x128xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
TritonGPUDialect::toLinearLayout()
AMDMfmaEncodingAttr::toLinearLayout
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
ensureLayoutNotLargerThan
dim0: 1 < 128
dim1: 1 < 128
ctaLayout.toString()
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
ensureLayoutNotLargerThan
dim1: 128 < 128
dim0: 128 < 128
srcShape: 128x128
srcShapePerCtaTile: 128x8
numReps: 1x16
Examining refined shape as LinearLayout
#ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %339 = "arith.mulf"(%178, %287) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
srcShape: 128
srcShapePerCtaTile: 128
numReps: 1
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %442 = "arith.addi"(%433, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
[tritonamdgpu-refine-ops]: failed to refine binary op: %442 = "arith.addi"(%433, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %446 = "arith.addi"(%443, %38) <{overflowFlags = #arith.overflow<none>}> : (tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>) -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
TritonGPUDialect::toLinearLayout()
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 1
dim1: 1 < 64
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim0 (size 128), dim1 (size 16)]
ensureLayoutNotLargerThan
dim0: 128 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 3, outValue: 64, actualSize: 128, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 2, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim1: 64 < 64
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
Examining refined shape as LinearLayout
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
TritonGPUDialect::toLinearLayout()
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 1
dim1: 1 < 16
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim0 (size 128), dim1 (size 16)]
ensureLayoutNotLargerThan
dim0: 128 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 3, outValue: 64, actualSize: 128, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 2, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim1: 16 < 16
#ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %459 = "arith.addi"(%445, %40) <{overflowFlags = #arith.overflow<none>}> : (tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
TritonGPUDialect::toLinearLayout()
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 64
dim1: 1 < 1
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim1 (size 128), dim0 (size 16)]
ensureLayoutNotLargerThan
dim1: 128 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 3, outValue: 64, actualSize: 128, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 2, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim0: 64 < 64
srcShape: 64
srcShapePerCtaTile: 16
numReps: 4
Examining refined shape as LinearLayout
#ttg.linear<{register = [[16], [32]], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
TritonGPUDialect::toLinearLayout()
TritonGPUDialect::toLinearLayout()
ensureLayoutNotLargerThan
dim0: 1 < 16
dim1: 1 < 1
ctaLayout.toString()
 - register=1 -> (1, 0)
   register=2 -> (2, 0)
   register=4 -> (4, 0)
 - lane=1 -> (8, 0)
   lane=2 -> (16, 0)
   lane=4 -> (32, 0)
   lane=8 -> (64, 0)
   lane=16 -> (0, 1)
   lane=32 -> (0, 2)
 - warp=1 -> (0, 4)
   warp=2 -> (0, 8)
where out dims are: [dim1 (size 128), dim0 (size 16)]
ensureLayoutNotLargerThan
dim1: 128 < 1
	inDimName: register
		basisIdx0
		basisIdx1
		basisIdx2
	inDimName: lane
		basisIdx0
		basisIdx1
		basisIdx2
		basisIdx3
		basisIdx4
		basisIdx5
	inDimName: warp
		basisIdx0
		basisIdx1
	FinalSelection of basis vectors
		inDimName: lane, basisIdx: 3, outValue: 64, actualSize: 128, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 2, outValue: 32, actualSize: 64, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 1, outValue: 16, actualSize: 32, desiredSize: 1
		basis=0
		inDimName: lane, basisIdx: 0, outValue: 8, actualSize: 16, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 2, outValue: 4, actualSize: 8, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 1, outValue: 2, actualSize: 4, desiredSize: 1
		basis=0
		inDimName: register, basisIdx: 0, outValue: 1, actualSize: 2, desiredSize: 1
		basis=0
dim0: 16 < 16
#ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1], [2]], warp = [[4], [8]], block = []}>
[tritonamdgpu-refine-ops]: rewriteElementWiseOp(): %482 = "arith.andi"(%481, %416) : (i1, i1) -> i1
[tritonamdgpu-refine-ops]: failed to refine binary op: %482 = "arith.andi"(%481, %416) : (i1, i1) -> i1
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:291:34: error: result layout must match source layout
            qk += (tl.dot(q, k) * QK_SCALE)
                                 ^
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:754:52: note: called from
                                                    INT8_KV)
                                                   ^
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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
    scf.while (%arg29 = %c0_i32) : (i32) -> () {
      %20 = arith.cmpi slt, %arg29, %c1_i32 : i32
      scf.condition(%20)
    } do {
      %20 = tt.load %16 : !tt.ptr<i32>
      %21 = tt.load %17 : !tt.ptr<i32>
      %22 = arith.subi %21, %20 : i32
      %23 = arith.cmpi sle, %3, %22 : i32
      %24 = tt.load %18 : !tt.ptr<i32>
      %25 = tt.load %19 : !tt.ptr<i32>
      %26 = arith.subi %25, %24 : i32
      scf.if %23 {
        %27 = arith.addi %26, %c63_i32 : i32
        %28 = arith.divsi %27, %c64_i32 : i32
        %29 = arith.addi %0, %c1_i32 : i32
        %30 = arith.muli %29, %c128_i32 : i32
        %31 = arith.addi %30, %26 : i32
        %32 = arith.subi %31, %22 : i32
        %33 = arith.addi %32, %c63_i32 : i32
        %34 = arith.divsi %33, %c64_i32 : i32
        %35 = arith.minsi %28, %34 : i32
        %36 = arith.cmpi sle, %35, %c0_i32 : i32
        %37 = arith.cmpi sgt, %35, %c0_i32 : i32
        scf.if %36 {
          %38 = arith.muli %2, %arg14 : i32
          %39 = tt.addptr %arg4, %38 : !tt.ptr<f16>, i32
          %40 = arith.muli %1, %arg15 : i32
          %41 = tt.addptr %39, %40 : !tt.ptr<f16>, i32
          %42 = arith.muli %20, %arg16 : i32
          %43 = tt.addptr %41, %42 : !tt.ptr<f16>, i32
          %44 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
          %45 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked>
          %46 = arith.muli %44, %45 : tensor<128x1xi32, #blocked>
          %47 = tt.splat %43 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
          %48 = tt.addptr %47, %46 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
          %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %50 = tt.expand_dims %49 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
          %51 = tt.broadcast %48 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
          %52 = tt.broadcast %50 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
          %53 = tt.addptr %51, %52 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
          %54 = tt.splat %22 : i32 -> tensor<128x1xi32, #blocked>
          %55 = arith.cmpi slt, %44, %54 : tensor<128x1xi32, #blocked>
          %56 = tt.broadcast %55 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
          tt.store %53, %cst_11, %56 : tensor<128x128x!tt.ptr<f16>, #blocked>
          %57 = arith.muli %2, %c54848_i32 : i32
          %58 = tt.addptr %arg3, %57 : !tt.ptr<f32>, i32
          %59 = arith.muli %1, %c3428_i32 : i32
          %60 = tt.addptr %58, %59 : !tt.ptr<f32>, i32
          %61 = tt.splat %60 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
          %62 = tt.addptr %61, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
          %63 = arith.cmpi slt, %12, %cst_13 : tensor<128xi32, #blocked2>
          tt.store %62, %cst_12, %63 : tensor<128x!tt.ptr<f32>, #blocked2>
        }
        scf.if %37 {
          %38 = arith.cmpi slt, %26, %c64_i32 : i32
          %39 = scf.if %38 -> (i32) {
            %145 = arith.subi %c64_i32, %26 : i32
            scf.yield %145 : i32
          } else {
            %145 = arith.remsi %26, %c64_i32 : i32
            scf.yield %145 : i32
          }
          %40 = arith.muli %2, %arg5 : i32
          %41 = tt.addptr %arg0, %40 : !tt.ptr<f16>, i32
          %42 = arith.muli %1, %arg6 : i32
          %43 = tt.addptr %41, %42 : !tt.ptr<f16>, i32
          %44 = arith.muli %20, %arg7 : i32
          %45 = tt.addptr %43, %44 : !tt.ptr<f16>, i32
          %46 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
          %47 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
          %48 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
          %49 = arith.muli %47, %48 : tensor<128x1xi32, #blocked>
          %50 = tt.splat %45 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
          %51 = tt.addptr %50, %49 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
          %52 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %53 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %54 = tt.expand_dims %52 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
          %55 = tt.expand_dims %53 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
          %56 = tt.broadcast %51 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
          %57 = tt.broadcast %54 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
          %58 = tt.broadcast %55 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
          %59 = tt.addptr %56, %58 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
          %60 = arith.muli %2, %arg8 : i32
          %61 = tt.addptr %arg1, %60 : !tt.ptr<f16>, i32
          %62 = arith.muli %1, %arg9 : i32
          %63 = tt.addptr %61, %62 : !tt.ptr<f16>, i32
          %64 = arith.muli %24, %arg10 : i32
          %65 = tt.addptr %63, %64 : !tt.ptr<f16>, i32
          %66 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %67 = tt.expand_dims %66 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
          %68 = tt.splat %65 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
          %69 = tt.addptr %68, %67 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
          %70 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
          %71 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
          %72 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked1>
          %73 = arith.muli %70, %72 : tensor<1x64xi32, #blocked1>
          %74 = tt.broadcast %69 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
          %75 = tt.broadcast %73 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
          %76 = tt.addptr %74, %75 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
          %77 = arith.muli %2, %arg11 : i32
          %78 = tt.addptr %arg2, %77 : !tt.ptr<f16>, i32
          %79 = arith.muli %1, %arg12 : i32
          %80 = tt.addptr %78, %79 : !tt.ptr<f16>, i32
          %81 = arith.muli %24, %arg13 : i32
          %82 = tt.addptr %80, %81 : !tt.ptr<f16>, i32
          %83 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
          %84 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked>
          %85 = arith.muli %83, %84 : tensor<64x1xi32, #blocked>
          %86 = tt.splat %82 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
          %87 = tt.addptr %86, %85 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
          %88 = tt.broadcast %87 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
          %89 = tt.broadcast %55 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
          %90 = tt.addptr %88, %89 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
          %91 = tt.splat %22 : i32 -> tensor<128x1xi32, #mma>
          %92 = tt.splat %22 : i32 -> tensor<128x1xi32, #blocked>
          %93 = arith.cmpi slt, %46, %91 : tensor<128x1xi32, #mma>
          %94 = arith.cmpi slt, %47, %92 : tensor<128x1xi32, #blocked>
          %95 = tt.broadcast %93 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
          %96 = tt.broadcast %94 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
          %97 = tt.load %59, %96, %cst_11 : tensor<128x128x!tt.ptr<f16>, #blocked>
          %98 = ttg.local_alloc %97 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
          %99 = ttg.local_load %98 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
          %100 = arith.cmpi eq, %39, %c0_i32 : i32
          %101 = arith.remsi %22, %c128_i32 : i32
          %102 = arith.cmpi eq, %101, %c0_i32 : i32
          %103 = arith.andi %100, %102 : i1
          %104 = arith.xori %103, %true : i1
          %105 = arith.extui %104 : i1 to i32
          %106 = arith.addi %105, %c2_i32 : i32
          %107 = arith.minsi %106, %35 : i32
          %108 = arith.subi %35, %107 : i32
          %109 = arith.muli %35, %c64_i32 : i32
          %110 = arith.cmpi sgt, %108, %c0_i32 : i32
          %111:4 = scf.if %110 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32) {
            %145 = arith.muli %108, %c64_i32 : i32
            %146 = arith.muli %arg10, %c64_i32 : i32
            %147 = tt.splat %146 : i32 -> tensor<128x64xi32, #blocked1>
            %148 = arith.muli %arg13, %c64_i32 : i32
            %149 = tt.splat %148 : i32 -> tensor<64x128xi32, #blocked>
            %150 = arith.cmpi sgt, %145, %c0_i32 : i32
            %151 = tt.splat %150 : i1 -> tensor<128x64xi1, #blocked1>
            %152 = tt.load %76, %151 : tensor<128x64x!tt.ptr<f16>, #blocked1>
            %153 = ttg.local_alloc %152 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
            %154 = arith.subi %145, %c64_i32 : i32
            %155:6 = scf.for %arg29 = %c0_i32 to %154 step %c64_i32 iter_args(%arg30 = %cst_4, %arg31 = %cst_7, %arg32 = %cst_8, %arg33 = %76, %arg34 = %90, %arg35 = %153) -> (tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)  : i32 {
              amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
              %187 = tt.addptr %arg33, %147 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
              %188 = tt.load %187 : tensor<128x64x!tt.ptr<f16>, #blocked1>
              %189 = ttg.local_load %arg35 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
              %190 = tt.load %arg34 : tensor<64x128x!tt.ptr<f16>, #blocked>
              %191 = tt.dot %99, %189, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
              %192 = arith.mulf %191, %cst_3 : tensor<128x64xf32, #mma>
              %193 = arith.addf %192, %cst_2 : tensor<128x64xf32, #mma>
              %194 = "tt.reduce"(%193) <{axis = 1 : i32}> ({
              ^bb0(%arg36: f32, %arg37: f32):
                %215 = arith.maxnumf %arg36, %arg37 : f32
                tt.reduce.return %215 : f32
              }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %195 = arith.maxnumf %arg32, %194 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %196 = tt.expand_dims %195 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
              %197 = tt.broadcast %196 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
              %198 = arith.subf %193, %197 : tensor<128x64xf32, #mma>
              %199 = math.exp2 %198 : tensor<128x64xf32, #mma>
              %200 = "tt.reduce"(%199) <{axis = 1 : i32}> ({
              ^bb0(%arg36: f32, %arg37: f32):
                %215 = arith.addf %arg36, %arg37 : f32
                tt.reduce.return %215 : f32
              }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %201 = arith.subf %arg32, %195 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %202 = math.exp2 %201 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %203 = tt.expand_dims %202 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
              %204 = tt.broadcast %203 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
              %205 = arith.mulf %arg30, %204 : tensor<128x128xf32, #mma>
              %206 = arith.mulf %arg31, %202 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %207 = arith.addf %206, %200 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %208 = arith.truncf %199 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
              %209 = ttg.convert_layout %208 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
              %210 = ttg.local_alloc %190 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
              %211 = ttg.local_load %210 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
              %212 = tt.dot %209, %211, %205, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
              %213 = tt.addptr %arg34, %149 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
              %214 = ttg.local_alloc %188 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
              scf.yield %212, %207, %195, %187, %213, %214 : tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
            }
            %156 = arith.addi %145, %c63_i32 : i32
            %157 = arith.divsi %156, %c64_i32 : i32
            %158 = arith.cmpi sge, %157, %c1_i32 : i32
            %159 = ttg.local_load %155#5 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %160 = tt.splat %158 : i1 -> tensor<64x128xi1, #blocked>
            %161 = tt.load %155#4, %160 : tensor<64x128x!tt.ptr<f16>, #blocked>
            %162 = scf.if %158 -> (tensor<128x64xf32, #mma>) {
              %187 = tt.dot %99, %159, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
              scf.yield %187 : tensor<128x64xf32, #mma>
            } else {
              scf.yield %cst_2 : tensor<128x64xf32, #mma>
            }
            %163 = arith.mulf %162, %cst_3 : tensor<128x64xf32, #mma>
            %164 = arith.addf %163, %cst_2 : tensor<128x64xf32, #mma>
            %165 = "tt.reduce"(%164) <{axis = 1 : i32}> ({
            ^bb0(%arg29: f32, %arg30: f32):
              %187 = arith.maxnumf %arg29, %arg30 : f32
              tt.reduce.return %187 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %166 = arith.maxnumf %155#2, %165 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %167 = tt.expand_dims %166 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %168 = tt.broadcast %167 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %169 = arith.subf %164, %168 : tensor<128x64xf32, #mma>
            %170 = math.exp2 %169 : tensor<128x64xf32, #mma>
            %171 = "tt.reduce"(%170) <{axis = 1 : i32}> ({
            ^bb0(%arg29: f32, %arg30: f32):
              %187 = arith.addf %arg29, %arg30 : f32
              tt.reduce.return %187 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %172 = arith.subf %155#2, %166 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %173 = math.exp2 %172 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %174 = tt.expand_dims %173 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %175 = tt.broadcast %174 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
            %176 = arith.mulf %155#0, %175 : tensor<128x128xf32, #mma>
            %177 = arith.mulf %155#1, %173 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %178 = arith.addf %177, %171 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %179 = arith.truncf %170 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %180 = ttg.convert_layout %179 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %181 = ttg.local_alloc %161 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
            %182 = ttg.local_load %181 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %183 = scf.if %158 -> (tensor<128x128xf32, #mma>) {
              %187 = tt.dot %180, %182, %176, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
              scf.yield %187 : tensor<128x128xf32, #mma>
            } else {
              scf.yield %176 : tensor<128x128xf32, #mma>
            }
            %184 = arith.select %158, %183, %155#0 : tensor<128x128xf32, #mma>
            %185 = arith.select %158, %178, %155#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %186 = arith.select %158, %166, %155#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            scf.yield %186, %185, %184, %145 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32
          } else {
            scf.yield %cst_8, %cst_7, %cst_4, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, i32
          }
          gpu.barrier
          %112 = arith.cmpi sgt, %107, %c0_i32 : i32
          %113:3 = scf.if %112 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>) {
            %145 = arith.subi %22, %26 : i32
            %146 = tt.splat %145 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %147 = arith.addi %14, %146 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %148 = arith.muli %108, %c64_i32 : i32
            %149 = arith.muli %148, %arg10 : i32
            %150 = tt.splat %149 : i32 -> tensor<128x64xi32, #blocked1>
            %151 = tt.addptr %76, %150 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
            %152 = arith.muli %148, %arg13 : i32
            %153 = tt.splat %152 : i32 -> tensor<64x128xi32, #blocked>
            %154 = tt.addptr %90, %153 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
            %155 = tt.splat %26 : i32 -> tensor<1x64xi32, #blocked1>
            %156 = tt.splat %26 : i32 -> tensor<1x64xi32, #mma>
            %157 = tt.splat %26 : i32 -> tensor<64x1xi32, #blocked>
            %158 = arith.cmpi ne, %39, %c0_i32 : i32
            %159 = tt.broadcast %46 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
            %160 = arith.muli %arg10, %c64_i32 : i32
            %161 = tt.splat %160 : i32 -> tensor<128x64xi32, #blocked1>
            %162 = arith.muli %arg13, %c64_i32 : i32
            %163 = tt.splat %162 : i32 -> tensor<64x128xi32, #blocked>
            %164 = arith.cmpi slt, %111#3, %109 : i32
            %165 = tt.splat %111#3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
            %166 = arith.addi %165, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
            %167 = tt.expand_dims %166 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
            %168 = arith.cmpi slt, %167, %155 : tensor<1x64xi32, #blocked1>
            %169 = tt.broadcast %168 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
            %170 = tt.splat %164 : i1 -> tensor<128x64xi1, #blocked1>
            %171 = arith.andi %170, %169 : tensor<128x64xi1, #blocked1>
            %172 = tt.load %151, %171, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
            %173 = ttg.local_alloc %172 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
            %174 = arith.subi %109, %c64_i32 : i32
            %175:6 = scf.for %arg29 = %111#3 to %174 step %c64_i32 iter_args(%arg30 = %111#2, %arg31 = %111#1, %arg32 = %111#0, %arg33 = %151, %arg34 = %154, %arg35 = %173) -> (tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>)  : i32 {
              amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
              %228 = tt.addptr %arg33, %161 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
              %229 = arith.addi %arg29, %c64_i32 : i32
              %230 = tt.splat %229 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
              %231 = tt.splat %arg29 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
              %232 = tt.splat %arg29 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %233 = arith.addi %230, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
              %234 = arith.addi %232, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %235 = tt.expand_dims %233 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
              %236 = arith.cmpi slt, %235, %155 : tensor<1x64xi32, #blocked1>
              %237 = tt.broadcast %236 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
              %238 = tt.load %228, %237, %cst_10 : tensor<128x64x!tt.ptr<f16>, #blocked1>
              %239 = ttg.local_load %arg35 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
              %240 = tt.expand_dims %234 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
              %241 = arith.cmpi slt, %240, %157 : tensor<64x1xi32, #blocked>
              %242 = tt.broadcast %241 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
              %243 = tt.load %arg34, %242, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
              %244 = arith.cmpi eq, %229, %109 : i32
              %245 = arith.andi %244, %158 : i1
              %246 = scf.if %245 -> (tensor<128x64xf32, #mma>) {
                %276 = tt.splat %arg29 : i32 -> tensor<1x64xi32, #mma>
                %277 = arith.addi %276, %71 : tensor<1x64xi32, #mma>
                %278 = arith.cmpi slt, %277, %156 : tensor<1x64xi32, #mma>
                %279 = arith.select %278, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
                %280 = tt.broadcast %279 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
                scf.yield %280 : tensor<128x64xf32, #mma>
              } else {
                scf.yield %cst_2 : tensor<128x64xf32, #mma>
              }
              %247 = arith.addi %231, %147 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
              %248 = tt.expand_dims %247 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
              %249 = tt.broadcast %248 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
              %250 = arith.cmpi sge, %159, %249 : tensor<128x64xi32, #mma>
              %251 = arith.select %250, %246, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
              %252 = tt.dot %99, %239, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
              %253 = arith.mulf %252, %cst_3 : tensor<128x64xf32, #mma>
              %254 = arith.addf %251, %253 : tensor<128x64xf32, #mma>
              %255 = "tt.reduce"(%254) <{axis = 1 : i32}> ({
              ^bb0(%arg36: f32, %arg37: f32):
                %276 = arith.maxnumf %arg36, %arg37 : f32
                tt.reduce.return %276 : f32
              }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %256 = arith.maxnumf %arg32, %255 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %257 = tt.expand_dims %256 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
              %258 = tt.broadcast %257 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
              %259 = arith.subf %254, %258 : tensor<128x64xf32, #mma>
              %260 = math.exp2 %259 : tensor<128x64xf32, #mma>
              %261 = "tt.reduce"(%260) <{axis = 1 : i32}> ({
              ^bb0(%arg36: f32, %arg37: f32):
                %276 = arith.addf %arg36, %arg37 : f32
                tt.reduce.return %276 : f32
              }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %262 = arith.subf %arg32, %256 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %263 = math.exp2 %262 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %264 = tt.expand_dims %263 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
              %265 = tt.broadcast %264 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
              %266 = arith.mulf %arg30, %265 : tensor<128x128xf32, #mma>
              %267 = arith.mulf %arg31, %263 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %268 = arith.addf %267, %261 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
              %269 = arith.truncf %260 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
              %270 = ttg.convert_layout %269 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
              %271 = ttg.local_alloc %243 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
              %272 = ttg.local_load %271 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
              %273 = tt.dot %270, %272, %266, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
              %274 = tt.addptr %arg34, %163 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
              %275 = ttg.local_alloc %238 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
              scf.yield %273, %268, %256, %228, %274, %275 : tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
            }
            %176 = arith.subi %109, %111#3 : i32
            %177 = arith.addi %176, %c63_i32 : i32
            %178 = arith.divsi %177, %c64_i32 : i32
            %179 = arith.subi %178, %c1_i32 : i32
            %180 = arith.maxsi %179, %c0_i32 : i32
            %181 = arith.muli %180, %c64_i32 : i32
            %182 = arith.addi %111#3, %181 : i32
            %183 = arith.cmpi sge, %178, %c1_i32 : i32
            %184 = tt.splat %182 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %185 = tt.splat %182 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %186 = arith.addi %185, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %187 = ttg.local_load %175#5 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %188 = tt.expand_dims %186 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
            %189 = arith.cmpi slt, %188, %157 : tensor<64x1xi32, #blocked>
            %190 = tt.broadcast %189 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
            %191 = tt.splat %183 : i1 -> tensor<64x128xi1, #blocked>
            %192 = arith.andi %191, %190 : tensor<64x128xi1, #blocked>
            %193 = tt.load %175#4, %192, %cst_9 : tensor<64x128x!tt.ptr<f16>, #blocked>
            %194 = arith.addi %182, %c64_i32 : i32
            %195 = arith.cmpi eq, %194, %109 : i32
            %196 = arith.andi %195, %158 : i1
            %197 = scf.if %196 -> (tensor<128x64xf32, #mma>) {
              %228 = tt.splat %182 : i32 -> tensor<1x64xi32, #mma>
              %229 = arith.addi %228, %71 : tensor<1x64xi32, #mma>
              %230 = arith.cmpi slt, %229, %156 : tensor<1x64xi32, #mma>
              %231 = arith.select %230, %cst_0, %cst : tensor<1x64xi1, #mma>, tensor<1x64xf32, #mma>
              %232 = tt.broadcast %231 : tensor<1x64xf32, #mma> -> tensor<128x64xf32, #mma>
              scf.yield %232 : tensor<128x64xf32, #mma>
            } else {
              scf.yield %cst_2 : tensor<128x64xf32, #mma>
            }
            %198 = arith.addi %184, %147 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %199 = tt.expand_dims %198 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
            %200 = tt.broadcast %199 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
            %201 = arith.cmpi sge, %159, %200 : tensor<128x64xi32, #mma>
            %202 = arith.select %201, %197, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
            %203 = scf.if %183 -> (tensor<128x64xf32, #mma>) {
              %228 = tt.dot %99, %187, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
              scf.yield %228 : tensor<128x64xf32, #mma>
            } else {
              scf.yield %cst_2 : tensor<128x64xf32, #mma>
            }
            %204 = arith.mulf %203, %cst_3 : tensor<128x64xf32, #mma>
            %205 = arith.addf %202, %204 : tensor<128x64xf32, #mma>
            %206 = "tt.reduce"(%205) <{axis = 1 : i32}> ({
            ^bb0(%arg29: f32, %arg30: f32):
              %228 = arith.maxnumf %arg29, %arg30 : f32
              tt.reduce.return %228 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %207 = arith.maxnumf %175#2, %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %208 = tt.expand_dims %207 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %209 = tt.broadcast %208 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %210 = arith.subf %205, %209 : tensor<128x64xf32, #mma>
            %211 = math.exp2 %210 : tensor<128x64xf32, #mma>
            %212 = "tt.reduce"(%211) <{axis = 1 : i32}> ({
            ^bb0(%arg29: f32, %arg30: f32):
              %228 = arith.addf %arg29, %arg30 : f32
              tt.reduce.return %228 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %213 = arith.subf %175#2, %207 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %214 = math.exp2 %213 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %215 = tt.expand_dims %214 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %216 = tt.broadcast %215 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
            %217 = arith.mulf %175#0, %216 : tensor<128x128xf32, #mma>
            %218 = arith.mulf %175#1, %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %219 = arith.addf %218, %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %220 = arith.truncf %211 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %221 = ttg.convert_layout %220 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %222 = ttg.local_alloc %193 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared2, #smem>
            %223 = ttg.local_load %222 : !ttg.memdesc<64x128xf16, #shared2, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %224 = scf.if %183 -> (tensor<128x128xf32, #mma>) {
              %228 = tt.dot %221, %223, %217, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
              scf.yield %228 : tensor<128x128xf32, #mma>
            } else {
              scf.yield %217 : tensor<128x128xf32, #mma>
            }
            %225 = arith.select %183, %224, %175#0 : tensor<128x128xf32, #mma>
            %226 = arith.select %183, %219, %175#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %227 = arith.select %183, %207, %175#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            scf.yield %227, %226, %225 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>
          } else {
            scf.yield %111#0, %111#1, %111#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>
          }
          %114 = tt.expand_dims %113#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %115 = arith.divf %cst_6, %114 : tensor<128x1xf32, #mma>
          %116 = tt.broadcast %115 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
          %117 = arith.mulf %113#2, %116 : tensor<128x128xf32, #mma>
          %118 = arith.subi %22, %26 : i32
          %119 = arith.truncf %117 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
          %120 = arith.cmpi sgt, %118, %3 : i32
          %121 = arith.cmpi slt, %118, %30 : i32
          %122 = arith.andi %120, %121 : i1
          %123 = scf.if %122 -> (tensor<128x128xf16, #mma>) {
            %145 = tt.splat %118 : i32 -> tensor<128x1xi32, #mma>
            %146 = arith.cmpi sge, %46, %145 : tensor<128x1xi32, #mma>
            %147 = tt.broadcast %146 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
            %148 = arith.select %147, %119, %cst_5 : tensor<128x128xi1, #mma>, tensor<128x128xf16, #mma>
            scf.yield %148 : tensor<128x128xf16, #mma>
          } else {
            scf.yield %119 : tensor<128x128xf16, #mma>
          }
          %124 = arith.muli %2, %c54848_i32 : i32
          %125 = tt.addptr %arg3, %124 : !tt.ptr<f32>, i32
          %126 = arith.muli %1, %c3428_i32 : i32
          %127 = tt.addptr %125, %126 : !tt.ptr<f32>, i32
          %128 = tt.splat %127 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
          %129 = tt.addptr %128, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
          %130 = arith.subi %30, %22 : i32
          %131 = arith.cmpi sgt, %130, %c0_i32 : i32
          scf.if %131 {
            %145 = arith.subi %c128_i32, %130 : i32
            %146 = tt.splat %145 : i32 -> tensor<128xi32, #blocked2>
            %147 = arith.cmpi slt, %6, %146 : tensor<128xi32, #blocked2>
            %148 = math.log2 %113#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %149 = arith.addf %113#0, %148 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %150 = ttg.convert_layout %149 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
            tt.store %129, %150, %147 : tensor<128x!tt.ptr<f32>, #blocked2>
          } else {
            %145 = math.log2 %113#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %146 = arith.addf %113#0, %145 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %147 = ttg.convert_layout %146 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
            tt.store %129, %147 : tensor<128x!tt.ptr<f32>, #blocked2>
          }
          %132 = arith.muli %2, %arg14 : i32
          %133 = tt.addptr %arg4, %132 : !tt.ptr<f16>, i32
          %134 = arith.muli %1, %arg15 : i32
          %135 = tt.addptr %133, %134 : !tt.ptr<f16>, i32
          %136 = arith.muli %20, %arg16 : i32
          %137 = tt.addptr %135, %136 : !tt.ptr<f16>, i32
          %138 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
          %139 = arith.muli %46, %138 : tensor<128x1xi32, #mma>
          %140 = tt.splat %137 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
          %141 = tt.addptr %140, %139 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
          %142 = tt.broadcast %141 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #mma>
          %143 = tt.addptr %142, %57 : tensor<128x128x!tt.ptr<f16>, #mma>, tensor<128x128xi32, #mma>
          %144 = arith.select %131, %95, %cst_14 : tensor<128x128xi1, #mma>
          tt.store %143, %123, %144 : tensor<128x128x!tt.ptr<f16>, #mma>
        }
      }
      scf.yield %c1_i32 : i32
    }
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(decompose-unsupported-amd-conversions{arch=gfx942}, optimize-amd-lds-usage{lds-limit=0 target-arch=gfx942}, convert-scf-to-cf, convert-index-to-llvm{index-bitwidth=0}, allocate-shared-memory, triton-amdgpu-membar-analysis, triton-amdgpu-refine-ops{arch=gfx942}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:527:0: error: Failures have been detected while processing an MLIR pass pipeline
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:527:0: note: Pipeline failed while executing [`TritonAMDGPURefineOps` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
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
  File "/home/dtanner/repos/triton/python/triton/backends/amd/compiler.py", line 296, in make_llir
    pm.run(mod)
RuntimeError: PassManager::run failed
