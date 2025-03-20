// #include "TritonAMDGPUToLLVM/Passes.h"

// #include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
// #include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// #include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
// #include "triton/Analysis/Allocation.h"
// #include "triton/Analysis/AxisInfo.h"
// #include "triton/Analysis/Membar.h"
// #include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
// #include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/DotTiling.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef LLVM_DEBUG
#define LLVM_DEBUG(X) X

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-refine-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define PRINT_SMALL_VECTOR(X) \
  { \
    std::string dbgStr; \
    llvm::raw_string_ostream dbgStream(dbgStr); \
    dbgStream << #X << ": "; \
    for (auto i = 0; i < X.size(); ++i) { \
      dbgStream << X[i]; \
      if (i < X.size()-1) { \
        dbgStream << "x"; \
      } \
    } \
    llvm::dbgs() << dbgStream.str() << "\n"; \
  }

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {


SmallVector<unsigned> getRefinedShapeElementWise(RankedTensorType type) {
  return ttg::toLinearEncoding(type).getShapePerCTATile();
}

SmallVector<unsigned> getRefinedShapeConvertLayout(RankedTensorType type) {
  return ttg::toLinearEncoding(type).getShapePerCTATile();
}

SmallVector<unsigned> getRefinedShapeReduce(RankedTensorType type) {
  return ttg::toLinearEncoding(type).getShapePerCTATile();
}

SmallVector<unsigned> getRefinedShapeExpandDims(RankedTensorType type) {
  return ttg::toLinearEncoding(type).getShapePerCTATile();
}

SmallVector<unsigned> getRefinedShapeBroadcast(RankedTensorType type) {
  return ttg::toLinearEncoding(type).getShapePerCTATile();
}

// TODO: take the implementation from `ReorderInstructions.cpp`
static SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

SmallVector<Value> createOffset(llvm::ArrayRef<Value> valueOffset,
                                llvm::ArrayRef<int64_t> intOffset,
                                OpBuilder &rewriter, Location loc) {
  SmallVector<Value> values;
  for (auto item : valueOffset) {
    values.push_back(item);
  }

  for (auto item : intOffset) {
    Value value = rewriter.create<arith::ConstantIntOp>(loc, item, 32);
    values.push_back(value);
  }
  return values;
}

class CoordinateAux {
public:
  CoordinateAux(llvm::ArrayRef<int64_t> layout) : layout(layout) {
    bounds.resize(layout.size());
    std::exclusive_scan(layout.rbegin(), layout.rend(), bounds.begin(), 1,
                        std::multiplies<>());
  }

  SmallVector<int64_t> map(int64_t index) {
    SmallVector<int64_t> coords(bounds.size(), 0);
    for (size_t i = 1; i < bounds.size(); ++i) {
      size_t d = bounds.size() - i;
      coords[d] = index / bounds[d];
      index = index % bounds[d];
    }
    coords[0] = index;
    std::reverse(coords.begin(), coords.end());
    return coords;
  }

private:
  llvm::ArrayRef<int64_t> layout;
  std::vector<int> bounds;
};

inline bool isRowMajor(::llvm::ArrayRef<unsigned> order) {
  auto rank = order.size();
  return order[rank - 1] == 0;
}

LogicalResult rewriteLocalLoad(OpBuilder &rewriter,
                               triton::gpu::LocalLoadOp op) {
  auto *ctx = op->getContext();
  auto loc = op->getLoc();

  auto resultType = cast<RankedTensorType>(op.getType());
  auto resultElementType = resultType.getElementType();
  auto resultEncode = cast<DotOperandEncodingAttr>(resultType.getEncoding());
  auto resultShape = resultType.getShape();

  const auto rank = resultShape.size();
  assert(rank == 2);

  auto opIdx = resultEncode.getOpIdx();
  const int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  const int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(resultEncode.getParent());
  int kWidth = resultEncode.getKWidth();
  auto numReps = mfmaLayout.getRepForOperand(resultShape, kWidth, opIdx);

  // indices into 3D numReps
  int kRepsIdx = opIdx == 0 ? 2 : 1;
  int nonKRepsIdx = opIdx == 0 ? 1 : 2;
  int bRepsIdx = 0;

  // 2D shape which drops batch dimension.
  SmallVector<int64_t> numReps2D = {numReps[1], numReps[2]};

  auto numRepsNonK = numReps[nonKRepsIdx];
  auto numRepsK = numReps[kRepsIdx];
  auto numRepsB = numReps[bRepsIdx];

  auto memDesc = op->getOperand(0);
  auto memDescType = cast<ttg::MemDescType>(memDesc.getType());
  auto memDescEncoding =
      cast<triton::gpu::SwizzledSharedEncodingAttr>(memDescType.getEncoding());

  SmallVector<int64_t> refinedShape = {resultShape[0] / numReps2D[0],
                                       resultShape[1] / numReps2D[1]};
  LDBG("refinedShape: " << refinedShape[0] << "x" << refinedShape[1]);
  int64_t refinedShapeNonK = refinedShape[nonKDimIdx];
  int64_t refinedShapeK = refinedShape[kDimIdx];

  auto refinedTensorType =
      RankedTensorType::get(refinedShape, resultElementType, resultEncode);

  constexpr bool mutableMemory = true;
  auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto subviewType = ttg::MemDescType::get(
      refinedShape, memDescType.getElementType(), memDescType.getEncoding(),
      sharedMemorySpace, mutableMemory, memDescType.getAllocShape());

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> subtiles;
  auto order = memDescEncoding.getOrder();
  for (int32_t j = 0; j < numReps2D[1]; ++j) {   // dim[1]
    for (int32_t i = 0; i < numReps2D[0]; ++i) { // dim[0]
      int32_t offset0 = i * refinedShape[0];
      int32_t offset1 = j * refinedShape[1];
      auto offset = createOffset({}, {offset0, offset1}, rewriter, loc);
      auto refinedView = rewriter.create<ttg::MemDescSubviewOp>(
          loc, subviewType, memDesc, offset);
      LDBG("RefinedLocalLoadSubvew: " << *refinedView);

      auto refinedLoad = rewriter.create<ttg::LocalLoadOp>(
          loc, refinedTensorType, refinedView);
      // LDBG("RefinedLocalLoad: " << *refinedLoad);
      subtiles.push_back(refinedLoad);
    }
  }

  // concat dims is correct shape 8x1 vs 1x8, else gives wrong output shape.
  auto concatDims = DenseI64ArrayAttr::get(ctx, numReps2D);
  auto joinedResult = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, resultType, subtiles, concatDims);
  LDBG("ConcatOp: " << *joinedResult);

  op.replaceAllUsesWith(joinedResult.getResult());
  return success();
}

struct DotOpMFMAConverter {
  AMDMfmaEncodingAttr mfmaLayout;
  OpBuilder &rewriter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConverter(AMDMfmaEncodingAttr mfmaLayout,
                              OpBuilder &rewriter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter), loc(loc),
        ctx(mfmaLayout.getContext()) {}

  LogicalResult convert(DotOp dotOp, DotOpAdaptor adaptor) const {
    InputPrecisionAttr precisionAttr = dotOp.getInputPrecisionAttr();
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    Value c = dotOp.getC();
    Value d = dotOp.getD();

    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto cTensorTy = cast<RankedTensorType>(c.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());

    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    auto elemTyC = cTensorTy.getElementType();
    auto elemTyD = dTensorTy.getElementType();

    auto encodeA = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto encodeB = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    auto encodeC = cast<AMDMfmaEncodingAttr>(cTensorTy.getEncoding());
    auto encodeD = cast<AMDMfmaEncodingAttr>(dTensorTy.getEncoding());

    auto shapeA = aTensorTy.getShape();
    auto shapeB = bTensorTy.getShape();
    auto shapeC = cTensorTy.getShape();
    auto shapeD = dTensorTy.getShape();

    const auto kDimOperandSize = aTensorTy.getShape().back();

    int kWidth = encodeA.getKWidth();
    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);
    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    const auto numRepM = repA[1];
    const auto numRepN = repB[2];

    // TODO(dtanner) This is a temporary workaround so that local_load and dot
    // are decomposed the same and the intervening extract_slice and concat can
    // be canonicalized away. Re-enable slicing dots along K when we know we can
    // slice local_load along K too.
    const auto numRepK = repA[2];
    // const int numRepK = 1;
    const auto numRepB = repA[0];
    SmallVector<int64_t> numRepShape = {numRepM, numRepN, numRepK};
    LDBG("totalReps: " << numRepShape[0] << "x" << numRepShape[1] << "x"
                       << numRepShape[2]);
    SmallVector<int64_t> refinedShapeA = {shapeA[0] / numRepM,
                                          shapeA[1] / numRepK};
    SmallVector<int64_t> refinedShapeB = {shapeB[0] / numRepK,
                                          shapeB[1] / numRepN};
    SmallVector<int64_t> refinedShapeCD = {shapeC[0] / numRepM,
                                           shapeC[1] / numRepN};

    // Calculate mfmas per rep.
    SmallVector<int64_t> ctaTile = {shapeC[0], shapeC[1], shapeA[1]};
    SmallVector<int64_t> warpTile = {
        shapeC[0] / warpsPerCTA[0],
        shapeC[1] / warpsPerCTA[1],
        shapeA[1],
    };
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    bool allowXF32 =
        dotOp.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;

    FailureOr<MfmaIntrinsic> maybeMfmaInsn = MfmaIntrinsic::selectFor(
        mfmaVersion, mDim, nDim, kDimOperandSize, elemTyA, elemTyB,
        /*withScale=*/false, allowXF32);

    SmallVector<unsigned> mfmaShape = {16, 16, 16};
    if (failed(maybeMfmaInsn)) {
      llvm::errs() << "No match found in MFMA database\n";
    } else {
      mfmaShape[0] = maybeMfmaInsn->mDim;
      mfmaShape[1] = maybeMfmaInsn->nDim;
      mfmaShape[2] = maybeMfmaInsn->kDim;
    }

    auto mfmasPerRep =
        getMfmasPerRep(ctaTile, warpsPerCTA, numRepShape, mfmaShape);

    // Calculate Dot-Tiling.
    unsigned cyclesPerMfma = getCyclesPerMfma(dotOp);
    // Prefer tile to be skinny along inner loop dimension to minimize
    // registers.
    const bool preferOuterLoopM =
        (warpTile[0] >= warpTile[1]); // true: row-major when tall warp-tile
    const bool preferTileLargerM =
        !preferOuterLoopM; // true: tall tiles when wide warp-tile
    // Calculate dot-tile shape (in reps per dot-tile).
    DotTileShapeType tileShape =
        calcDotTileShape(mfmasPerRep, preferTileLargerM, cyclesPerMfma);

    tileShape[0] = std::min(tileShape[0], static_cast<unsigned>(numRepM));
    tileShape[1] = std::min(tileShape[1], static_cast<unsigned>(numRepN));
    tileShape[2] = std::min(tileShape[2], static_cast<unsigned>(numRepK));

    LDBG("repsPerDotTile: " << tileShape[0] << "x" << tileShape[1] << "x"
                            << tileShape[2]);
    const int tileShapeM = tileShape[0];
    const int tileShapeN = tileShape[1];
    const int tileShapeK = tileShape[2];
    const DotTileOrder dotTileOrder(numRepM, numRepN, tileShapeM, tileShapeN,
                                    preferOuterLoopM);

    // Extract slices for A operands.
    int64_t elementsPerSliceM = refinedShapeCD[0];
    int64_t elementsPerSliceN = refinedShapeCD[1];
    int64_t elementsPerSliceK = refinedShapeA[1];
    auto extractSliceTypeA =
        RankedTensorType::get(refinedShapeA, elemTyA, encodeA);
    rewriter.setInsertionPointAfter(dotOp);
    SmallVector<SmallVector<amdgpu::ExtractSliceOp>> subtilesA;
    unsigned tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<amdgpu::ExtractSliceOp> subtilesK;
      for (int32_t i = 0; i < numRepM; ++i) {
        LDBG("local_load_a[" << i << "][" << k << "] extract_slice");
        int32_t shiftM = i * elementsPerSliceM;
        int32_t shiftK = k * elementsPerSliceK;
        auto extract = rewriter.create<amdgpu::ExtractSliceOp>(
            loc, Type{extractSliceTypeA}, Value{a},
            DenseI64ArrayAttr::get(ctx, {shiftM, shiftK}));
        // Add dot-tile info to local_load's slice;
        // this specifies which dot-tile this load is needed for.
        int32_t tileM = i / tileShapeM;
        int32_t tileN = -1;
        int32_t tileK = k / tileShapeK;
        int32_t tileSerial = dotTileOrder.getOuterTileM()
                                 ? tileM * dotTileOrder.getNumTilesN()
                                 : tileM;
        tileSerial +=
            k * dotTileOrder.getNumTilesM() * dotTileOrder.getNumTilesN();
        int32_t elementM = i % tileShapeM; // dots are n-major within tile
        int32_t elementN = -1;
        int32_t elementK = k % tileShapeK;
        int32_t elementSerial =
            elementM * tileShapeN; // dots are n-major within tile
        auto dotTileAttr = triton::amdgpu::DotTileAttr::get(
            ctx, tileM, tileN, tileK, tileSerial, elementM, elementN, elementK,
            elementSerial);
        extract->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(),
                         dotTileAttr);
        subtilesK.push_back(extract);
      }
      subtilesA.push_back(subtilesK);
    }

    // Extract slices for B operands.
    auto extractSliceTypeB =
        RankedTensorType::get(refinedShapeB, elemTyB, encodeB);
    SmallVector<SmallVector<amdgpu::ExtractSliceOp>> subtilesB;
    tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<amdgpu::ExtractSliceOp> subtilesK;
      for (int32_t j = 0; j < numRepN; ++j) {
        LDBG("local_load_b[" << k << "][" << j << "] extact_slice");
        int32_t shiftN = j * elementsPerSliceN;
        int32_t shiftK = k * elementsPerSliceK;
        auto extract = rewriter.create<amdgpu::ExtractSliceOp>(
            loc, Type{extractSliceTypeB}, Value{b},
            DenseI64ArrayAttr::get(ctx, {shiftK, shiftN}));
        // Add dot-tile info to local_load's slice;
        // this specifies which dot-tile this load is needed for.
        int32_t tileM = -1;
        int32_t tileN = j / tileShapeN;
        int32_t tileK = k / tileShapeK;
        int32_t tileSerial = dotTileOrder.getOuterTileM()
                                 ? tileN
                                 : tileN * dotTileOrder.getNumTilesM();
        tileSerial +=
            k * dotTileOrder.getNumTilesM() * dotTileOrder.getNumTilesN();
        int32_t elementM = -1;
        int32_t elementN = j % tileShapeN; // dots are n-major within tile
        int32_t elementK = k % tileShapeK;
        int32_t elementSerial = elementN; // dots are n-major within tile
        auto dotTileAttr = triton::amdgpu::DotTileAttr::get(
            ctx, tileM, tileN, tileK, tileSerial, elementM, elementN, elementK,
            elementSerial);
        extract->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(),
                         dotTileAttr);
        subtilesK.push_back(extract);
      }
      subtilesB.push_back(subtilesK);
    }

    auto refinedTensorTypeC =
        RankedTensorType::get(refinedShapeCD, elemTyC, encodeC);
    auto refinedTensorTypeD =
        RankedTensorType::get(refinedShapeCD, elemTyD, encodeD);
    SmallVector<Value> refinedDotValues;
    // Extract slices for refined C operands for first slice of K.
    // Create these in same order that concat wants them.
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        SmallVector<int64_t> offset = {m * elementsPerSliceM,
                                       n * elementsPerSliceN};
        auto refinedTensorC = rewriter.create<triton::amdgpu::ExtractSliceOp>(
            loc, Type{refinedTensorTypeC}, Value{c}, offset);
        refinedDotValues.push_back(refinedTensorC);
      }
    }
    auto dotAttrs = dotOp->getAttrs();
    int32_t tileSerial = 0;
    // Iterate over dot-tiles.
    for (int32_t tileIdxK = 0; tileIdxK < numRepK / tileShapeK; ++tileIdxK) {
      for (int tileOuterIdx = 0; tileOuterIdx < dotTileOrder.getNumTilesOuter();
           ++tileOuterIdx) {
        for (int tileInnerIdx = 0;
             tileInnerIdx < dotTileOrder.getNumTilesInner(); ++tileInnerIdx) {
          const int tileStartM =
              dotTileOrder.getTileStartM(tileOuterIdx, tileInnerIdx);
          const int tileStartN =
              dotTileOrder.getTileStartN(tileOuterIdx, tileInnerIdx);
          for (int k = tileIdxK * tileShapeK; k < (tileIdxK + 1) * tileShapeK;
               ++k) {
            int32_t elementSerial = 0;
            LDBG("dot-tile[" << tileSerial << "]");
            // Iterate over dots within dot-tile.
            for (int m = tileStartM; m < tileStartM + tileShapeM; ++m) {
              for (int n = tileStartN; n < tileStartN + tileShapeN; ++n) {
                LDBG("  dot[" << m << "][" << n << "][" << k << "]");
                auto refinedTensorA = subtilesA[k][m];
                auto refinedTensorB = subtilesB[k][n];
                auto dotOp = rewriter.create<tt::DotOp>(
                    loc, refinedTensorTypeD,
                    ValueRange{refinedTensorA, refinedTensorB,
                               refinedDotValues[int32_t(m * numRepN + n)]},
                    dotAttrs);
                // Add dot-tile info to dot.
                int32_t tileM = tileStartM / tileShapeM;
                int32_t tileN = tileStartN / tileShapeN;
                int32_t tileK = k;
                int32_t elementM = m - tileStartM;
                int32_t elementN = n - tileStartN;
                int32_t elementK = 0;
                auto dotTileAttr = triton::amdgpu::DotTileAttr::get(
                    ctx, tileM, tileN, tileK, tileSerial, elementM, elementN,
                    elementK, elementSerial);
                dotOp->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(),
                               dotTileAttr);
                refinedDotValues[int32_t(m * numRepN + n)] = dotOp;
                elementSerial++;
              }
            }
          }
          tileSerial++;
        }
      }
    }

    auto concatDims = DenseI64ArrayAttr::get(ctx, {numRepM, numRepN});
    auto joinedDotsResult = rewriter.create<triton::amdgpu::ConcatOp>(
        loc, dTensorTy, refinedDotValues, concatDims);

    d.replaceAllUsesWith(joinedDotsResult);

    // Note: dangling localLoadA or/and localLoadB (if exist)
    // should be removed by the dead code elimination pass
    dotOp.erase();
    return success();
  }
};

inline RankedTensorType rankedTType(Value tensor) {
  return cast<RankedTensorType>(tensor.getType());
};

LogicalResult rewriteMFMA(OpBuilder &rewriter, triton::DotOp op) {
  if (!(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
        isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()))) {
    LDBG("Both $a and %b should be DotOperand layout");
    return failure();
  }

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  if (!isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding())) {
    LDBG("Currently, we only support $c with a mfma layout");
    return failure();
  }

  if (!(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
        cTensorTy.getShape()[1] == dTensorTy.getShape()[1])) {
    LDBG("DotOp's $c operand should pass the same number of values as $d");
    return failure();
  }

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConverter converter(mfmaLayout, rewriter, loc);
  return converter.convert(op, DotOpAdaptor(op));
}

struct RefinedBlock {
  RefinedBlock(ArrayRef<int64_t> shape, Type elemType,
               BlockedEncodingAttr encoding)
      : encoding(encoding), elemType(elemType) {
    auto ctaOrder = encoding.getCTAOrder();
    auto warpsPerCTA = encoding.getWarpsPerCTA();
    auto threadsPerWarp = encoding.getThreadsPerWarp();
    auto sizePerThread = encoding.getSizePerThread();

    numDims = warpsPerCTA.size();
    elementsPerWorkGroup.resize(numDims);
    numPerDims.resize(numDims);
    refinedShape.resize(numDims);
    numSubTiles = 1;
    for (size_t dim = 0; dim < numDims; ++dim) {
      elementsPerWorkGroup[dim] =
          sizePerThread[dim] * threadsPerWarp[dim] * warpsPerCTA[dim];
      numPerDims[dim] = shape[dim] / elementsPerWorkGroup[dim];
      refinedShape[dim] = shape[dim] / numPerDims[dim];
      numSubTiles *= numPerDims[dim];
    }

    tensorType =
        RankedTensorType::get(elementsPerWorkGroup, elemType, encoding);
  }

  BlockedEncodingAttr encoding;
  Type elemType;
  SmallVector<int64_t> elementsPerWorkGroup;
  SmallVector<int64_t> numPerDims;
  SmallVector<int64_t> refinedShape;
  size_t numDims;
  size_t numSubTiles;
  RankedTensorType tensorType;
};

LogicalResult rewriteLoadOp(OpBuilder &rewriter, triton::LoadOp loadOp) {
  auto ctx = loadOp->getContext();
  auto loc = loadOp.getLoc();

  Value origSrc = loadOp->getOperand(0);
  Value origResult = loadOp.getResult();
  Type origResultType = loadOp.getResult().getType();
  auto origPtrs = rankedTType(origSrc);
  auto origShape = origPtrs.getShape();
  auto elemType = origPtrs.getElementType();
  auto encoding = dyn_cast<BlockedEncodingAttr>(origPtrs.getEncoding());
  if (encoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origShape, elemType, encoding);

  rewriter.setInsertionPointAfter(loadOp);
  SmallVector<Value> refinedTensors;

  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  auto boundaryCheck = loadOp.getBoundaryCheck();
  auto padding = loadOp.getPadding();
  auto cache = loadOp.getCache();
  auto evict = loadOp.getEvict();
  auto isVolatile = loadOp.getIsVolatile();

  CoordinateAux aux(refinedBlock.numPerDims);
  for (size_t counter = 0; counter < refinedBlock.numSubTiles; ++counter) {
    auto coords = aux.map(counter);
    SmallVector<int64_t> offset(refinedBlock.numDims, 0);
    for (auto [dim, coord] : llvm::enumerate(coords)) {
      offset[dim] = coord * refinedBlock.elementsPerWorkGroup[dim];
    }

    auto slice = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedBlock.tensorType}, Value{origSrc}, offset);

    auto refinedTensor =
        rewriter.create<triton::LoadOp>(loc, slice, mask, other, boundaryCheck,
                                        padding, cache, evict, isVolatile);
    refinedTensors.push_back(refinedTensor);
  }

  auto concatDims = DenseI64ArrayAttr::get(ctx, refinedBlock.numPerDims);
  auto joinedResult = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, origResultType, refinedTensors, concatDims);

  origResult.replaceAllUsesWith(joinedResult);
  return success();
}

LogicalResult rewriteLocalStoreOp(OpBuilder &rewriter,
                                  triton::gpu::LocalStoreOp loadStoreOp) {
  auto ctx = loadStoreOp->getContext();
  auto loc = loadStoreOp.getLoc();

  Value origSrc = loadStoreOp->getOperand(0);
  auto origMemViewOp =
      cast<ttg::MemDescSubviewOp>(loadStoreOp->getOperand(1).getDefiningOp());
  Value origMemView = origMemViewOp->getOperand(0);
  Value selectValue = origMemViewOp.getOffsets().front();

  auto origSrcType = rankedTType(origSrc);
  auto blockEncoding = dyn_cast<BlockedEncodingAttr>(origSrcType.getEncoding());
  if (blockEncoding == nullptr)
    return failure();

  auto origMemViewType = cast<ttg::MemDescType>(origMemView.getType());
  auto sharedEncoding = cast<triton::gpu::SwizzledSharedEncodingAttr>(
      origMemViewType.getEncoding());
  if (sharedEncoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origSrcType.getShape(),
                            origSrcType.getElementType(), blockEncoding);

  constexpr bool mutableMemory = true;
  auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);

  auto subviewType = ttg::MemDescType::get(
      refinedBlock.refinedShape, refinedBlock.elemType, sharedEncoding,
      sharedMemorySpace, mutableMemory, origMemViewType.getAllocShape());

  rewriter.setInsertionPointAfter(loadStoreOp);
  CoordinateAux aux(refinedBlock.numPerDims);
  for (size_t counter = 0; counter < refinedBlock.numSubTiles; ++counter) {
    auto coords = aux.map(counter);
    SmallVector<int64_t> offset(refinedBlock.numDims, 0);
    for (auto [dim, coord] : llvm::enumerate(coords)) {
      offset[dim] = coord * refinedBlock.elementsPerWorkGroup[dim];
    }
    auto offsetValues = createOffset({selectValue}, offset, rewriter, loc);
    auto slicedSharedMemView = rewriter.create<ttg::MemDescSubviewOp>(
        loc, subviewType, origMemView, offsetValues);

    auto slice = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedBlock.tensorType}, Value{origSrc}, offset);

    rewriter.create<ttg::LocalStoreOp>(loc, slice, slicedSharedMemView);
  }

  loadStoreOp.erase();
  return success();
}

// Reduce ops have different intput and output shapes and produce
// sliced layouts.
// This currently only supports 2d inputs.
LogicalResult rewriteReduceOp(OpBuilder &rewriter, triton::ReduceOp op) {
  auto ctx = op->getContext();
  auto loc = op.getLoc();
  uint32_t axisReduce = op.getAxis();
  uint32_t axisNonReduce = (axisReduce + 1) % 2;
  if (op.getNumOperands() != 1)
    return failure();

  // Calculate refined shape.
  auto src = op->getOperand(0);
  auto srcType = rankedTType(src);
  if (srcType.getRank() != 2)
    return failure();
  auto srcShape = srcType.getShape();
  auto srcEncoding = srcType.getEncoding();
  auto srcShapePerCtaTile = triton::gpu::getShapePerCTATile(srcType);
  SmallVector<int64_t> repShape = {srcShape[0] / srcShapePerCtaTile[0],
                                   srcShape[1] / srcShapePerCtaTile[1]};
  int numReps = repShape[axisNonReduce];
  SmallVector<int64_t> refinedSrcShape = {srcShape[0], srcShape[1]};
  refinedSrcShape[axisNonReduce] /= numReps;
  int64_t elementsPerRep = refinedSrcShape[axisNonReduce];
  auto elemTy = srcType.getElementType();
  auto refinedTensorType =
      RankedTensorType::get(refinedSrcShape, elemTy, srcEncoding);

  // Create refined ops.
  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> refinedReduces;
  for (int i = 0; i < numReps; ++i) {
    SmallVector<int64_t> offset(refinedSrcShape.size(), 0);
    offset[axisReduce] = 0;
    offset[axisNonReduce] = i * elementsPerRep;
    auto sliceOp = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedTensorType}, Value{src}, offset);
    auto reduceOp =
        rewriter.create<triton::ReduceOp>(loc, ValueRange{sliceOp}, axisReduce);
    IRMapping mapping;
    mapping.map(reduceOp.getOperand(0), sliceOp);
    op.getCombineOp().cloneInto(&reduceOp->getRegion(0), mapping);
    refinedReduces.push_back(reduceOp->getResult(0));
  }

  // Concat reduce slices.
  auto reduceResultType = op.getResultTypes()[0];
  SmallVector<int64_t> concatDimShape = {numReps};
  auto concatDims = DenseI64ArrayAttr::get(ctx, concatDimShape);
  auto concatOp = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, reduceResultType, refinedReduces, concatDims);
  auto origOpResult = op.getResult();
  origOpResult.replaceAllUsesWith(concatOp);
  op.erase();
  return success();
}

// Refine ops with distributed layouts.
// Assumes same layout for operands.
template <typename OpTy>
LogicalResult rewriteElementWiseOp(OpBuilder &rewriter, OpTy op) {
  // Verify opd[0] is valid.
  int numOperands = op->getNumOperands();
  if (op->getNumOperands() < 1)
    return failure();
  auto src = op->getOperand(0);
  if (!isa<mlir::RankedTensorType>(src.getType()))
    return failure();
  auto srcType = rankedTType(src);
  auto rank = srcType.getRank();
  if (rank != 2) { // TODO(dtanner) remove me
    return failure();
  }
  auto srcShape = srcType.getShape();
  auto srcEncoding = srcType.getEncoding();
  auto srcShapePerCtaTile = getRefinedShapeElementWise(srcType);

  // Verify subsequent operands match opd[0].
  for (int i = 1; i < numOperands; ++i) {
    if (!isa<mlir::RankedTensorType>(op->getOperand(i).getType()))
      return failure();
    if (rankedTType(op->getOperand(i)).getRank()!=rank)
      return failure();
    if (getRefinedShapeElementWise(rankedTType(op->getOperand(i)))
      != srcShapePerCtaTile)
      return failure();
  }

  // Result tensor.
  auto res = op->getResult(0);
  if (!isa<mlir::RankedTensorType>(res.getType()))
    return failure();
  auto resType = rankedTType(res);
  auto resShape = resType.getShape();
  if (resShape != srcShape)
    return failure();

  // TODO(dtanner) remove this restriction.
  // this was placed here just to focus on a single certain op.
  if (srcShape[1] == 128)
    return failure();

  LDBG("rewriteElementWiseOp(): " << op);
  PRINT_SMALL_VECTOR(srcShape)
  PRINT_SMALL_VECTOR(srcShapePerCtaTile)


  // DEBUG check if concat op results in correct linear layout
  auto leRes = ttg::toLinearEncoding(resType);
  auto llRes = leRes.getLinearLayout();
  llvm::dbgs() << "original op result LL: " << llRes.toString() << "\n";

  auto resEncoding = resType.getEncoding();
  auto resShapePerCtaTile = getRefinedShapeElementWise(resType);

/*
    TODO(dtanner) remove this restruction.
    - Even with this restriction in place, the old concat errors.

    - With this restruction in place, the fix causes mfma16 tests to pass.
flash-attention.py ................................................................................................ [ 16%]
................................................................................................................... [ 36%]
..................................................

    - Commenting out these restrictions causes the fix to still fail tests.
flash-attention.py FFFF.F...FFFFFF.FFFF.F...FFFFFF.................................FFFF.F...FFFFFF.FFFF.F...FFFFFF. [ 16%]
................................FFFF.F...FFFFFF.FFFF.F...FFFFFF.................................FFFF.F...FFFFFF.FFF [ 36%]
F.F...FFFFFF.................................FFFF.F...FFFFFF.FFFF.F...FFFFFF.................................FFFF.F [ 55%]
...FFFFFF.FFFF.F...FFFFFF.................................F

*/
  srcShapePerCtaTile[0] = srcShape[0];
  resShapePerCtaTile[0] = resShape[0];

  // Calculate refined shapes.
  SmallVector<int64_t> refinedShape;
  SmallVector<int64_t> numReps;
  for (int i = 0; i < rank; ++i) {
    // src and res can have different refineable shapes if different layouts.
    refinedShape.push_back(std::max(srcShapePerCtaTile[i], resShapePerCtaTile[i]));
    numReps.push_back(srcShape[i] / srcShapePerCtaTile[i]);
  }
  PRINT_SMALL_VECTOR(numReps)
  if (product<int64_t>(numReps) == 1)
    return success();

  // Create refined ops.
  auto refinedTensorTypeSrc = RankedTensorType::get(
    refinedShape, srcType.getElementType(), srcEncoding);
  auto refinedTensorTypeRes = RankedTensorType::get(
    refinedShape, resType.getElementType(), resEncoding);


  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> refinedOps;
  SmallVector<int64_t> offset(rank, 0);
  int outerIdx = 0; // rank-1;
  int innerIdx = 1; // rank-2;
  for (int i = 0; i < numReps[outerIdx]; ++i) {
    offset[outerIdx] = i * refinedShape[outerIdx];

#define SLICE_OP \
SmallVector<Value> slicedOperands; \
    for (int opdIdx = 0; opdIdx < numOperands; ++opdIdx) { \
      auto slicedOperand = rewriter.create<triton::amdgpu::ExtractSliceOp>( \
        op.getLoc(), Type{refinedTensorTypeSrc}, \
        Value{op->getOperand(opdIdx)}, offset); \
      slicedOperands.push_back(slicedOperand); \
    } \
    auto refinedOp = rewriter.create<OpTy>( \
      op.getLoc(), refinedTensorTypeRes, slicedOperands); \
    refinedOps.push_back(refinedOp->getResult(0)); \

    if (rank == 1) {
      SLICE_OP
    } else if (rank == 2) {
      for (int j = 0; j < numReps[innerIdx]; ++j) {
        offset[innerIdx] = j * refinedShape[innerIdx];
        PRINT_SMALL_VECTOR(offset);
        SLICE_OP
      }
    } else {
      LDBG("rewriteElementWiseOp() doesn't yet support rank>2");
      return failure();
    }
#undef SLICE_OP
  }

  // Concat slices.
  auto resultType = op->getResultTypes()[0];
  auto concatDims = DenseI64ArrayAttr::get(op->getContext(), numReps);
  auto concatOp = rewriter.create<triton::amdgpu::ConcatOp>(
      op.getLoc(), resultType, refinedOps, concatDims);

  // DEBUG check if concat op results in correct linear layout
  auto leConcat = ttg::toLinearEncoding(rankedTType(concatOp->getResult(0)));
  auto llConcat = leConcat.getLinearLayout();
  llvm::dbgs() << "concat op result LL: " << llConcat.toString() << "\n";

  auto origOpResult = op.getResult();
  origOpResult.replaceAllUsesWith(concatOp);
  LDBG("rewriteElementWiseOp() - SUCCESS " << op);
  op.erase();
  return success();
}

// Refine ExpandDims ops.
// Since expanding dims increases tensor rank,
// this refinement multipe intermediate shapes,
//    ExSl     ExpD        Conct
// <M> -> <M/m> -> <M/m x 1> -> <Mx1>.
// TODO(dtanner) only need to support 1D sliceLayout input, same as ViewOpToLLVM.cpp ?
LogicalResult rewriteExpandDimsOp(OpBuilder &rewriter, triton::ExpandDimsOp op) {
  llvm::dbgs() << "rewriteExpandDimsOp: " << op << "\n";
  int numOperands = op->getNumOperands();
  if (op->getNumOperands() != 1)
    return failure();
  auto src = op->getOperand(0);
  if (!isa<mlir::RankedTensorType>(src.getType()))
    return failure();
  auto srcType = rankedTType(src);
  if (srcType.getElementTypeBitWidth() == 1)
    return failure();
  auto rank = srcType.getRank();
  auto srcShape = srcType.getShape();
  auto srcEncoding = srcType.getEncoding();
  auto srcShapePerCtaTile = getRefinedShapeExpandDims(srcType);
  llvm::dbgs() << "\nrewriteExpandDimsOp(): " << op << "\n";
  // Calculate refined shape.
  SmallVector<int64_t> refinedSrcShape;
  SmallVector<int64_t> numReps;
  for (int i = 0; i < rank; ++i) {
    refinedSrcShape.push_back(srcShapePerCtaTile[i]);
    numReps.push_back(srcShape[i] / srcShapePerCtaTile[i]);
  }
  if (product<int64_t>(numReps) == 1)
    return success();
  auto refinedResultShape = refinedSrcShape;
  refinedResultShape.insert(refinedResultShape.begin() + op.getAxis(), 1);
  auto refinedSrcTensorType = RankedTensorType::get(
    refinedSrcShape, srcType.getElementType(), srcEncoding);
  PRINT_SMALL_VECTOR(refinedSrcShape)
  PRINT_SMALL_VECTOR(refinedResultShape)
  PRINT_SMALL_VECTOR(numReps)
  llvm::dbgs() << "\nrewriteExpandDimsOp() starting loops\n";

  // Create refined ops.
  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> refinedReduces;
  SmallVector<int64_t> offset(rank, 0);
  for (int i = 0; i < numReps[rank-1]; ++i) {
    offset[rank-1] = i * refinedSrcShape[rank-1];

#define SLICE_OP \
    auto slicedOp = rewriter.create<triton::amdgpu::ExtractSliceOp>( \
      op.getLoc(), Type{refinedSrcTensorType}, \
      Value{op->getOperand(0)}, offset); \
    auto sliceRes = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>( \
      slicedOp->getResult(0)); \
    auto sliceResTy = sliceRes.getType(); \
    Attribute refinedResultEncoding; \
    if (auto refinedSrcEncoding = sliceResTy.getEncoding()) { \
      PRINT_SMALL_VECTOR(sliceResTy.getShape()) \
      if (cast<DialectInferLayoutInterface>(&srcEncoding.getDialect()) \
            ->inferExpandDimsOpEncoding(refinedSrcEncoding, \
            op.getAxis(), refinedResultEncoding, \
            op.getLoc()).failed()) { \
        return emitOptionalError(op.getLoc(), \
                                  "Failed to infer layout for ExpandDimsOp"); \
      } \
    } \
    auto sliceResTensorType = RankedTensorType::get( \
      refinedResultShape, sliceResTy.getElementType(), \
      refinedResultEncoding); \
    auto refinedOp = rewriter.create<triton::ExpandDimsOp>( \
      op.getLoc(), sliceResTensorType, sliceRes, op.getAxis()); \
    refinedReduces.push_back(refinedOp->getResult(0));

    // TODO(dtanner) how to iterate over Nd array?
    if (rank == 1) {
      SLICE_OP
    } else if (rank == 2) {
      for (int j = 0; j < numReps[rank-2]; ++j) {
        offset[rank-2] = j * refinedSrcShape[rank-2];
        SLICE_OP
      }
    } else {
      LDBG("rewriteExpandDimsOp() doesn't yet support rank>2");
      return failure();
    }
#undef SLICE_OP
  }

  // Concat refined ops.
  auto reduceResultType = op->getResultTypes()[0];
  // Expand dims of numReps also before concat.
  numReps.insert(numReps.begin() + op.getAxis(), 1);
  auto concatDims = DenseI64ArrayAttr::get(op->getContext(), numReps);
  auto concatOp = rewriter.create<triton::amdgpu::ConcatOp>(
      op.getLoc(), reduceResultType, refinedReduces, concatDims);
  auto origOpResult = op.getResult();
  origOpResult.replaceAllUsesWith(concatOp);
  op.erase();
  llvm::dbgs() << "\nrewriteExpandDimsOp() - SUCCESS: " << concatOp << "\n";

  return success();
}


// Refine Broadcast ops.
// Since inputs are roughtly 1D and outputs are roughly 2D,
// Then the op and outputs are sliced more than the inputs.
// In the below example, shapePerCtaTile is 64x32,
// so the input can be cut in half, while the BroadcastOp
// can be cut into fourths, and the Concat will have dims=2x2.
// Presumably this means the 1st and 3rd Broadcasts are redundant,
// the 2nd and 4th Broadcasts are redundant, and some will be
// eliminated by CSE in the backend compiler.
// Example:
//       ExSl      Brdcst       Concat
//<128x1> ->  <64x1> -> <64x32>    -> 128x64
//        \             <64x32>   /
//         -> <64x1> -> <64x32>  /
//                      <64x32> /
LogicalResult rewriteBroadcastOp(OpBuilder &rewriter, BroadcastOp op) {

  // src tensor e.g. <128x1>.
  llvm::dbgs() << "rewriteBroadcastOp(): " << op << "\n";
  int numOperands = op->getNumOperands();
  if (op->getNumOperands() != 1)
    return failure();
  auto src = op->getOperand(0);
  if (!isa<mlir::RankedTensorType>(src.getType()))
    return failure();
  auto srcType = rankedTType(src);
  auto rank = srcType.getRank();
  if (rank != 2)
    return failure();
  if (srcType.getElementTypeBitWidth() == 1)
    return failure();
  auto srcShape = srcType.getShape();
  auto srcEncoding = srcType.getEncoding();
  auto srcShapePerCtaTile = getRefinedShapeBroadcast(srcType);
  PRINT_SMALL_VECTOR(srcShapePerCtaTile)

  // Result tensor e.g. <128x64>.
  auto res = op->getResult(0);
  if (!isa<mlir::RankedTensorType>(res.getType()))
    return failure();
  auto resType = rankedTType(res);
  auto resShape = resType.getShape();
  auto resEncoding = resType.getEncoding();
  auto resShapePerCtaTile = getRefinedShapeBroadcast(resType);
  PRINT_SMALL_VECTOR(resShapePerCtaTile)

  // numReps
  SmallVector<int64_t> refinedSrcShape;
  SmallVector<int64_t> refinedResShape;
  SmallVector<int64_t> numReps;
  for (int i = 0; i < rank; ++i) {
    refinedSrcShape.push_back(srcShapePerCtaTile[i]);
    refinedResShape.push_back(resShapePerCtaTile[i]);
    numReps.push_back(resShape[i] / resShapePerCtaTile[i]);
  }
  if (product<int64_t>(numReps) == 1)
    return success();
  PRINT_SMALL_VECTOR(numReps)

  // Determine indices and values of reps.
  // numRepsSrc is the non-one size, because the src can be sliced.
  // numRepsRes is the one size, because the result will be repeated.
  unsigned numRepsSrcIdx = 0; // <*128x 1>
  unsigned numRepsResIdx = 1; // < 128x*1>
  if (refinedSrcShape[numRepsSrcIdx] == 1) {
    numRepsSrcIdx = 1; // < 1x*64>
    numRepsResIdx = 0; // <*1x 64>
  }
  unsigned numRepsSrc = numReps[numRepsSrcIdx];
  unsigned numRepsRes = numReps[numRepsResIdx];

  // Refined src/result tensor types.
  auto refinedSrcTensorType = RankedTensorType::get(
    refinedSrcShape, srcType.getElementType(), srcEncoding);
  auto refinedResTensorType = RankedTensorType::get(
    refinedResShape, srcType.getElementType(), srcEncoding);

  // Create refined ops.
  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> refinedBroadcasts;
  SmallVector<int64_t> offset(rank, 0);
  for (int i = 0; i < numRepsSrc; ++i) {
    offset[numRepsSrcIdx] = i * refinedSrcShape[numRepsSrcIdx];
    // Create slice.
    auto slicedOp = rewriter.create<triton::amdgpu::ExtractSliceOp>(
      op.getLoc(), Type{refinedSrcTensorType},
      Value{op->getOperand(0)}, offset);
    auto sliceRes = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(
      slicedOp->getResult(0));
    auto sliceResTensorType = RankedTensorType::get(
      refinedResShape, srcType.getElementType(),
      resEncoding);
    for (int j = 0; j < numRepsRes; ++j) {
      // Create broadcast.
      auto broadcastOp = rewriter.create<triton::BroadcastOp>(
        op.getLoc(), sliceResTensorType, sliceRes);
        refinedBroadcasts.push_back(broadcastOp->getResult(0));
    }
  }

  // Concat refined ops.
  auto reduceResultType = op->getResultTypes()[0];
  auto concatDims = DenseI64ArrayAttr::get(
    op->getContext(), numReps);
  auto concatOp = rewriter.create<triton::amdgpu::ConcatOp>(
      op.getLoc(), reduceResultType, refinedBroadcasts, concatDims);

  auto origOpResult = op.getResult();
  origOpResult.replaceAllUsesWith(concatOp);
  op.erase();

  return success();
}

struct TritonAMDGPURefineOps
    : public TritonAMDGPURefineOpsBase<TritonAMDGPURefineOps> {
  explicit TritonAMDGPURefineOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == mlir::triton::AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    mod->walk([&](amdgpu::InstructionSchedHint hint) {
      if (hint.getVariant() != amdgpu::SchedHint::refine_ops) {
        return WalkResult::advance();
      }
      llvm::dbgs() << "refining ops\n";
      auto *block = hint->getBlock();
#if 0
      block->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
        OpBuilder rewriter(localLoadOp->getContext());
        if (localLoadOp->getNumOperands() == 1) {
          if (failed(rewriteLocalLoad(rewriter, localLoadOp))) {
            LDBG("failed to refine ttg.localLoad: " << *localLoadOp);
          }
        }
      });

      block->walk([&](triton::DotOp dotOp) {
        OpBuilder rewriter(dotOp->getContext());
        // TODO: extend to WMMA instructions
        if (failed(rewriteMFMA(rewriter, dotOp))) {
          LDBG("failed to refine tt.dotOp: " << *dotOp);
        }
      });

      block->walk([&](triton::LoadOp loadOp) {
        OpBuilder rewriter(loadOp->getContext());
        if (loadOp->getNumOperands() == 1) {
          if (failed(rewriteLoadOp(rewriter, loadOp))) {
            LDBG("failed to refine tt.loadOp: " << *loadOp);
          }
        }
      });

      block->walk([&](triton::gpu::LocalStoreOp storeOp) {
        OpBuilder rewriter(storeOp->getContext());
        if (storeOp->getNumOperands() == 2) {
          if (failed(rewriteLocalStoreOp(rewriter, storeOp))) {
            LDBG("failed to refine ttg.localLoadOp: " << *storeOp);
          }
        }
      });

      block->walk([&](triton::ReduceOp reduceOp) {
        OpBuilder rewriter(reduceOp->getContext());
        if (failed(rewriteReduceOp(rewriter, reduceOp))) {
          LDBG("failed to refine tt.reduce: " << *reduceOp);
        }
      });
#endif
      // Refine Element-Wise Ops.
      #define REFINE_ELEMENTWISE_OP(OP_TYPE) \
      block->walk([&](OP_TYPE op) { \
        OpBuilder rewriter(op->getContext()); \
        if (failed(rewriteElementWiseOp<OP_TYPE>(rewriter, op))) { \
          LDBG("failed to refine binary op: " << *op); \
        } \
      });
#if 0
      // Refine Unary Element-Wise Ops.
      REFINE_ELEMENTWISE_OP(math::RsqrtOp)
      REFINE_ELEMENTWISE_OP(math::Exp2Op)
      REFINE_ELEMENTWISE_OP(arith::TruncFOp)
      REFINE_ELEMENTWISE_OP(arith::ExtFOp)
      REFINE_ELEMENTWISE_OP(arith::FPToSIOp)
      REFINE_ELEMENTWISE_OP(arith::SIToFPOp)
      REFINE_ELEMENTWISE_OP(triton::FpToFpOp)
      REFINE_ELEMENTWISE_OP(triton::PreciseSqrtOp)
      REFINE_ELEMENTWISE_OP(math::SqrtOp)
      REFINE_ELEMENTWISE_OP(math::ExpOp)
      // Refine Binary Element-Wise Ops.
      REFINE_ELEMENTWISE_OP(arith::SubIOp)  // -
      REFINE_ELEMENTWISE_OP(arith::AddIOp)  // +
      REFINE_ELEMENTWISE_OP(arith::MulIOp)  // *
      REFINE_ELEMENTWISE_OP(arith::DivSIOp)
      REFINE_ELEMENTWISE_OP(arith::DivUIOp)
      REFINE_ELEMENTWISE_OP(arith::RemFOp)  // %
      REFINE_ELEMENTWISE_OP(arith::RemSIOp)
      REFINE_ELEMENTWISE_OP(arith::RemUIOp)
      REFINE_ELEMENTWISE_OP(arith::AndIOp)  // &
      REFINE_ELEMENTWISE_OP(arith::OrIOp)   // |
      REFINE_ELEMENTWISE_OP(arith::XOrIOp)  // ^
      REFINE_ELEMENTWISE_OP(arith::ShLIOp)  // <<
      REFINE_ELEMENTWISE_OP(arith::ShRSIOp) // >>
      REFINE_ELEMENTWISE_OP(arith::ShRUIOp) // >>
      REFINE_ELEMENTWISE_OP(arith::MinNumFOp)
      REFINE_ELEMENTWISE_OP(arith::MaxNumFOp)
      REFINE_ELEMENTWISE_OP(arith::MinSIOp) // smin
      REFINE_ELEMENTWISE_OP(arith::MaxSIOp) // smax
      REFINE_ELEMENTWISE_OP(arith::MinUIOp) // umin
      REFINE_ELEMENTWISE_OP(arith::MaxUIOp) // umax
      // Floating-point Ops.
      REFINE_ELEMENTWISE_OP(arith::AddFOp)
      REFINE_ELEMENTWISE_OP(arith::SubFOp)
#endif
      REFINE_ELEMENTWISE_OP(arith::MulFOp)
#if 0
      REFINE_ELEMENTWISE_OP(arith::DivFOp)
      REFINE_ELEMENTWISE_OP(arith::MaximumFOp)
      REFINE_ELEMENTWISE_OP(arith::MinimumFOp)
#endif

#if 0
      // Other ops which fit the "elementwise" refinement pattern.
      REFINE_ELEMENTWISE_OP(triton::gpu::ConvertLayoutOp)
#endif

#if 0
      // Refine ExpandDimsOp: 128 -> 128x1
      block->walk([&](triton::ExpandDimsOp op) {
        OpBuilder rewriter(op->getContext());
        if (failed(rewriteExpandDimsOp(rewriter, op))) {
          LDBG("failed to refine tt.expand_dims: " << *op);
        }
      });
#endif

#if 0
      // Refine BroadcastOp: 128x1 -> 128x64
      block->walk([&](triton::BroadcastOp op) {
        OpBuilder rewriter(op->getContext());
        if (failed(rewriteBroadcastOp(rewriter, op))) {
          LDBG("failed to refine tt.broadcast: " << *op);
        }
      });
#endif


      return WalkResult::advance();
    });
  }

private:
};

} // namespace

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURefineOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURefineOps>(targetArch);
}

} // namespace mlir
