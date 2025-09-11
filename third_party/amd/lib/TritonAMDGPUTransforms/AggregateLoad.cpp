#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/MapVector.h"

using llvm::MapVector;
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
using triton::AMD::ISAFamily;

int64_t getAllocSize(ShapedType type) {
  int64_t numel = ShapedType::getNumElements(type.getShape());
  int64_t bitwidth = type.getElementType().getIntOrFloatBitWidth();
  // TODO: Remove this, only using this to simplify shared memory calculation
  //       of using 65536 "bytes". If we change it to bit that can handle
  //       practically anything.
  assert(bitwidth >= 8 && "Only support bitwidth > 1 byte");
  int64_t allocSize = numel * (bitwidth / 8);
  return allocSize;
}

Operation *getLoadOpFromScale(Value scale) {
  Operation *op = scale.getDefiningOp();
  while (!isa<triton::LoadOp>(op)) {
    if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      op = convert.getSrc().getDefiningOp();
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      op = reshape.getSrc().getDefiningOp();
    } else {
      return nullptr;
    }
  }
  return op;
}

void findValidLoads(scf::ForOp forOp,
                    SetVector<std::pair<Operation *, Operation *>> &validLoads,
                    SmallVector<std::pair<int64_t, int64_t>> &hoistLoopSpecs,
                    int ub, int aggregateFactor,
                    int64_t totalSharedMemoryUsage) {
  int64_t currentSharedMemoryUsage = totalSharedMemoryUsage;
  for (Operation &op : forOp) {
    if (auto dotScaledOp = dyn_cast<triton::DotScaledOp>(&op)) {
      Value aScale = dotScaledOp.getAScale();
      Value bScale = dotScaledOp.getBScale();
      Operation *aScaleLoadOp = getLoadOpFromScale(aScale);
      Operation *bScaleLoadOp = getLoadOpFromScale(bScale);
      if (!aScaleLoadOp || !bScaleLoadOp) {
        llvm::outs() << "Can't find loadOp of aScale or bScale;\n";
        continue;
      }

      auto aScaleTy = dyn_cast<RankedTensorType>(aScale.getType());
      auto bScaleTy = dyn_cast<RankedTensorType>(bScale.getType());
      assert(isa<ttg::LinearEncodingAttr>(aScaleTy.getEncoding()) &&
             isa<ttg::LinearEncodingAttr>(bScaleTy.getEncoding()) &&
             "Both aScale and bScale should be linear layout.");

      const int64_t kMaxSharedMemory = 163840;
      assert(currentSharedMemoryUsage <= kMaxSharedMemory &&
             "Even without hoisting, block size is too large.");

      int64_t scaleLDSUsage = getAllocSize(aScaleTy) + getAllocSize(bScaleTy);

      auto aScaleShape = aScaleTy.getShape();
      auto bScaleShape = bScaleTy.getShape();

      // llvm::outs() << "ub: " << ub << "\n";
      // llvm::outs() << "aScaleShape: (" << aScaleShape[0] << ", "
      //              << aScaleShape[1] << ")\n";
      // llvm::outs() << "bScaleShape: (" << bScaleShape[0] << ", "
      //              << bScaleShape[1] << ")\n";
      assert((aScaleShape[1] == bScaleShape[1]) &&
             "aScale and bScale should have the same K size.");

      int64_t expandableMemory = kMaxSharedMemory - currentSharedMemoryUsage;
      if (scaleLDSUsage >= expandableMemory) {
        llvm::outs() << "Already maxed out!\n";
        continue;
      }

      // Scale is always E8M0-encoded, so its width is always 1B.
      constexpr int byteWidth = 1;

      int newUpperBound = ub;
      int64_t hoistFactor = 1;
      if (aggregateFactor != -1) {
        newUpperBound = aggregateFactor;
        hoistFactor = ub / aggregateFactor;
      }
      auto getAlignedScaleLDSUsage = [&](int numBlocks) {
        return (aScaleShape[0] + bScaleShape[0]) *
               llvm::PowerOf2Ceil(aScaleShape[1] * numBlocks) * byteWidth;
      };

      // The following binary search may fail if the overall K dimension doesn't
      // have enough factors of 2. But it should be fine in common cases. May
      // consider better algorithm if needed.
      //
      // Due to Linear Layout's requirement, the aggregated blocks need to be
      // power of 2. This is done by padding LDS, which may introduce LDS
      // overhead. Considering scales are relatively small and LDS hasn't become
      // the bottleneck on gfx950, this should be fine.
      int64_t newMemoryUsed = 0;
      bool foundHoistKDim = false;
      while (newUpperBound > 0 && newUpperBound % 2 == 0) {
        newMemoryUsed = getAlignedScaleLDSUsage(newUpperBound);
        if (newMemoryUsed < expandableMemory) {
          hoistFactor = ub / newUpperBound;
          foundHoistKDim = true;
          break;
        }
        newUpperBound /= 2;
      }
      if (ub % newUpperBound != 0) {
        // Not supported configuration yet
        continue;
      }

      if (!foundHoistKDim) {
        continue;
      }

      hoistLoopSpecs.push_back({newUpperBound, hoistFactor});
      currentSharedMemoryUsage -= newMemoryUsed;
      validLoads.insert({aScaleLoadOp, bScaleLoadOp});
    }
  }
}

int isUpperBoundConstant(scf::ForOp forOp) {
  auto ub = forOp.getUpperBound();
  if (auto constant = dyn_cast<arith::ConstantOp>(ub.getDefiningOp())) {
    return cast<IntegerAttr>(constant.getValue()).getInt();
  } else {
    llvm::outs() << "Non constant upper bound??\n";
    return 0;
  }
}

// make a new make_range op that extend the existing end to be end*ub
// and update tensor shape accordingly
Value extendMakeRange(OpBuilder &builder, triton::MakeRangeOp makeRangeOp,
                      int64_t hoistDimSize) {
  auto tensorTy = dyn_cast<RankedTensorType>(makeRangeOp.getType());
  auto tensorShape = tensorTy.getShape();
  assert(tensorShape.size() == 1 && "make_range should be 1D");
  auto elemTy = tensorTy.getElementType();
  auto enc = tensorTy.getEncoding();
  int makeRangeNewEnd = hoistDimSize;
  SmallVector<int64_t> newTensorShape(1, makeRangeNewEnd);
  RankedTensorType newTensorTy =
      RankedTensorType::get(newTensorShape, elemTy, enc);
  Value range = builder.create<triton::MakeRangeOp>(
      makeRangeOp.getLoc(), newTensorTy, 0, makeRangeNewEnd);
  return range;
}

Value extendBroadcast(OpBuilder &builder, Operation *op, int dim,
                      int64_t hoistedDimSize, Value newSrc) {
  auto broadcastOp = dyn_cast<triton::BroadcastOp>(op);
  assert(broadcastOp && "We are not starting with a broadcast op");
  auto bTensorTy = dyn_cast<RankedTensorType>(broadcastOp.getType());
  auto bShape = bTensorTy.getShape();
  SmallVector<int64_t> newShape(bShape.begin(), bShape.end());
  newShape[dim] = hoistedDimSize;
  RankedTensorType newBTensorTy = RankedTensorType::get(
      newShape, bTensorTy.getElementType(), bTensorTy.getEncoding());
  return builder.create<triton::BroadcastOp>(broadcastOp.getLoc(), newBTensorTy,
                                             newSrc);
}

Value expandPathBcastM(Operation *bcastM, OpBuilder &builder,
                       int64_t hoistDimSize) {
  // Assume the following chain of IRs
  // %0 = make_range {0, 128}
  // %1 = expand_dims %0: -> tensor<1x128>
  // optional %strided = muli %1, %cst -> tensor<1x128>
  // %2 = broadcast %1/%strided: --> tensor<16x128>
  // bcastM is the broadcast op
  auto broadcastOp = dyn_cast<triton::BroadcastOp>(bcastM);
  assert(broadcastOp && "We are not starting with a broadcast op");
  auto bcastKParentOp = broadcastOp.getSrc().getDefiningOp();
  auto strideMulOp = dyn_cast<arith::MulIOp>(bcastKParentOp);
  Value strideVal;
  if (strideMulOp) {
    strideVal = strideMulOp.getRhs();
    assert(
        (isa<arith::ConstantOp, triton::SplatOp>(strideVal.getDefiningOp())) &&
        "anticipated K stride is not constant or single scalar");
    bcastKParentOp = strideMulOp.getLhs().getDefiningOp();
  }
  auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(bcastKParentOp);
  assert(expandDimsOp && "broadcast's parent must be a expand_dims op");
  auto expandDimsOpParent = expandDimsOp.getSrc().getDefiningOp();
  auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(expandDimsOpParent);
  assert(makeRangeOp && "expandDims' parent must be a make_range op");

  // new make_range {0, 128*ub}
  auto newMakeRangeValue = extendMakeRange(builder, makeRangeOp, hoistDimSize);
  // new expand_dims 1x{128*ub}
  int expandDim = expandDimsOp.getAxisAttr().getInt();
  mlir::Value newExpandDimsValue = builder.create<triton::ExpandDimsOp>(
      expandDimsOp.getLoc(), newMakeRangeValue, expandDim);

  if (strideMulOp) {
    auto loc = strideMulOp.getLoc();
    RankedTensorType oldStrideTy = cast<RankedTensorType>(strideVal.getType());
    SmallVector<int64_t> newStrideShape(oldStrideTy.getShape());
    assert(newStrideShape.size() == 2);
    newStrideShape[1] = hoistDimSize;
    auto newStrideTy =
        RankedTensorType::get(newStrideShape, oldStrideTy.getElementType(),
                              oldStrideTy.getEncoding());
    Value newStrideValue;
    if (auto strideConstant =
            dyn_cast<arith::ConstantOp>(strideVal.getDefiningOp())) {
      DenseElementsAttr strideConstantAttr =
          dyn_cast<mlir::DenseElementsAttr>(strideConstant.getValueAttr());
      assert(strideConstantAttr && "K stride constant is not Dense");
      assert(strideConstantAttr.isSplat() &&
             "The attribute of the constantOp is not a splat");

      auto reshapedStrideAttr = strideConstantAttr.resizeSplat(newStrideTy);

      newStrideValue = builder.create<arith::ConstantOp>(loc, newStrideTy,
                                                         reshapedStrideAttr);
    } else {
      auto strideSplat = dyn_cast<triton::SplatOp>(strideVal.getDefiningOp());
      assert(strideSplat && "expect strideVal defining op to be splat");
      Value strideScalar = strideSplat.getSrc();
      newStrideValue =
          builder.create<triton::SplatOp>(loc, newStrideTy, strideScalar);
    }

    newExpandDimsValue =
        builder.create<arith::MulIOp>(loc, newExpandDimsValue, newStrideValue);
  }

  // erase ops
  // makeRangeOp.erase();
  // expandDimsOp.erase();

  // new broadcast
  return extendBroadcast(builder, bcastM, /*which dim to extend*/ 1,
                         hoistDimSize, newExpandDimsValue);
}

Value expandPathBcastK(Operation *bcastK, OpBuilder &builder, int ub) {
  // %27 = tt.broadcast %24 : tensor<16x1x!tt.ptr<f16>, #blocked> ->
  // tensor<16x128x!tt.ptr<f16>, #blocked> extend shape[1] with shape[1]*ub

  auto broadcastOp = dyn_cast<triton::BroadcastOp>(bcastK);

  return broadcastOp.getResult();
}

Value createLocalAlloc(OpBuilder &builder, Location loc, Value loadVal,
                       int64_t hoistKSize, int64_t blockKSize,
                       StringRef archGen) {
  auto tensorTy = dyn_cast<RankedTensorType>(loadVal.getType());
  SmallVector<int64_t> bufferShape(tensorTy.getShape().begin(),
                                   tensorTy.getShape().end());
  // llvm::outs() << "bufferShape(" << bufferShape.size() << "): ("
  //              << bufferShape[0] << ", " << bufferShape[1] << ")\n";

  auto isaFamily = triton::AMD::deduceISAFamily(archGen);
  const unsigned numBanks = (isaFamily == ISAFamily::CDNA4) ? 64 : 32;
  const unsigned bankBitWidth = 32;
  const unsigned simdWidth = 16;
  constexpr int elemBitWidth = 8; // scale is 8-bit E8M0 float
  int elemsPerOneBanksRow = (numBanks * bankBitWidth) / elemBitWidth;
  int perPhase =
      std::max(1, static_cast<int>(elemsPerOneBanksRow / hoistKSize));
  int maxPhase =
      std::max(std::min(simdWidth / perPhase,
                        static_cast<unsigned>(hoistKSize / blockKSize)),
               1u);

  Type eType = tensorTy.getElementType();
  auto CTALayout = ttg::getCTALayout(tensorTy.getEncoding());
  auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
      tensorTy.getContext(), blockKSize, perPhase, maxPhase,
      ttg::getOrder(cast<ttg::DistributedEncodingTrait>(tensorTy.getEncoding()),
                    tensorTy.getShape()),
      CTALayout);
  auto ldsBufferType = ttg::MemDescType::get(
      bufferShape, eType, sharedEnc,
      triton::gpu::SharedMemorySpaceAttr::get(tensorTy.getContext()),
      /*mutableMemory=*/true);
  return builder.create<ttg::LocalAllocOp>(loc, ldsBufferType, loadVal);
}

Value createLocalAlloc(OpBuilder &builder, Location loc,
                       RankedTensorType tensorTy, int64_t hoistKSize,
                       int64_t blockKSize, StringRef archGen) {
  SmallVector<int64_t> bufferShape(tensorTy.getShape().begin(),
                                   tensorTy.getShape().end());

  auto isaFamily = triton::AMD::deduceISAFamily(archGen);
  const unsigned numBanks = (isaFamily == ISAFamily::CDNA4) ? 64 : 32;
  const unsigned bankBitWidth = 32;
  const unsigned simdWidth = 16;
  constexpr int elemBitWidth = 8; // scale is 8-bit E8M0 float
  int elemsPerOneBanksRow = (numBanks * bankBitWidth) / elemBitWidth;
  int perPhase =
      std::max(1, static_cast<int>(elemsPerOneBanksRow / hoistKSize));
  int maxPhase =
      std::max(std::min(simdWidth / perPhase,
                        static_cast<unsigned>(hoistKSize / blockKSize)),
               1u);

  Type eType = tensorTy.getElementType();
  auto CTALayout = ttg::getCTALayout(tensorTy.getEncoding());
  auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
      tensorTy.getContext(), blockKSize, perPhase, maxPhase,
      ttg::getOrder(cast<ttg::DistributedEncodingTrait>(tensorTy.getEncoding()),
                    tensorTy.getShape()),
      CTALayout);
  auto ldsBufferType = ttg::MemDescType::get(
      bufferShape, eType, sharedEnc,
      triton::gpu::SharedMemorySpaceAttr::get(tensorTy.getContext()),
      /*mutableMemory=*/true);
  return builder.create<ttg::LocalAllocOp>(loc, ldsBufferType);
}

Value widen2dPtrCase(OpBuilder builder, Operation *aPtrs, int64_t hoistKSize) {
  // We assume the operands of this addptr come from broadcast
  Operation *bcastK, *bcastM;
  for (Value ptrOperand : aPtrs->getOperands()) {
    Operation *broadcastOp = ptrOperand.getDefiningOp();
    Value bcastSrc = broadcastOp->getOperand(0);
    auto srcShape = dyn_cast<RankedTensorType>(bcastSrc.getType()).getShape();
    if (srcShape[1] == 1)
      bcastK = broadcastOp;
    else // srcShape[0] == 1
      bcastM = broadcastOp;
  }

  // addptr has form: res = addptr ptr, offset
  // bcastM refers to the broadcast along M dim, which is assumed to be the
  // offset of the addptr. So its shape is <1 x BLOCK_K> --> <BLOCK_M x BLOCK_K>
  // bcastK refers to the broadcast along K dim, which is assumed to be the
  // ptr of the addptr. So its shape is <BLOCK_M x 1> --> <BLOCK_M x BLOCK_K>
  //
  // We also assume that bcastM comes from the chain of the following ops
  // 1. make_range <BLOCK_K>
  // 2. expand_dims <BLOCK_K> --> <1xBLOCK_K>
  // 3. broadcast <1 x BLOCK_K> --> <BLOCK_M x BLOCK_K>
  // Therefore, we need to go all the way to make_range and extend BLOCK_K
  // to BLOCK_K*ub
  //
  // For bcastK, we only need to extend the broadcast to be
  // <BLOCK_M x 1> --> <BLOCK_M x {BLOCK_K*ub}>

  auto newBcastMVal = expandPathBcastM(bcastM, builder, hoistKSize);
  auto newBcastKVal =
      extendBroadcast(builder, bcastK, /*which dim to extend*/ 1, hoistKSize,
                      dyn_cast<triton::BroadcastOp>(bcastK).getSrc());

  // After expanding BLOCK_K to BLOCK_K*ub, we create the new addptr
  // with the new broadcast values: addptr newBcastKVal, newBcastMVal
  auto newPtrVal = builder.create<triton::AddPtrOp>(
      aPtrs->getLoc(), newBcastKVal.getType(), newBcastKVal, newBcastMVal);
  return newPtrVal;
}

Value widen1dPtrCase(OpBuilder builder, triton::BroadcastOp bcast,
                     int64_t hoistKSize) {
  // %34 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32,
  // #ttg.slice<{dim = 0, parent = #blocked4}>> loc(#loc34) %36 = tt.expand_dims
  // %34 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent =
  // #blocked4}>> -> tensor<1x8xi32, #blocked4> loc(#loc34) %37 = tt.splat %arg3
  // : !tt.ptr<i8> -> tensor<1x8x!tt.ptr<i8>, #blocked4> loc(#loc35) %38 =
  // tt.addptr %37, %36 : tensor<1x8x!tt.ptr<i8>, #blocked4>, tensor<1x8xi32,
  // #blocked4> loc(#loc35) %39 = tt.broadcast %38 : tensor<1x8x!tt.ptr<i8>,
  // #blocked4> -> tensor<32x8x!tt.ptr<i8>, #blocked4> loc(#loc35)
  auto addPtr = dyn_cast<triton::AddPtrOp>(bcast.getSrc().getDefiningOp());
  assert(addPtr && "expected addPtr before broadcast in 1d case");
  auto baseSplat =
      dyn_cast<triton::SplatOp>(addPtr.getOperand(0).getDefiningOp());
  assert(baseSplat && "expeceted splat befor addPtr");
  auto expandDims =
      dyn_cast<triton::ExpandDimsOp>(addPtr.getOperand(1).getDefiningOp());
  assert(expandDims && "expected expand dims before addPtr");
  auto makeRange =
      dyn_cast<triton::MakeRangeOp>(expandDims.getSrc().getDefiningOp());
  assert(makeRange && "expected make range before expand dims");

  // new make_range
  auto newMakeRangeValue = extendMakeRange(builder, makeRange, hoistKSize);
  // new expand_dims
  int expandDim = expandDims.getAxisAttr().getInt();
  mlir::Value newExpandDimsValue = builder.create<triton::ExpandDimsOp>(
      expandDims.getLoc(), newMakeRangeValue, expandDim);
  // new splat ptr
  auto basePtr = baseSplat.getSrc();
  SmallVector<int64_t> newSplatShape(baseSplat.getType().getShape());
  assert(newSplatShape[0] == 1 && "expect M/N == 1");
  assert(newSplatShape.size() == 2);
  newSplatShape[1] = hoistKSize;
  auto newSplatTy = RankedTensorType::get(newSplatShape, basePtr.getType(),
                                          baseSplat.getType().getEncoding());
  auto newBaseSplat =
      builder.create<triton::SplatOp>(baseSplat.getLoc(), newSplatTy, basePtr);
  // new addptr
  auto newAddPtrTy = newBaseSplat.getType();
  auto newAddPtr = builder.create<triton::AddPtrOp>(
      addPtr.getLoc(), newAddPtrTy, newBaseSplat, newExpandDimsValue);
  // new broadcast
  auto newBcastValue = extendBroadcast(
      builder, bcast, /*which dim to extend*/ 1, hoistKSize, newAddPtr);
  return newBcastValue;
}

struct HoistedLoad {
  Operation *loadOp;
  Value localAllocVal;
  Value asyncTokenVal;
};

HoistedLoad hoistLoad(scf::ForOp forOp, Operation *op, int64_t newUpperBound,
                      int ub, StringRef archGen) {
  triton::LoadOp loadOp = dyn_cast<triton::LoadOp>(op);
  int64_t blockKSize =
      dyn_cast<RankedTensorType>(loadOp.getPtr().getType()).getShape().back();
  // Here I assume
  // 1. There is no mask along k dim in the loadOp
  // 2. The ptr of loadOp comes from a block arg of the loop
  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);

  // Dealing with mask
  Value maskM = loadOp.getMask();
  Value newMaskVal, newOtherVal;
  int64_t hoistKSize = llvm::PowerOf2Ceil(blockKSize * newUpperBound);
  if (maskM) {
    // We assume the mask along the M dim is NOT loop carried
    assert(maskM.getParentRegion() != forOp.getRegion() &&
           "load mask should not be loop carried");
    Operation *maskOp = maskM.getDefiningOp();
    auto bcastMask = dyn_cast<tt::BroadcastOp>(maskOp);
    assert(bcastMask && "load mask does not come from a broadcast op");

    newMaskVal = extendBroadcast(builder, maskOp, /*dim*/ 1,
                                 /*hoist size*/ hoistKSize, bcastMask.getSrc());

    // Dealing with other
    Value other = loadOp.getOther();
    auto otherConstant = dyn_cast<arith::ConstantOp>(other.getDefiningOp());
    // auto attr = otherConstant.getValue();
    auto denseAttr =
        dyn_cast<DenseFPElementsAttr>(otherConstant.getValueAttr());
    auto ty = dyn_cast<RankedTensorType>(denseAttr.getType());
    SmallVector<int64_t> newShape(ty.getShape().begin(), ty.getShape().end());
    newShape[1] = hoistKSize;
    auto newTy =
        RankedTensorType::get(newShape, ty.getElementType(), ty.getEncoding());
    assert(denseAttr.isSplat() &&
           "The attribute of the constantOp is not a splat");
    auto reshapedAttr = denseAttr.resizeSplat(newTy);
    newOtherVal =
        builder.create<arith::ConstantOp>(forOp.getLoc(), newTy, reshapedAttr);
  }

  // Dealing with ptr
  auto blockArg = dyn_cast<BlockArgument>(loadOp.getOperand(0));
  assert(blockArg && "ptr is not a block arg");

  OpOperand &operand = *forOp.getTiedLoopInit(blockArg);
  // This is assumed to be the addptr op to compute the final aptrs for loadOp
  // say %29 = tt.addptr %27, %28 : tensor<16x128x!tt.ptr<f16>, #blocked>
  Operation *aPtrs = operand.get().getDefiningOp();
  Value newPtrVal;
  if (isa<triton::AddPtrOp>(aPtrs)) {
    newPtrVal = widen2dPtrCase(builder, aPtrs, hoistKSize);
  } else if (auto bcast = dyn_cast<triton::BroadcastOp>(aPtrs)) {
    newPtrVal = widen1dPtrCase(builder, bcast, hoistKSize);
  }

  std::string useAsyncCopy =
      mlir::triton::tools::getStrEnv("TRITON_HIP_USE_ASYNC_COPY");
  if (useAsyncCopy == "" || useAsyncCopy == "0") {
    // The we create the aggregated load with the "fat" pointer
    // create: load newPtr
    Value aggregatedLoadVal;
    if (maskM)
      aggregatedLoadVal = builder.create<triton::LoadOp>(
          forOp.getLoc(), newPtrVal, newMaskVal, newOtherVal, loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile());
    else
      aggregatedLoadVal = builder.create<triton::LoadOp>(
          forOp.getLoc(), newPtrVal, loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

    // Store loaded tensor into LDS
    Value localAllocVal =
        createLocalAlloc(builder, forOp.getLoc(), aggregatedLoadVal, hoistKSize,
                         blockKSize, archGen);
    HoistedLoad result;
    result.loadOp = aggregatedLoadVal.getDefiningOp();
    result.localAllocVal = localAllocVal;
    return result;
  } else {
    auto ty = dyn_cast<RankedTensorType>(loadOp.getType());
    SmallVector<int64_t> newShape(ty.getShape().begin(), ty.getShape().end());
    newShape[1] = hoistKSize;
    auto newTy =
        RankedTensorType::get(newShape, ty.getElementType(), ty.getEncoding());

    // The we create the aggregated load with the "fat" pointer
    // create: load newPtr
    auto loadTy = cast<RankedTensorType>(newTy);
    Value localAllocVal = createLocalAlloc(builder, forOp.getLoc(), loadTy,
                                           hoistKSize, blockKSize, archGen);
    Value aggregatedLoadToken;
    aggregatedLoadToken = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
        forOp.getLoc(), newPtrVal, localAllocVal, newMaskVal, newOtherVal,
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    HoistedLoad result;
    result.loadOp = aggregatedLoadToken.getDefiningOp();
    result.localAllocVal = localAllocVal;

    auto asyncCommitGroup = builder.create<ttg::AsyncCommitGroupOp>(
        forOp.getLoc(), aggregatedLoadToken);
    // create async wait
    result.asyncTokenVal = builder.create<ttg::AsyncWaitOp>(
        forOp.getLoc(), asyncCommitGroup.getResult(), 0);

    return result;
  }
}

void processLoopBody(scf::ForOp forOp, Operation *op,
                     HoistedLoad localAllocData) {
  // Now we have hoisted the loadOp out of the loop and make it "fat"
  // We have also inserted a local_alloc op right after the load to put
  // everything into LDS.
  // Now we need to process the forOp:
  // The current loop body has the following chain:
  // 1. aScale = load aScale_ptrs #linear
  // 2. bScale = load bScale_ptrs #linear
  // 3. acc = dot opA, aScale, opB, bScale, acc
  //
  // What we need is to replace the above with
  // 1. bufOff = i * BLOCK_K
  // 2. aScaleLocalBuf = memdesc_subview aScaleLdsBuffer[0, bufOff]
  // 3. bScaleLocalBuf = memdesc_subview bScaleLdsBuffer[0, bufOff]
  // 4. aScale = local_load aScaleLocalBuf
  // 5. bScale = local_load bScaleLocalBuf
  // 6. acc = dot opA, aScale, opB, bScale, acc
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  builder.setInsertionPoint(op);
  // step 1: bufOff = i * BLOCK_K
  auto loadOp = dyn_cast<tt::LoadOp>(op);
  // llvm::outs() << "ORIGINAL LOAD:" << loadOp << "\n";
  auto subviewShape = dyn_cast<RankedTensorType>(loadOp.getType()).getShape();

  Value BLOCK_K =
      builder.create<arith::ConstantIntOp>(loc, subviewShape[1], 32);
  auto forOpIV = forOp.getInductionVar();
  auto bufOffVal = builder.create<arith::MulIOp>(loc, forOpIV, BLOCK_K);

  // step 2: localBuf = memdesc_subview ldsBuffer[0, bufOff]
  SmallVector<Value> localBufOff(2);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  localBufOff[0] = zero;      // along M dim
  localBufOff[1] = bufOffVal; // along K dim

  Value localAllocVal = localAllocData.localAllocVal;
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(localAllocVal.getType());
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      subviewShape, allocTy.getElementType(), allocTy.getEncoding(),
      sharedMemorySpace, /*mutableMemory=*/true, allocTy.getShape());
  auto ldsSubview = builder.create<ttg::MemDescSubviewOp>(
      loc, subviewTy, localAllocVal, localBufOff);

  // step 3 & 4: local_load & replace opA in dotScaledOp
  Operation *use = *loadOp.getResult().getUsers().begin();
  if (isa<triton::DotScaledOp>(use)) {
    // direct load
    auto localLoadVal = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), ldsSubview, localAllocData.asyncTokenVal);
    loadOp.getResult().replaceAllUsesWith(localLoadVal);
  } else {
    assert(isa<ttg::ConvertLayoutOp>(use) &&
           "User of load scale should be either cvt or dot_scaled");
    auto cvt = dyn_cast<ttg::ConvertLayoutOp>(use);
    auto localLoadVal = builder.create<ttg::LocalLoadOp>(
        loc, cvt.getType(), ldsSubview, localAllocData.asyncTokenVal);

    cvt.getResult().replaceAllUsesWith(localLoadVal);
    cvt.erase();
  }

  // step 5: cleanup
  auto blockArg = dyn_cast<BlockArgument>(loadOp.getOperand(0));
  for (OpOperand &operand : blockArg.getUses()) {
    auto user = operand.getOwner();
    // Skip the loadOp, which will be removed later
    if (user != loadOp) {
      // We will not update the blockArg (ptr) of the hoisted load
      // So we replace all uses of the updated ptr with the original one
      user->getResult(0).replaceAllUsesWith(blockArg);
      user->erase();
    }
  }
  loadOp.erase();
}

// TODO: replace with general axis analysis.
// for now this function is a plain hack
Value getKStride(OpBuilder &builder, Value ptr) {
  // %56 = arith.muli %54, %cst_1 : tensor<1x8xi32, #blocked2> loc(#loc43)
  // %57 = tt.broadcast %51 : tensor<32x1x!tt.ptr<i8>, #blocked2> ->
  // tensor<32x8x!tt.ptr<i8>, #blocked2> loc(#loc44) %58 = tt.broadcast %56 :
  // tensor<1x8xi32, #blocked2> -> tensor<32x8xi32, #blocked2> loc(#loc44) %59 =
  // tt.addptr %57, %58 :
  auto loc = ptr.getLoc();

  auto addptr = dyn_cast<triton::AddPtrOp>(ptr.getDefiningOp());
  if (!addptr)
    return Value();

  triton::BroadcastOp bcast;
  for (auto bcastVal : addptr.getOperands()) {
    bcast = dyn_cast<triton::BroadcastOp>(bcastVal.getDefiningOp());
    if (!bcast)
      return Value();
    auto opShape = bcast.getSrc().getType().getShape();
    if (opShape[0] == 1)
      break;
  }

  auto mulOp = dyn_cast<arith::MulIOp>(bcast.getSrc().getDefiningOp());
  if (!mulOp)
    return Value();

  if (auto cst = dyn_cast<arith::ConstantOp>(mulOp.getRhs().getDefiningOp())) {
    auto strideConstantAttr =
        dyn_cast<mlir::DenseElementsAttr>(cst.getValueAttr());
    auto strideAttr = strideConstantAttr.getSplatValue<IntegerAttr>();
    int64_t constStride = strideAttr.getInt();
    return builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(constStride));
  } else if (auto splat =
                 dyn_cast<triton::SplatOp>(mulOp.getRhs().getDefiningOp())) {
    return splat.getSrc();
  }
  assert(false && "expect constantOp or splatOp as a stride multiplier");
}

void generateOuterLoop(scf::ForOp forOp, HoistedLoad hoistedAScale,
                       HoistedLoad hoistedBScale, int64_t hoistFactor,
                       int64_t newUpperBound) {
  // Set up ops/info required to build outer loop.
  auto aScaleLocalAllocOp = llvm::cast<ttg::LocalAllocOp>(
      hoistedAScale.localAllocVal.getDefiningOp());
  auto bScaleLocalAllocOp = llvm::cast<ttg::LocalAllocOp>(
      hoistedBScale.localAllocVal.getDefiningOp());
  auto aScalePtr = hoistedAScale.loadOp->getOperand(0);
  auto bScalePtr = hoistedBScale.loadOp->getOperand(0);
  assert(aScalePtr && bScalePtr &&
         "Expected src of local alloc to be loadOp to generate outer loop.");

  int64_t hoistKSize =
      dyn_cast<RankedTensorType>(aScalePtr.getType()).getShape().back();

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Value lb =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  Value ub = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(hoistFactor));
  Value step =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
  ValueRange oldInits = forOp.getInits();
  int innerUB = isUpperBoundConstant(forOp);
  Value newInnerUB = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(newUpperBound));

  auto createGlobalLoadLocalAlloc =
      [&builder, &hoistKSize](Location loc, HoistedLoad loadData,
                              Value offsetEl) -> std::tuple<Value, Value> {
    auto aPtr = loadData.loadOp->getOperand(0);
    auto aPtrTy = cast<RankedTensorType>(aPtr.getType());
    auto offsetTy = RankedTensorType::get(
        aPtrTy.getShape(), builder.getIntegerType(32), aPtrTy.getEncoding());
    Value offset = builder.create<tt::SplatOp>(loc, offsetTy, offsetEl);
    Value newAPtr = builder.create<tt::AddPtrOp>(loc, aPtrTy, aPtr, offset);

    if (!loadData.asyncTokenVal) {
      IRMapping loadMapping;
      loadMapping.map(aPtr, newAPtr);
      Operation *newLoadOp = builder.clone(*loadData.loadOp, loadMapping);

      auto newLocalAllocOp = builder.create<ttg::LocalAllocOp>(
          loc, loadData.localAllocVal.getType(), newLoadOp->getResults()[0]);

      return {newLoadOp->getResults()[0], newLocalAllocOp.getResult()};
    } else {
      IRMapping loadMapping;
      loadMapping.map(aPtr, newAPtr);
      Operation *newLoadOp = builder.clone(*loadData.loadOp, loadMapping);

      // Create new token
      auto asyncCommitGroup = builder.create<ttg::AsyncCommitGroupOp>(
          loc, newLoadOp->getResults()[0]);
      // create async wait
      Value newToken = builder.create<ttg::AsyncWaitOp>(
          loc, asyncCommitGroup.getResult(), 0);

      return {newToken, loadData.localAllocVal};
    }
  };

  auto outerDimLoop = builder.create<scf::ForOp>(
      loc, lb, ub, step, oldInits,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value aKStride = getKStride(builder, aScalePtr);
        Value bKStride = getKStride(builder, bScalePtr);

        Value offsetElA = builder.create<arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(hoistKSize));
        if (aKStride)
          offsetElA = builder.create<arith::MulIOp>(loc, offsetElA, aKStride);
        offsetElA = builder.create<arith::MulIOp>(loc, iv, offsetElA);

        Value offsetElB = builder.create<arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(hoistKSize));
        if (bKStride)
          offsetElB = builder.create<arith::MulIOp>(loc, offsetElB, bKStride);
        offsetElB = builder.create<arith::MulIOp>(loc, iv, offsetElB);

        auto [newAScaleLoadedVal, newAScaleLocalAllocVal] =
            createGlobalLoadLocalAlloc(loc, hoistedAScale, offsetElA);
        auto [newBScaleLoadedVal, newBScaleLocalAllocVal] =
            createGlobalLoadLocalAlloc(loc, hoistedBScale, offsetElB);

        IRMapping mapping;
        mapping.map(hoistedAScale.loadOp->getResult(0), newAScaleLoadedVal);
        mapping.map(hoistedBScale.loadOp->getResult(0), newBScaleLoadedVal);
        if (hoistedAScale.asyncTokenVal)
          mapping.map(hoistedAScale.asyncTokenVal, newAScaleLoadedVal);
        if (hoistedBScale.asyncTokenVal)
          mapping.map(hoistedBScale.asyncTokenVal, newBScaleLoadedVal);
        mapping.map(hoistedAScale.localAllocVal, newAScaleLocalAllocVal);
        mapping.map(hoistedBScale.localAllocVal, newBScaleLocalAllocVal);
        for (auto [index, initVal] : llvm::enumerate(oldInits)) {
          mapping.map(initVal, args[index]);
        }
        Operation *newInnerLoop = builder.clone(*forOp.getOperation(), mapping);
        auto newInnerForOp = llvm::cast<scf::ForOp>(newInnerLoop);
        newInnerForOp.setUpperBound(newInnerUB);
        builder.create<scf::YieldOp>(loc, newInnerLoop->getResults());
        // auto unrolled = loopUnrollFull(newInnerForOp);
      });
  forOp.getResults()[0].replaceAllUsesWith(outerDimLoop.getResults()[0]);
  forOp.erase();

  if (hoistedAScale.asyncTokenVal) {
    auto asyncWait = hoistedAScale.asyncTokenVal.getDefiningOp();
    auto commitGroup = asyncWait->getOperand(0).getDefiningOp();
    asyncWait->erase();
    commitGroup->erase();
  }

  if (hoistedBScale.asyncTokenVal) {
    auto asyncWait = hoistedBScale.asyncTokenVal.getDefiningOp();
    auto commitGroup = asyncWait->getOperand(0).getDefiningOp();
    asyncWait->erase();
    commitGroup->erase();
  }

  if (hoistedAScale.loadOp->use_empty()) {
    hoistedAScale.loadOp->erase();
  }

  if (hoistedBScale.loadOp->use_empty()) {
    hoistedBScale.loadOp->erase();
  }

  if (aScaleLocalAllocOp->use_empty()) {
    aScaleLocalAllocOp.erase();
  }

  if (bScaleLocalAllocOp->use_empty()) {
    bScaleLocalAllocOp.erase();
  }

  if (hoistedAScale.loadOp->use_empty()) {
    hoistedAScale.loadOp->erase();
  }

  if (hoistedBScale.loadOp->use_empty()) {
    hoistedBScale.loadOp->erase();
  }
}

// Stream Pipeline
struct AggregateLoad : public TritonAMDGPUAggregateLoadBase<AggregateLoad> {
  AggregateLoad() = default;
  AggregateLoad(StringRef archGen, int factor) {
    this->archGenerationName = archGen.data();
    this->aggregateFactor = factor;

    std::string envAggregateFactor =
        mlir::triton::tools::getStrEnv("TRITON_HIP_AGGREGATE_LOAD_FACTOR");
    if (envAggregateFactor != "") {
      this->aggregateFactor = std::stoi(envAggregateFactor);
    }
  }

  void runOnOperation() override {
    if (aggregateFactor == 0 || aggregateFactor == 1) {
      return;
    }
    // return;
    int64_t totalSharedMemoryUsage = 0;
    bool foundDotScaledOp = false;
    getOperation()->walk([&](triton::DotScaledOp dotScaledOp) -> void {
      Value aScale = dotScaledOp.getAScale();
      Value bScale = dotScaledOp.getBScale();
      // Get aScale and bScale size.
      auto aScaleType = llvm::cast<ShapedType>(aScale.getType());
      auto bScaleType = llvm::cast<ShapedType>(bScale.getType());
      assert(aScaleType.hasStaticShape() &&
             "Expected tt.dot to have aScale as static shape.");
      assert(bScaleType.hasStaticShape() &&
             "Expected tt.dot to have bScale as static shape.");
      int64_t aScaleAllocSize = getAllocSize(aScaleType);
      int64_t bScaleAllocSize = getAllocSize(bScaleType);
      totalSharedMemoryUsage += aScaleAllocSize + bScaleAllocSize;
      assert(!foundDotScaledOp &&
             "Currently only support a single dot_scaled operation.");
      foundDotScaledOp = true;
    });
    // llvm::outs() << "before: " << *getOperation() << "\n";

    if (!foundDotScaledOp) {
      llvm::outs() << "Didn't find dotScaledOp for AggregateLoad\n";
      return;
    }

    // llvm::outs() << "Total smem Usage:" << totalSharedMemoryUsage << "\n";

    // Do the pipelining
    size_t cnt = 0;
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      // We need K to be constant, i.e. the upper bound of the loop is a
      // constant
      auto ub = isUpperBoundConstant(forOp);
      if (!ub)
        return;
      SetVector<std::pair<Operation *, Operation *>> validLoads;
      // newUpperBound, hoistFactor
      SmallVector<std::pair<int64_t, int64_t>> hoistLoopSpecs;
      findValidLoads(forOp, validLoads, hoistLoopSpecs, ub, aggregateFactor,
                     totalSharedMemoryUsage);
      // llvm::outs() << "validLoads.size(): " << validLoads.size() << "\n";
      for (auto [index, loadOps] : llvm::enumerate(validLoads)) {
        auto [newUpperBound, hoistFactor] = hoistLoopSpecs[index];
        auto [aScaleLoadOp, bScaleLoadOp] = loadOps;
        auto newAScaleData = hoistLoad(forOp, aScaleLoadOp, newUpperBound, ub,
                                       this->archGenerationName);
        auto newBScaleData = hoistLoad(forOp, bScaleLoadOp, newUpperBound, ub,
                                       this->archGenerationName);
        processLoopBody(forOp, aScaleLoadOp, newAScaleData);
        processLoopBody(forOp, bScaleLoadOp, newBScaleData);
        if (hoistFactor > 1) {
          generateOuterLoop(forOp, newAScaleData, newBScaleData, hoistFactor,
                            newUpperBound);
        }
        cnt++;
      }
    });
    // llvm::outs() << cnt << " for-loop has been modified\n";
    // llvm::outs() << "after: " << *getOperation() << "\n";
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPUAggregateLoadPass(std::string archGen,
                                          int aggregateFactor) {
  return std::make_unique<AggregateLoad>(archGen, aggregateFactor);
}
