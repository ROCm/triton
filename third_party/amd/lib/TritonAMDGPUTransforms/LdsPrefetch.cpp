//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot to avoid
// stalling on LDS load latency.
//
// Example transformation:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//
// It is assumed that the loop body has already been pipelined
// for loading data into LDS; now the data will be prefetched from
// lds to improve performance in two ways.
// (1) Rather than a pipelined for loop starting with local_loads
// and incurring an LDS latency at the top of the loop, we prefetch the first
// few local loads at the end of the previous iteration, so that the first op of
// the for loop can be a wmma (ideally).
// (2) Rather than the for loop
// structure being
// %a = ttg.local_load
// %b = ttg.local_load %d = tt.dot %a, %b, %c
// which lowers to llvm with all loads preceeding all wmmas, we prefetch the
// lds data for subtiles which naturally interleaves the local_loads and wmmas.
// Which looks like (for 2x2 slicing along M and N)
// scf.for ... iter_args(%a0, %b0, ...) {
// %b1 = ttg.local_load
// %d00 = tt.dot %a0, %b0, %c00
// %a1 = ttg.local_load
// %d01 = tt.dot %a0, %b1, %c01
// %d10 = tt.dot %a1, %b0, %c10
// %a0' = ttg.local_load
// %b0' = ttg.local_load
// %d11 = tt.dot %a1, %b1, %c11
// scf.yield %a0', %b0', ...
// }
// The improved llvm-ir interleaving can improve backend scheduling.
//
// To control the degree of prefetching, the dots are sliced along M, N, K
// according to prefetchWidthM,N,K. Because the dot is 3D sliced,
// many sliced dots and sliced local_loads are created,
// and the they are emitted such that there is 1 sliced dot between
// local_loads and the dot which depends on them; data for the 0th dot is moved
// to the end of the previous iteration. E.g.
//
// %a1 = ttg.local_load %a_tmp
// %d0 = tt.dot %a0, %b0, %c0 <-- hides ds_read latency
// %d1 = tt.dot %a1, %b1, %c1 <-- %a1 waited for here
//
// Prefetching *all* the A, B operands of dots from local memory would be
// prohibitively expensive in terms of registers and cycles.
// This pass supports slicing along K (which doesn't change the D, C operands
// of the dot) and slicing along M and N.
//
// Currently, prefetching has the following restrictions
// - Only a single dot per loop is supported; relaxing this will rely on have
// the local_loads of one dot being placed inside another sliced dot.
// - Intermediate ops between the dot and local_load are not supported;
// improving this requires being able to identify which ops (e.g. reshape,
// trans) support slicing and to what degree. Then need to prefetch the
// intermediate ops too.
// - DotScaled (with scales) is not supported; improving this requires creating
// new local_loads for scales which behave differently than for operands.
// - Ops which are prefetched at the end of the previous iteration must not
// have any dependencies in the loop body; all dependencies must be loop-carried
// out outside the loop. Therefore loops with async_copies must have the
// async_wait tokens not at the top of the loop, but rather at the end of the
// previous iteration.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/LdsPrefetchUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>
#include <tuple>
#undef LLVM_DEBUG
#define LLVM_DEBUG(X) X
#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-lds-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPULDSPREFETCH
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace ttg = triton::gpu;

namespace triton {
namespace amdgpu {

namespace {

// Helper function to split a value (Dot C operand) along a specific axis into
// numSlices. Performs Reshape + Transpose + Split + ConvertLayout.
static SmallVector<Value>
splitValueAlongAxis(Value input, int32_t numSlices, int axis,
                    SmallVector<RankedTensorType> &typesBeforeSplitting,
                    Location loc, OpBuilder &builder) {
  if (numSlices == 1) {
    return {input};
  }

  auto splitOnce = [&](Value val) -> std::pair<Value, Value> {
    RankedTensorType inputType = cast<RankedTensorType>(val.getType());
    auto shape = inputType.getShape();
    int rank = shape.size();
    assert(axis < rank);
    // Reshape to split the target axis into <..., 2, N/2, ...>
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[axis] /= 2;
    newShape.insert(newShape.begin() + axis, 2);
    Value reshaped = triton::ReshapeOp::create(builder, loc, newShape, val);

    // Permute so 2 is last dim <..., 2>
    rank++; // After reshape, we have one more dimension
    SmallVector<int32_t> order;
    for (int i = 0; i < rank; ++i) {
      if (i != axis)
        order.push_back(i);
    }
    order.push_back(axis);
    Value transposed = triton::TransOp::create(builder, loc, reshaped, order);

    // Split along the last dimension.
    triton::SplitOp split = triton::SplitOp::create(builder, loc, transposed);
    Attribute originalEncoding =
        cast<RankedTensorType>(input.getType()).getEncoding();
    SmallVector<Value> converted;
    for (Value result : split.getResults()) {
      auto resultType = cast<RankedTensorType>(result.getType());
      auto targetType = RankedTensorType::get(
          resultType.getShape(), resultType.getElementType(), originalEncoding);
      converted.push_back(
          ttg::ConvertLayoutOp::create(builder, loc, targetType, result));
    }
    return {converted[0], converted[1]};
  };

  // Iteratively split until we reach numSlices
  SmallVector<Value> tiles;
  tiles.push_back(input);
  int32_t currentCount = 1;
  while (currentCount < numSlices) {
    RankedTensorType tileType = cast<RankedTensorType>(tiles[0].getType());
    typesBeforeSplitting.push_back(tileType);
    SmallVector<Value> nextTiles;
    for (Value tile : tiles) {
      auto [left, right] = splitOnce(tile);
      nextTiles.push_back(left);
      nextTiles.push_back(right);
    }
    tiles = std::move(nextTiles);
    currentCount *= 2;
  }
  return tiles;
}

// Helper function (inverse of splitValueAlongAxis) to join Values (D opd of
// Dot) along a specific axis. Performs Join + Transpose + Reshape +
// ConvertLayout.
static Value
joinValuesAlongAxis(SmallVector<Value> tiles, int axis,
                    SmallVector<RankedTensorType> &typesBeforeSplitting,
                    Location loc, OpBuilder &builder) {

  auto joinOnce = [&](Value left, Value right,
                      RankedTensorType dstType) -> Value {
    auto leftType = cast<RankedTensorType>(left.getType());
    auto shape = leftType.getShape();
    int rank = shape.size();
    assert(axis < rank);

    // Join creates dim <..., 2>
    Value joined = triton::JoinOp::create(builder, loc, left, right);

    // Transpose to <..., 2, N,...>
    // for axis=0, trans=[2, 0, 1]
    // for axis=1, trans=[0, 2, 1]
    SmallVector<int32_t> trans(rank + 1);
    for (int j = 0; j < rank; ++j) {
      trans[j < axis ? j : j + 1] = j;
    }
    trans[axis] = rank;
    Value transposed = triton::TransOp::create(builder, loc, joined, trans);

    // Reshape to <..., 2N, ...>
    auto transposedType = cast<RankedTensorType>(transposed.getType());
    auto transposedShape = transposedType.getShape();
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[axis] *= 2;
    Value reshaped =
        triton::ReshapeOp::create(builder, loc, newShape, transposed);

    // Convert back to dst encoding (saved during splits)
    Value converted =
        ttg::ConvertLayoutOp::create(builder, loc, dstType, reshaped);
    return converted;
  };

  // Iteratively join
  while (tiles.size() > 1) {
    RankedTensorType dstType = typesBeforeSplitting.pop_back_val();
    SmallVector<Value> nextTiles;
    for (size_t i = 0; i < tiles.size(); i += 2) {
      Value joined = joinOnce(tiles[i], tiles[i + 1], dstType);
      nextTiles.push_back(joined);
    }
    tiles = std::move(nextTiles);
  }

  return tiles[0];
}

/// Create a new dot or dot_scaled op with the given slice operands.
/// For DotScaledOp, scale operands are looked up from the mapping.
static Operation *createDotOp(Operation *dotOp, OpBuilder &builder,
                              Location loc, RankedTensorType dType,
                              Value aSlice, Value bSlice, Value cSlice,
                              IRMapping *mapping) {
  if (auto dot = dyn_cast<triton::DotOp>(dotOp)) {
    return triton::DotOp::create(builder, loc, dType,
                                 ValueRange{aSlice, bSlice, cSlice},
                                 dot->getAttrs());
  }
  if (auto scaledDot = dyn_cast<triton::DotScaledOp>(dotOp)) {
    Value aScale;
    Value bScale;
    if (Value v = scaledDot.getAScale())
      aScale = mapping->lookup(v);
    if (Value v = scaledDot.getBScale())
      bScale = mapping->lookup(v);
    return triton::DotScaledOp::create(
        builder, loc, dType, aSlice, bSlice, cSlice, aScale, bScale,
        scaledDot.getAElemType(), scaledDot.getBElemType(),
        scaledDot.getFastMath(), scaledDot.getLhsKPack(),
        scaledDot.getRhsKPack());
  }
  llvm_unreachable("createDotOp: unsupported dot op type");
}

class Prefetcher {
public:
  Prefetcher() = delete;
  ~Prefetcher() = default;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();

private:
  /// Compute prefetch widths from dot encoding and shapes.
  /// Returns true if widths were set and this dot should be prefetched;
  /// false to skip this dot (e.g. kSize too small or dot not recognized).
  bool computePrefetchWidthForDotType(Attribute dotEncoding,
                                      unsigned aTypeBitWidth,
                                      ArrayRef<int64_t> dShape, unsigned mSize,
                                      unsigned nSize, unsigned kSize,
                                      unsigned kWidth, bool transA,
                                      bool transB);

  std::tuple<unsigned, unsigned, unsigned>
  computePrefetchWidth(unsigned mSize, unsigned nSize, unsigned kSize,
                       bool transA, bool transB, ArrayRef<unsigned> instrShape,
                       ArrayRef<unsigned> warpsPerCta, unsigned numInsts);

  FailureOr<Value> getAsyncWaitTokenForLocalLoad(Operation *localLoadOp,
                                                 bool fromPriorIter,
                                                 OpBuilder &builder,
                                                 IRMapping *mapping = nullptr);

  Value generateLocalLoad(Value v, unsigned opIdx, bool isPrologue,
                          Attribute dotEncoding, OpBuilder &builder,
                          std::optional<Value> asyncWaitToken = std::nullopt,
                          std::optional<int64_t> offsetM = std::nullopt,
                          std::optional<int64_t> shapeM = std::nullopt,
                          std::optional<int64_t> offsetN = std::nullopt,
                          std::optional<int64_t> shapeN = std::nullopt,
                          std::optional<int64_t> offsetK = std::nullopt,
                          std::optional<int64_t> shapeK = std::nullopt);

  Operation *generateDotsAndNonPrefetchingLocalLoads(Operation *dotOp,
                                                     Attribute dotEncoding,
                                                     OpBuilder &builder,
                                                     IRMapping &mapping,
                                                     scf::ForOp newForOp);

  void generatePrefetchingLocalLoads(Operation *dotOp, OpBuilder &builder,
                                     IRMapping &mapping,
                                     SmallVector<Value> &yieldValues);

  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  /// dots to be prefetched (DotOp or DotScaledOp, both implement
  /// DotOpInterface)
  SetVector<Operation *> dots;
  /// dot op => dot operand
  DenseMap<Operation *, Value> dot2aLoopArg;
  DenseMap<Operation *, Value> dot2aHeaderDef;
  DenseMap<Operation *, Value> dot2bLoopArg;
  DenseMap<Operation *, Value> dot2bHeaderDef;
  DenseMap<Operation *, Value> dot2aYield;
  DenseMap<Operation *, Value> dot2bYield;
  DenseMap<Operation *, SmallVector<Value>> dot2aVals;
  DenseMap<Operation *, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  /// Prefetch tile dimensions; set by computePrefetchWidths().
  unsigned prefetchWidthM;
  unsigned prefetchWidthN;
  unsigned prefetchWidthK;
  /// Store original kWidth to maintain when creating new local_loads.
  unsigned kWidth;
};

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<ttg::TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<Operation *> dotsInFor;
  for (Operation &op : *loop) {
    if (auto dotScaled = dyn_cast<triton::DotScaledOp>(&op)) {
      // TODO: need to support prefetching scales and slicing intermediate ops
      // before supporting DotScaledOp.
      if (dotScaled.getAScale() || dotScaled.getBScale()) {
        LDBG("DotScaledOp with scales is not supported.");
        LDBG(dotScaled);
        return failure();
      }
    }
    if (auto dotInterface = dyn_cast<triton::DotOpInterface>(&op)) {
      dotsInFor.push_back(&op);
    }
  }
  if (dotsInFor.empty())
    return failure();

  // TODO: enabling multiple dots per loop requires logic for prefetching
  // one dot's local_load inside another dot.
  if (dotsInFor.size() > 1)
    return failure();

  for (Operation *dot : dotsInFor) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    auto aOpd = dotInterface.getA();
    auto bOpd = dotInterface.getB();
    auto aType = cast<RankedTensorType>(aOpd.getType());
    auto bType = cast<RankedTensorType>(bOpd.getType());
    auto dType = cast<RankedTensorType>(dot->getResult(0).getType());
    auto aEnc = mlir::cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = mlir::cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
    assert(aEnc.getKWidth() == bEnc.getKWidth());
    kWidth = aEnc.getKWidth();

    auto transOp = [&](Operation *op, int opdIdx) -> bool {
      while (op) {
        if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
          auto srcType = localLoad.getSrc().getType();
          auto order = getOrder(srcType);
          return (order[0] == opdIdx);
        }
        if (op->getNumOperands() < 1)
          break;
        op = op->getOperand(0).getDefiningOp();
      }
      // For any other op type, fallback cautiously such that
      // subslices will be larger.
      return true;
    };

    // Get sizes for all three dimensions
    unsigned mSize = aType.getShape()[0];     // M dimension from operand A
    unsigned nSize = bType.getShape().back(); // N dimension from operand B
    unsigned kSize = aType.getShape().back(); // K dimension
    LDBG("dotShape: " << mSize << "x" << nSize << "x" << kSize);

    bool transA = transOp(aOpd.getDefiningOp(), 0);
    bool transB = transOp(bOpd.getDefiningOp(), 1);
    Attribute dotEncoding =
        cast<RankedTensorType>(dot->getResult(0).getType()).getEncoding();
    // TODO: calculating the prefetch(slicing) width also needs to examine
    // how the intermediate ops (between dot and local_load) are sliceable.
    if (!computePrefetchWidthForDotType(
            dotEncoding, aType.getElementTypeBitWidth(), dType.getShape(),
            mSize, nSize, kSize, kWidth, transA, transB))
      continue;
    LDBG("subtile: " << prefetchWidthM << "x" << prefetchWidthN << "x"
                     << prefetchWidthK);
    assert(prefetchWidthM > 0 && prefetchWidthN > 0 && prefetchWidthK > 0);
    auto aVals = ttg::findLocalLoadForDotOperand(dotInterface.getA());
    auto bVals = ttg::findLocalLoadForDotOperand(dotInterface.getB());

    if (succeeded(aVals) && succeeded(bVals)) {
      Value aSmem = aVals->front();
      Value bSmem = bVals->front();
      Value aHeaderDef = ttg::getIncomingLoopArg(forOp, aSmem);
      Value bHeaderDef = ttg::getIncomingLoopArg(forOp, bSmem);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aVals[dot] = *aVals;
        dot2bVals[dot] = *bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = ttg::getYieldOperand(forOp, yieldOp, aSmem);
        dot2bYield[dot] = ttg::getYieldOperand(forOp, yieldOp, bSmem);
      }
    } else {
      LDBG("findLocalLoadForDotOperand failed for dot: " << *dot);
      return failure();
    }
  }
  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Operation *dot : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
        dot2aVals[dot].back().getDefiningOp(), false, builder);
    FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
        dot2bVals[dot].back().getDefiningOp(), false, builder);
    Attribute dotEncoding =
        cast<RankedTensorType>(dot->getResult(0).getType()).getEncoding();
    Value aPrefetched = generateLocalLoad(
        dot2aHeaderDef[dot], 0, true, dotEncoding, builder,
        failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
    ttg::clonePrefetchElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched = generateLocalLoad(
        dot2bHeaderDef[dot], 1, true, dotEncoding, builder,
        failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
    ttg::clonePrefetchElementwiseOps(bPrefetched, dot2bVals[dot], builder);
    operand2headPrefetch[dotInterface.getA()] = aPrefetched;
    operand2headPrefetch[dotInterface.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (Operation *dot : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    loopArgs.push_back(operand2headPrefetch[dotInterface.getA()]);
    loopArgs.push_back(operand2headPrefetch[dotInterface.getB()]);
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // The insertion point should be placed before the yield op
  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // If we're currently trying to sink a prefetched dot, we need to stop
    // sinking it (by resetting the insertion point to the end) if we find
    // control flow, or anything that depends on the dot op.
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        if (dots.contains(def)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    if (dots.contains(&op)) {
      Attribute dotEncoding =
          cast<RankedTensorType>(op.getResult(0).getType()).getEncoding();
      newOp = generateDotsAndNonPrefetchingLocalLoads(&op, dotEncoding, builder,
                                                      mapping, newForOp);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (Operation *dot : dots) {
    generatePrefetchingLocalLoads(dot, builder, mapping, yieldValues);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

//------------------------------------------------------------------------------

/*
  AMD-specific prefetch is based on prefetching enough data early enough
  to not be stalled by lds latency, but not prefetch too much as this
  consumes excessive vgprs, and reduces cycles which can be used for
  HBM-latency hiding.

  The degree of prefetching is expressed as numInsts,
  which is the number of mfma instructions to prefetch lds data by.
  It is tuned from the lds latency cycles and the mfma cycles.
  E.g. if max lds latency is 40 cycles, and mfma is 16 cycles,
  then we choose 40/16 = 2.5 -> 4 (to get next power of 2).
  This will ensure there are 4 mfma in one sliced dot,
  and in the llvm, there will be 4 mfmas between a local_load
  and the mfma which depends on it.

  Therefore higher lds latency, and faster mfma instructions
  leads to larger numInsts -> larger prefetchWidths,
  fewer splits/joins, and we prefetch more data which uses more vgprs.

  numInsts is fed into computePrefetchWidth() to determine prefetchWidthMNK
  which specifies the tile shape of the sliced dots.
  The logic for this tile shape is
  - If tranposing A or B (e.g. ds_read_tr), then K and M or N are not sliced
    in order to preserve a minimum transpose width.
  - Whether the MxN shape prefers to be square or row; e.g. for numInst=4,
    square => 2x2 while row => 4x1. Square vs row impacts
    (a) Number of local_loads ops needed to prefetch data for first 4 mfmas.
    (b) How local_loads are distributed during the series of mfmas.
    (c) More research can be done here.
*/
bool Prefetcher::computePrefetchWidthForDotType(Attribute dotEncoding,
                                                unsigned aTypeBitWidth,
                                                ArrayRef<int64_t> dShape,
                                                unsigned mSize, unsigned nSize,
                                                unsigned kSize, unsigned kWidth,
                                                bool transA, bool transB) {
  if (auto mfmaEnc = dyn_cast<ttg::AMDMfmaEncodingAttr>(dotEncoding)) {
    unsigned numInsts = 4;
    std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) =
        computePrefetchWidth(mSize, nSize, kSize, transA, transB,
                             mfmaEnc.getInstrShape(), mfmaEnc.getWarpsPerCTA(),
                             numInsts);
    return true;
  }
  if (auto wmmaEnc = dyn_cast<ttg::AMDWmmaEncodingAttr>(dotEncoding)) {
    unsigned numInsts = 8;
    auto warpsPerCTA = ttg::getWarpsPerCTA(wmmaEnc, dShape);
    std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) =
        computePrefetchWidth(mSize, nSize, kSize, transA, transB,
                             wmmaEnc.getInstrShape(), warpsPerCTA, numInsts);
    return true;
  }
  return false;
}

std::tuple<unsigned, unsigned, unsigned> Prefetcher::computePrefetchWidth(
    unsigned mSize, unsigned nSize, unsigned kSize, bool transA, bool transB,
    ArrayRef<unsigned> instrShape, ArrayRef<unsigned> warpsPerCta,
    unsigned numInsts) {

  // minimum transpose width
  ModuleOp module = this->forOp.getOperation()->getParentOfType<ModuleOp>();
  std::optional<StringRef> arch = getAMDArch(module);
  if (!arch)
    return {mSize, nSize, kSize};
  std::string archStr = arch->str();
  unsigned mtw = 32;
  if (archStr == "gfx1250") {
    mtw = 128;
  } else if (archStr == "gfx942" || archStr == "gfx950" ||
             archStr == "gfx951") {
    mtw = 64;
  }

  LDBG("instrShape: " << instrShape[0] << "x" << instrShape[1] << "x"
                      << instrShape[2]);
  LDBG("warpsPerCta: " << warpsPerCta[0] << "x" << warpsPerCta[1]);
  // mnk specify num ops a sliced dot
  unsigned m = 1, n = 1, k = 1;
  unsigned maxM = mSize / (instrShape[0] * warpsPerCta[0]);
  unsigned maxN = nSize / (instrShape[1] * warpsPerCta[1]);
  unsigned maxK = kSize / (instrShape[2]);
  LDBG("maxInsts: " << maxM << "x" << maxN << "x" << maxK << " (p=" << numInsts
                    << ")");
  if (transA) {
    m = std::max<unsigned>(m, std::min<unsigned>(maxM, mtw / instrShape[0]));
    k = std::max<unsigned>(k, std::min<unsigned>(maxK, mtw / instrShape[2]));
  }
  if (transB) {
    n = std::max<unsigned>(n, std::min<unsigned>(maxN, mtw / instrShape[1]));
    k = std::max<unsigned>(k, std::min<unsigned>(maxK, mtw / instrShape[2]));
  }
  LDBG("numInstrsPerSubtile: " << m << "x" << n << "x" << k
                               << " (based on Trans)");
  numInsts /= (m * n * k);
  // Iteratively increase the tile shape until we reach numInsts
  // according to the preferred shape.
  // Currently, LLVM scheduling seems to schedule rows better than squares.
  bool preferSquare = false;
  while (numInsts > 1) {
    if ((m <= n || !preferSquare) && m < maxM && !transA) {
      m *= 2;
    } else if (n < maxN) {
      n *= 2;
    } else if (k < maxK) {
      k *= 2;
    } else {
      break;
    }
    numInsts /= 2;
  }
  LDBG("numInstrsPerSubtile: " << m << "x" << n << "x" << k);
  LDBG("numSubTiles: " << maxM / m << "x" << maxN / n << "x" << maxK / k);
  // convert num ops to CTA tile shape
  m *= instrShape[0] * warpsPerCta[0];
  n *= instrShape[1] * warpsPerCta[1];
  k *= instrShape[2];
  m = std::min<unsigned>(m, mSize);
  n = std::min<unsigned>(n, nSize);
  k = std::min<unsigned>(k, kSize);
  // Ensure these are power of 2; subsequent logic
  // of slicing/joining is based on these.
  assert(llvm::isPowerOf2_32(m) && "m must be a power of 2");
  assert(llvm::isPowerOf2_32(n) && "n must be a power of 2");
  assert(llvm::isPowerOf2_32(k) && "k must be a power of 2");
  return {m, n, k};
}

// Since dots have 3D slicing, the MemDescSubslice for loca loads
// will have 2D offsets and shapes.
Value Prefetcher::generateLocalLoad(
    Value v, unsigned opIdx, bool isPrologue, Attribute dotEncoding,
    OpBuilder &builder, std::optional<Value> asyncWaitToken,
    std::optional<int64_t> offsetM, std::optional<int64_t> shapeM,
    std::optional<int64_t> offsetN, std::optional<int64_t> shapeN,
    std::optional<int64_t> offsetK, std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<ttg::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // For operand A (opIdx=0): shape is [M, K], so mIdx=0, kIdx=1
  // For operand B (opIdx=1): shape is [K, N], so kIdx=0, nIdx=1
  int64_t mIdx = 0; // M dimension index (only for operand A)
  int64_t nIdx = 1; // N dimension index (only for operand B)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  // Handle m dim for opd A
  if (opIdx == 0) {
    offset[mIdx] = isPrologue ? 0 : prefetchWidthM;
    shape[mIdx] = isPrologue ? prefetchWidthM : (shape[mIdx] - prefetchWidthM);
    if (shapeM)
      shape[mIdx] = *shapeM;
    if (offsetM)
      offset[mIdx] = *offsetM;
  }

  // Handle n dim for opd B
  if (opIdx == 1) {
    offset[nIdx] = isPrologue ? 0 : prefetchWidthN;
    shape[nIdx] = isPrologue ? prefetchWidthN : (shape[nIdx] - prefetchWidthN);
    if (shapeN)
      shape[nIdx] = *shapeN;
    if (offsetN)
      offset[nIdx] = *offsetN;
  }

  // Handle k dim
  offset[kIdx] = isPrologue ? 0 : prefetchWidthK;
  shape[kIdx] = isPrologue ? prefetchWidthK : (shape[kIdx] - prefetchWidthK);
  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = ttg::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      ttg::MemDescType::get(shape, elementType, type.getEncoding(),
                            type.getMemorySpace(), type.getMutableMemory(),
                            type.getAllocShape()),
      v, offset);
  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);
  Value prefetchSlice = ttg::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem,
      asyncWaitToken.value_or(nullptr));
  return prefetchSlice;
}

// Insert a sched.barrier for MFMA/WMMA and lds ops.
void insertSchedBarrier(OpBuilder &builder, Location loc) {
  int32_t mask = 0 | (int32_t)mlir::amdgpu::sched_barrier_opt_enum::valu |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::salu |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::all_vmem |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::vmem_read |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::vmem_write |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::transcendental;
  ROCDL::SchedBarrier::create(builder, loc, mask);
}

// Generates all dots and first N-1 local_loads.
// First splits C opd along M and N, then loop over M, N, K creating dot
// sub-tiles and local_loads, and finally joins D opds along M and N.
Operation *Prefetcher::generateDotsAndNonPrefetchingLocalLoads(
    Operation *dotOp, Attribute dotEncoding, OpBuilder &builder,
    IRMapping &mapping, scf::ForOp newForOp) {
  auto dotInterface = cast<triton::DotOpInterface>(dotOp);
  // Get total dimensions from operands
  auto aType = cast<RankedTensorType>(dotInterface.getA().getType());
  auto bType = cast<RankedTensorType>(dotInterface.getB().getType());
  int64_t totalM = aType.getShape()[0];
  int64_t totalK = aType.getShape().back();
  int64_t totalN = bType.getShape().back();
  Location loc = dotOp->getLoc();

  // Map from (M, N) offsets to dot C/D opds
  DenseMap<std::pair<int32_t, int32_t>, Value> mnToDot;

  // Assert that dimensions are evenly divisible by prefetch widths
  assert(totalM % prefetchWidthM == 0 &&
         "totalM must be divisible by prefetchWidthM");
  assert(totalN % prefetchWidthN == 0 &&
         "totalN must be divisible by prefetchWidthN");
  assert(totalK % prefetchWidthK == 0 &&
         "totalK must be divisible by prefetchWidthK");
  // Slice c opd along M
  Value cOperand = mapping.lookup(dotOp->getOperand(2));
  int mAxis = 0;
  int nAxis = 1;
  int32_t numSlicesM = totalM / prefetchWidthM;
  SmallVector<RankedTensorType> typesBeforeSplitting;
  SmallVector<Value> mSlices = splitValueAlongAxis(
      cOperand, numSlicesM, mAxis, typesBeforeSplitting, loc, builder);
  // Slice c opds along N
  int32_t numSlicesN = totalN / prefetchWidthN;
  for (int32_t mIdx = 0; mIdx < numSlicesM; ++mIdx) {
    int32_t mOff = mIdx * prefetchWidthM;
    SmallVector<Value> mnSlices = splitValueAlongAxis(
        mSlices[mIdx], numSlicesN, nAxis, typesBeforeSplitting, loc, builder);
    for (int32_t nIdx = 0; nIdx < numSlicesN; ++nIdx) {
      int32_t nOff = nIdx * prefetchWidthN;
      mnToDot[{mOff, nOff}] = mnSlices[nIdx];
    }
  }

  // Generate dots[m, n, k] and local_loads[m, n, k] (except for local_load[0,
  // 0, 0] which is prefetched) Insertion point is manipulated to ensure
  // ordering of local_load[x+1] before dot[x]
  Operation *lastDotOp = nullptr;
  for (int32_t kOff = 0; kOff < totalK; kOff += prefetchWidthK) {
    // Store local loads, since one loaded opd is reused for multiple dots.
    DenseMap<int32_t, Value> aSlices;
    DenseMap<int32_t, Value> bSlices;
    for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
      for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
        if (lastDotOp)
          builder.setInsertionPoint(lastDotOp);

        Value aSlice; // used for dot creation
        if (kOff == 0 && mOff == 0) {
          // Opd was prefetched in prior kernel loop iter.
          Value a = operand2headPrefetch[dotInterface.getA()];
          aSlice = newForOp.getTiedLoopRegionIterArg(&*a.use_begin());
          aSlices[mOff] = aSlice;
        } else {
          if (nOff == 0) {
            // Create new load for kOff>0 and nOff=0.
            FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
                dot2aVals[dotOp].back().getDefiningOp(), false, builder,
                &mapping);
            aSlice = generateLocalLoad(
                mapping.lookup(dot2aLoopArg[dotOp]), 0, false, dotEncoding,
                builder,
                failed(awtA) ? std::nullopt : std::optional<Value>(*awtA), mOff,
                prefetchWidthM, std::nullopt, std::nullopt, kOff,
                prefetchWidthK);
            ttg::clonePrefetchElementwiseOps(aSlice, dot2aVals[dotOp], builder);
            aSlices[mOff] = aSlice;
          } else {
            // Reuse the opd previously created during nOff=0 for nOff>0.
            aSlice = aSlices[mOff];
          }
        }

        Value bSlice; // used for dot creation
        if (kOff == 0 && nOff == 0) {
          // Opd was prefetched in prior iter.
          Value b = operand2headPrefetch[dotInterface.getB()];
          bSlice = newForOp.getTiedLoopRegionIterArg(&*b.use_begin());
          bSlices[nOff] = bSlice;
        } else {
          if (mOff == 0) {
            // Create new local load for kOff>0 and mOff=0.
            FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
                dot2bVals[dotOp].back().getDefiningOp(), false, builder,
                &mapping);
            bSlice = generateLocalLoad(
                mapping.lookup(dot2bLoopArg[dotOp]), 1, false, dotEncoding,
                builder,
                failed(awtB) ? std::nullopt : std::optional<Value>(*awtB),
                std::nullopt, std::nullopt, nOff, prefetchWidthN, kOff,
                prefetchWidthK);
            ttg::clonePrefetchElementwiseOps(bSlice, dot2bVals[dotOp], builder);
            bSlices[nOff] = bSlice;
          } else {
            // Reuse the opd previously created during mOff=0 for mOff>0.
            bSlice = bSlices[nOff];
          }
        }

        // Sched.barrier after local_loads and before dot to enforce
        // prefetching; 0th dot skipped b/c prefetched.
        if (mOff > 0 || nOff > 0 || kOff > 0)
          insertSchedBarrier(builder, loc);

        if (lastDotOp)
          builder.setInsertionPointAfter(lastDotOp);
        Value cSlice = mnToDot[{mOff, nOff}];
        auto dType = cast<RankedTensorType>(cSlice.getType());
        Operation *newDot = createDotOp(dotOp, builder, loc, dType, aSlice,
                                        bSlice, cSlice, &mapping);
        mnToDot[{mOff, nOff}] = newDot->getResult(0);
        lastDotOp = newDot;
      }
    }
  }

  // Concatenate all M×N tiles back into a single tensor with original shape
  // Join d opds along N
  SmallVector<Value> mJoins;
  for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
    SmallVector<Value> mnSlices;
    for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
      mnSlices.push_back(mnToDot[{mOff, nOff}]);
    }
    Value mJoin = joinValuesAlongAxis(mnSlices, nAxis, typesBeforeSplitting,
                                      loc, builder);
    mJoins.push_back(mJoin);
  }
  // Join d opds along M
  Value result =
      joinValuesAlongAxis(mJoins, mAxis, typesBeforeSplitting, loc, builder);
  Operation *newOp = result.getDefiningOp();
  // Reset insertion point to before the last dot for the prefetched local loads
  builder.setInsertionPoint(lastDotOp);
  return newOp;
}

// Generates the prefetched local loads which are for dot[m=0,n=0,k=0]
void Prefetcher::generatePrefetchingLocalLoads(
    Operation *dotOp, OpBuilder &builder, IRMapping &mapping,
    SmallVector<Value> &yieldValues) {
  Attribute dotEncoding =
      cast<RankedTensorType>(dotOp->getResult(0).getType()).getEncoding();
  // Get async wait tokens from async_wait at end of prior iteration.
  FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
      dot2aVals[dotOp].back().getDefiningOp(), true, builder, &mapping);
  FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
      dot2bVals[dotOp].back().getDefiningOp(), true, builder, &mapping);
  Value aToYield = generateLocalLoad(
      mapping.lookup(dot2aYield[dotOp]), 0, true, dotEncoding, builder,
      failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
  ttg::clonePrefetchElementwiseOps(aToYield, dot2aVals[dotOp], builder);
  yieldValues.push_back(aToYield);
  Value bToYield = generateLocalLoad(
      mapping.lookup(dot2bYield[dotOp]), 1, true, dotEncoding, builder,
      failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
  ttg::clonePrefetchElementwiseOps(bToYield, dot2bVals[dotOp], builder);
  yieldValues.push_back(bToYield);
  // Sched.barrier after local_loads (and before dot) to enforce prefetching.
  insertSchedBarrier(builder, dotOp->getLoc());
}

// Get async wait token (awt), if any, for new LocalLoad in newForOp
// based on old LocalLoad; args determine 3 cases where to
// get/create awt.
//
// Args
// - fromPriorIter, used for prefetching slice[0], means track the awt
//   through block args, yield and find it in the previous loop iteration.
// - mapping maps original forOp to newForOp, and is not used with
//   not in for loop, e.g. for emitPrologue.
//
// Case 0 - Prologue. awt is loop arg; returns init value before loop.
//  - fromPriorIter=false
//  - mapping=nullptr
// Case 1 - Slice[1,N-1]. awt is loop arg; returns same arg but mapped to
// newForLoop.
//  - fromPriorIter=false
//  - mapping=valid
// Case 2 - Slice[0] prefetched. awt comes from end of prior loop iteration.
//  - fromPriorIter=true
//  - mapping=valid
//
//  NOTE: fromPriorIter=true & mapping=nullptr is invalid combination.
FailureOr<Value> Prefetcher::getAsyncWaitTokenForLocalLoad(
    Operation *localLoadOp, bool fromPriorIter, OpBuilder &builder,
    IRMapping *mapping) {
  auto llOp = dyn_cast<ttg::LocalLoadOp>(localLoadOp);
  if (!llOp)
    return failure();
  if (llOp->getNumOperands() != 2)
    return failure();
  Value awt = llOp->getOperand(1);
  if (!isa<mlir::gpu::AsyncTokenType>(awt.getType()))
    return failure();

  if (!fromPriorIter) {
    if (!mapping) {
      // Case 0: return async wait token in prologue.
      if (mlir::BlockArgument loopArg = dyn_cast<mlir::BlockArgument>(awt)) {
        unsigned argIdx = loopArg.getArgNumber() - forOp.getNumInductionVars();
        Value initAwt = forOp.getInitArgs()[argIdx];
        return initAwt;
      } else {
        assert(false && "Expected async wait token to be loop arg.");
        return failure();
      }
    } else {
      // Case 1: return new async wait token from for(args) for
      // LocalLoad[1, N-1].
      return mapping->lookup(awt);
    }
  }
  assert(mapping);
  assert(fromPriorIter);

  mlir::BlockArgument loopArg = dyn_cast<mlir::BlockArgument>(awt);
  if (!loopArg) {
    assert(false && "fromPriorIter specified but awt isn't a loop arg.");
    return failure();
  }

  // Case 2: return new async wait token from end of prior iteration,
  // this occurs for the prefetching LocalLoads at the end of the loop;
  // which may or may not have been created yet i.e. is in mapping.
  // Note: awt may already be in mapping for two reasons,
  // (a) it is a duplicate of async_wait created below,
  // (b) associated async_wait was already created previously in new loop
  // even though want prior iter of it. Now we want to wrap around the loop
  // body and find this token in the previous iteration because it was
  // prefetched.
  unsigned argIdx = loopArg.getArgNumber() - forOp.getNumInductionVars();
  Value initAwt = forOp.getInitArgs()[argIdx];
  Value yieldedAwt = yieldOp.getOperand(argIdx);
  if (mapping->contains(yieldedAwt))
    return mapping->lookup(yieldedAwt);

  // Want awt fromPriorIter, but it isn't in map yet because the async_wait op
  // hasn't been visited yet, so create and place in mapping.
  LDBG("Case 2 yieldedAwt not yet in map");
  auto awOp = yieldedAwt.getDefiningOp();
  // Create new async_wait op in new loop
  Operation *newAwOp = builder.clone(*awOp, *mapping);
  for (unsigned dstIdx : llvm::seq(unsigned(0), awOp->getNumResults()))
    mapping->map(awOp->getResult(dstIdx), newAwOp->getResult(dstIdx));
  return newAwOp->getResult(0);
}

} // anonymous namespace
} // namespace amdgpu
} // namespace triton

static bool hasPrecedingCondBarrier(scf::ForOp forOp) {
  for (Operation *op = forOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<triton::amdgpu::CondBarrierOp>(op))
      return true;
  return false;
}

struct TritonAMDGPULdsPrefetchPass
    : public impl::TritonAMDGPULdsPrefetchBase<TritonAMDGPULdsPrefetchPass> {
  using Base::Base;

  void runOnOperation() override {
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                      &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      // Don't run LdsPrefetch on loops preceded by a conditional barrier,
      // as this signifies that PingPong scheduling succeeded.
      if (hasPrecedingCondBarrier(forOp)) {
        LDBG("Skipping loop (CondBarrierOp signifies PingPong).");
        return;
      }

      triton::amdgpu::Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed()) {
        LDBG("Prefetching failed.");
        return;
      }

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
      LDBG("Prefetching succeeded.");
    });
  }
};

} // namespace mlir
