//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
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
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

  // Helper function to split a value along a specific axis into numSlices pieces
// SplitOp only splits along the last dimension, so we need to transpose, reshape, split, reshape, and transpose back
// This function recursively splits in half until numSlices pieces are created
static SmallVector<Value> splitValueAlongAxis(Value input, int32_t numSlices, int axis, Location loc, OpBuilder &builder) {
  LDBG("splitValueAlongAxis(): n=" << numSlices << ", a=" << axis << ", op=" << input);
  // Base case: if numSlices is 1, return the input as-is
  if (numSlices == 1) {
    return {input};
  }
  
  // Lambda to perform a single binary split
  auto splitOnce = [&](Value val) -> std::pair<Value, Value> {
    LDBG("\n\n\nsplitOnce() axis=" << axis);
    LDBG("val: " << val);
    RankedTensorType inputType = cast<RankedTensorType>(val.getType());
    LDBG("inputType: " << inputType);

    auto inputLL = toLinearLayout(inputType);
    LDBG("inputLL: " << inputLL);

    auto shape = inputType.getShape();
    int rank = shape.size();
    
    // Reshape to split the target axis into [N/2, 2]
    // TODO(dtanner) Fred says this may be the most difficult step, so what I have may be too simple.
    SmallVector<int64_t> newShape;
    int splitDimPos = axis;  // Position of the "2" dimension after reshape
    for (int i = 0; i < rank; ++i) {
      if (i == axis) {
        //if (axis == 1) {
          newShape.push_back(2);
          newShape.push_back(shape[i] / 2);
          //splitDimPos = i;
        //} else {
        //  splitDimPos = i;
        //  newShape.push_back(2);
        //  newShape.push_back(shape[i] / 2);
        //}
      } else {
        newShape.push_back(shape[i]);
      }
    }
    LDBG("newShape: " << newShape[0] << ", " << newShape[1] << ", " << newShape[2]);
    LDBG("splitDimPos: " << splitDimPos);

    // Use ReshapeOp builder that infers encoding automatically
    // When rank changes, some encodings (like AMDMfmaEncodingAttr) may not support it,
    // so we need to handle encoding inference carefully
    Value reshaped = triton::ReshapeOp::create(builder, loc, newShape, val);
    LDBG("reshaped: " << reshaped);
    auto reshapedLL = toLinearLayout(cast<RankedTensorType>(reshaped.getType()));
    LDBG("reshapedLL: " << reshapedLL);
    
    // Permute to move the "2" dimension to last position if needed
    Value transposed = reshaped;
    int newRank = rank + 1;  // After reshape, we have one more dimension
    if (splitDimPos != newRank - 1) {
      LDBG("transposing");
      SmallVector<int32_t> trans;
      for (int i = 0; i < newRank; ++i) {
        if (i != splitDimPos) trans.push_back(i);
      }
      trans.push_back(splitDimPos);  // Move "2" dimension to end
      transposed = triton::TransOp::create(builder, loc, reshaped, trans);
    }
    LDBG("transposed: " << transposed);
    auto transposedLL = toLinearLayout(cast<RankedTensorType>(transposed.getType()));
    LDBG("transposedLL: " << transposedLL);

    // Split along the last dimension
    auto split = triton::SplitOp::create(builder, loc, transposed);
    Value left = split.getResult(0);
    Value right = split.getResult(1);
    LDBG("left: " << left);
    // LDBG("right: " << right);
    auto leftLL = toLinearLayout(cast<RankedTensorType>(left.getType()));
    LDBG("leftLL: " << leftLL);
    return {left, right};
  };
  
  // Iteratively split in half until we have numSlices pieces
  SmallVector<Value> tiles;
  tiles.push_back(input);
  
  int32_t currentCount = 1;
  while (currentCount < numSlices) {
    LDBG("while " << currentCount << " < " << numSlices);
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

// Helper function to join tensors along a specific axis (inverse of splitAlongAxis)
// JoinOp only joins along the last dimension, so we need to transpose, join, reshape, and transpose back
static Value joinValuesAlongAxis(SmallVector<Value> tiles, int axis, Location loc, OpBuilder &builder) {
  LDBG("joinValuesAlongAxis(): n=" << tiles.size() << ", a=" << axis);

  if (tiles.size() == 1) {
    return tiles[0];
  }
  
  // Lambda to perform a single binary join
  auto joinOnce = [&](Value left, Value right) -> Value {
    LDBG("joinOnce");
    auto leftType = cast<RankedTensorType>(left.getType());
    auto shape = leftType.getShape();
    int rank = shape.size();
    
    // Transpose to move target axis to last position if needed
    Value leftToJoin = left;
    Value rightToJoin = right;
    if (axis != rank - 1) {
      SmallVector<int32_t> perm;
      for (int j = 0; j < rank; ++j) {
        if (j != axis) perm.push_back(j);
      }
      perm.push_back(axis);  // Move target axis to end
      leftToJoin = triton::TransOp::create(builder, loc, left, perm);
      rightToJoin = triton::TransOp::create(builder, loc, right, perm);
    }
    
    // Join creates a new trailing dimension of size 2
    auto joined = triton::JoinOp::create(builder, loc, leftToJoin, rightToJoin);
    
    // Reshape to merge the trailing dimension: [..., N, 2] -> [..., 2*N]
    auto joinedType = cast<RankedTensorType>(joined.getType());
    auto joinedShape = joinedType.getShape();
    SmallVector<int64_t> newShape(joinedShape.begin(), joinedShape.end() - 2);
    newShape.push_back(joinedShape[joinedShape.size() - 2] * 2);
    auto reshapeType = RankedTensorType::get(newShape, joinedType.getElementType(), joinedType.getEncoding());
    Value reshaped = triton::ReshapeOp::create(builder, loc, reshapeType, joined);
    
    // Transpose back if we transposed earlier
    Value result = reshaped;
    if (axis != rank - 1) {
      SmallVector<int32_t> invTrans(rank);
      for (int j = 0; j < rank - 1; ++j) {
        invTrans[j < axis ? j : j + 1] = j;
      }
      invTrans[axis] = rank - 1;
      result = triton::TransOp::create(builder, loc, reshaped, invTrans);
    }
    
    return result;
  };
  
  // Iteratively join pairs using log2 iterations
  while (tiles.size() > 1) {
    SmallVector<Value> nextTiles;
    for (size_t i = 0; i < tiles.size(); i += 2) {
      Value joined = joinOnce(tiles[i], tiles[i + 1]);
      nextTiles.push_back(joined);
    }
    tiles = std::move(nextTiles);
  }
  
  return tiles[0];
}

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidthK
  unsigned prefetchWidthM = 64;
  unsigned prefetchWidthN = 64;
  unsigned prefetchWidthK = 32;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  FailureOr<Value> getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                 bool fromPriorIter,
                                                 OpBuilder &builder,
                                                 IRMapping *mapping = nullptr);

  Value generateLocalLoadSlice(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<Value> asyncWaitToken = std::nullopt,
                         std::optional<int64_t> offsetM = std::nullopt,
                         std::optional<int64_t> shapeM = std::nullopt,
                         std::optional<int64_t> offsetN = std::nullopt,
                         std::optional<int64_t> shapeN = std::nullopt,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

  Operation *generateDotsAndNonPrefetchingLocalLoads(triton::DotOp dot,
                                        Attribute dotEncoding,
                                        OpBuilder &builder,
                                        IRMapping &mapping,
                                        scf::ForOp newForOp);

  void generatePrefetchingLocalLoads(triton::DotOp dot, OpBuilder &builder,
                              IRMapping &mapping,
                              SmallVector<Value> &yieldValues);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

// Generates all dots and first N-1 local_loads.
Operation *Prefetcher::generateDotsAndNonPrefetchingLocalLoads(triton::DotOp dot,
                                                  Attribute dotEncoding,
                                                  OpBuilder &builder,
                                                  IRMapping &mapping,
                                                  scf::ForOp newForOp) {
  // Get total dimensions from operands
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  int64_t totalM = aType.getShape()[0];
  int64_t totalK = aType.getShape().back();
  int64_t totalN = bType.getShape().back();


  // Map from (M, N) offsets to accumulated dot Values
  DenseMap<std::pair<int32_t, int32_t>, Value> mnToDot;

  // Check if we only have one slice (no remaining parts)
#if 0
  bool onlyOneSlice = (totalM <= prefetchWidthM) && 
                       (totalN <= prefetchWidthN) && 
                       (totalK <= prefetchWidthK);
  if (onlyOneSlice) {
    // There is only one dot while prefetchWidth == size so delay issuing it
    Operation *firstDot = builder.clone(*dot, mapping);
    if (Value a = operand2headPrefetch.lookup(dot.getA()))
      firstDot->setOperand(
          0, newForOp.getTiedLoopRegionIterArg(&*a.use_begin()));
    if (Value b = operand2headPrefetch.lookup(dot.getB()))
      firstDot->setOperand(
          1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));
    builder.setInsertionPoint(firstDot);
    return firstDot;
  }
  #endif

  // Assert that dimensions are evenly divisible by prefetch widths
  assert(totalM % prefetchWidthM == 0 && "totalM must be divisible by prefetchWidthM");
  assert(totalN % prefetchWidthN == 0 && "totalN must be divisible by prefetchWidthN");
  assert(totalK % prefetchWidthK == 0 && "totalK must be divisible by prefetchWidthK");

  // Split the accumulator operand (c) into M×N tiles using vendor-neutral SplitOp
  // SplitOp only splits along the last dimension, so we need to use TransOp
  // to permute dimensions. We also need to reshape to add a trailing dimension of size 2.
  Value cOperand = mapping.lookup(dot.getC());
  auto cType = cast<RankedTensorType>(cOperand.getType());
  
  // Split along N dimension (axis 1) to get rows
  int32_t numRowTiles = totalM / prefetchWidthM;
  SmallVector<Value> rowTiles = splitValueAlongAxis(cOperand, numRowTiles, 1, dot.getLoc(), builder);
  
  // Now split each row along M dimension (axis 0) to get individual tiles
  int32_t numColTiles = totalN / prefetchWidthN;
  
  for (int32_t mIdx = 0; mIdx < numRowTiles; ++mIdx) {
    int32_t mOff = mIdx * prefetchWidthM;
    
    // Split this row along N dimension (axis 0) to get individual tiles
    SmallVector<Value> colTiles = splitValueAlongAxis(rowTiles[mIdx], numColTiles, 0, dot.getLoc(), builder);
    
    // Store the column tiles in the map
    for (int32_t nIdx = 0; nIdx < numColTiles; ++nIdx) {
      int32_t nOff = nIdx * prefetchWidthN;
      mnToDot[{mOff, nOff}] = colTiles[nIdx];
    }
  }

  // Generate dots[m, n, k] and local_loads[m, n, k] for all slices
  // Triple nested loop over K, M, N dimensions (K outermost)
  Operation *lastOp = nullptr;

  for (int32_t kOff = 0; kOff < totalK; kOff += prefetchWidthK) {
    for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
      for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
        LDBG("MNK: " << mOff << ", " << nOff << ", " << kOff);
        
        Value aSlice;
        Value bSlice;
        
        // For the first K slice, use prefetched values from prologue
        if (kOff == 0) {
          // Use the prefetched operands from the loop args
          if (Value a = operand2headPrefetch.lookup(dot.getA()))
            aSlice = newForOp.getTiedLoopRegionIterArg(&*a.use_begin());
          else
            aSlice = mapping.lookup(dot.getA());
          LDBG("aSlice0: " << aSlice);
          
          if (Value b = operand2headPrefetch.lookup(dot.getB()))
            bSlice = newForOp.getTiedLoopRegionIterArg(&*b.use_begin());
          else
            bSlice = mapping.lookup(dot.getB());
          LDBG("bSlice0: " << bSlice);
        } else {
          // Generate prefetch for operand A (sliced in M and K dimensions)
          FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
              dot2aVals[dot].back().getDefiningOp(), false, builder, &mapping);
          aSlice = generateLocalLoadSlice(
              mapping.lookup(dot2aLoopArg[dot]), 0, false, dotEncoding, builder,
              failed(awtA) ? std::nullopt : std::optional<Value>(*awtA),
              mOff, prefetchWidthM, std::nullopt, std::nullopt, kOff, prefetchWidthK);
          cloneElementwiseOps(aSlice, dot2aVals[dot], builder);
          LDBG("aSlice: " << aSlice);
          
          // Generate prefetch for operand B (sliced in K and N dimensions)
          FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
              dot2bVals[dot].back().getDefiningOp(), false, builder, &mapping);
          bSlice = generateLocalLoadSlice(
              mapping.lookup(dot2bLoopArg[dot]), 1, false, dotEncoding, builder,
              failed(awtB) ? std::nullopt : std::optional<Value>(*awtB),
              std::nullopt, std::nullopt, nOff, prefetchWidthN, kOff, prefetchWidthK);
          cloneElementwiseOps(bSlice, dot2bVals[dot], builder);
          LDBG("bSlice: " << bSlice);
        }
        
        // Get the accumulator for this (M,N) tile
        Value cSlice = mnToDot[{mOff, nOff}];
        LDBG("cSlice[" << mOff << "," << nOff << "]: " << cSlice);

        // Create the dot operation
        Operation *newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aSlice);
        newOp->setOperand(1, bSlice);
        newOp->setOperand(2, cSlice);
        LDBG("newDot[" << mOff << "," << nOff << "]: " << *newOp);

        // Update the accumulator for this (M,N) tile
        mnToDot[{mOff, nOff}] = newOp->getResult(0);
        lastOp = newOp;
        
        // Delay issuing the last dot
        //bool isLastK = (kOff + prefetchWidthK == totalK);
        //bool isLastM = (mOff + prefetchWidthM == totalM);
        //bool isLastN = (nOff + prefetchWidthN == totalN);
        //if (isLastK && isLastM && isLastN) {
          // We want to delay issuing the last dot as long as possible, ideally
          // until after the prefetch.  To accomplish this, set the insertion
          // point above the dot.  If we find anything dependent on the dot (at
          // the top of this loop), we resume inserting after it.
        //}
      }
    }
  }
  builder.setInsertionPoint(lastOp);

  // Concatenate all M×N tiles back into a single tensor with original shape
  // JoinOp only joins along the last dimension (creates new trailing dim of size 2)
  // So we need to use TransOp and ReshapeOp, similar to splitting
  
  // First join tiles along N dimension (within each row)
  SmallVector<Value> rowResults;
  
  for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
    SmallVector<Value> rowTiles;
    
    // Collect all tiles in this row (along N dimension)
    for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
      rowTiles.push_back(mnToDot[{mOff, nOff}]);
    }
    
    // Join tiles in this row along N dimension (axis 1)
    Value rowResult = joinValuesAlongAxis(rowTiles, 1, dot.getLoc(), builder);
    rowResults.push_back(rowResult);
  }
  
  // Then join all rows along M dimension (axis 0)
  Value result = joinValuesAlongAxis(rowResults, 0, dot.getLoc(), builder);
  
  Operation *newOp = result.getDefiningOp();
  
  return newOp;
}

// Generates the last local_loads which are local_load[k=0]'
void Prefetcher::generatePrefetchingLocalLoads(triton::DotOp dot, OpBuilder &builder,
                                        IRMapping &mapping,
                                        SmallVector<Value> &yieldValues) {
  Attribute dotEncoding = dot.getType().getEncoding();
  // Get async wait tokens from async_wait at end of prior iteration.
  FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
      dot2aVals[dot].back().getDefiningOp(), true, builder, &mapping);
  FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
      dot2bVals[dot].back().getDefiningOp(), true, builder, &mapping);
  Value aToYield = generateLocalLoadSlice(
      mapping.lookup(dot2aYield[dot]), 0, true, dotEncoding, builder,
      failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
  cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
  yieldValues.push_back(aToYield);
  Value bToYield = generateLocalLoadSlice(
      mapping.lookup(dot2bYield[dot]), 1, true, dotEncoding, builder,
      failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
  cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
  yieldValues.push_back(bToYield);
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
FailureOr<Value> Prefetcher::getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                           bool fromPriorIter,
                                                           OpBuilder &builder,
                                                           IRMapping *mapping) {
  auto llOp = dyn_cast<triton::gpu::LocalLoadOp>(cvt);
  if (!llOp)
    return failure();
  if (llOp->getNumOperands() != 2)
    return failure();
  Value awt = llOp->getOperand(1);
  if (!isa<AsyncTokenType>(awt.getType()))
    return failure();

  if (!fromPriorIter) {
    if (!mapping) {
      // Case 0: return async wait token in prologue.
      if (mlir::BlockArgument loopArg =
              dyn_cast<mlir::BlockArgument>(awt)) {
        unsigned argIdx =
            loopArg.getArgNumber() - forOp.getNumInductionVars();
        Value initAwt = forOp.getInitArgs()[argIdx];
        return initAwt;
      } else {
        assert(false || "Expected async wait token to be loop arg.");
        return failure();
      }
      return awt;
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
    assert(
      false ||
      "fromPriorIter specified but awt isn't a loop arg.");
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

Value Prefetcher::generateLocalLoadSlice(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<Value> asyncWaitToken,
                                   std::optional<int64_t> offsetM,
                                   std::optional<int64_t> shapeM,
                                   std::optional<int64_t> offsetN,
                                   std::optional<int64_t> shapeN,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // For operand A (opIdx=0): shape is [M, K], so mIdx=0, kIdx=1
  // For operand B (opIdx=1): shape is [K, N], so kIdx=0, nIdx=1
  int64_t mIdx = 0;  // M dimension index (only for operand A)
  int64_t nIdx = 1;  // N dimension index (only for operand B)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  // Set default M dimension handling (operand A only)
  if (opIdx == 0) {
    offset[mIdx] = isPrologue ? 0 : prefetchWidthM;
    shape[mIdx] = isPrologue ? prefetchWidthM : (shape[mIdx] - prefetchWidthM);
    if (shapeM)
      shape[mIdx] = *shapeM;
    if (offsetM)
      offset[mIdx] = *offsetM;
  }

  // Set default N dimension handling (operand B only)
  if (opIdx == 1) {
    offset[nIdx] = isPrologue ? 0 : prefetchWidthN;
    shape[nIdx] = isPrologue ? prefetchWidthN : (shape[nIdx] - prefetchWidthN);
    if (shapeN)
      shape[nIdx] = *shapeN;
    if (offsetN)
      offset[nIdx] = *offsetN;
  }

  // Set K dimension handling (both operands)
  offset[kIdx] = isPrologue ? 0 : prefetchWidthK;
  shape[kIdx] = isPrologue ? prefetchWidthK : (shape[kIdx] - prefetchWidthK);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = triton::gpu::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offset);

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, prefetchWidthK / 8);
  Value prefetchSlice;
  if (asyncWaitToken) {
    prefetchSlice = triton::gpu::LocalLoadOp::create(
        builder, v.getLoc(),
        RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem,
        *asyncWaitToken);
  } else {
    prefetchSlice = triton::gpu::LocalLoadOp::create(
        builder, v.getLoc(),
        RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem);
  }

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as Nvidia MMA v2 or AMD MFMA
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstMfmaEnc =
          dyn_cast<AMDMfmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstWmmaEnc =
          dyn_cast<AMDWmmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2) &&
          !dstWmmaEnc)
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOperand = [this](Value v) -> Value {
    auto arg = mlir::cast<BlockArgument>(v);
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aType = dot.getA().getType();
    auto bType = dot.getB().getType();
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    // Get sizes for all three dimensions
    auto mSize = aType.getShape()[0];  // M dimension from operand A
    auto nSize = bType.getShape().back();  // N dimension from operand B
    auto kSize = aType.getShape().back();  // K dimension
    LDBG("mSize: " << mSize);
    LDBG("nSize: " << nSize);
    LDBG("kSize: " << kSize);

    // Get the dot result encoding to determine instruction dimensions
    Attribute dotEncoding = dot.getType().getEncoding();
    unsigned instrM = 16, instrN = 16, instrK = 16;  // Default values
    if (auto mfmaEnc = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding)) {
      auto instrShape = mfmaEnc.getInstrShape();
      instrM = instrShape[0];
      instrN = instrShape[1];
      instrK = instrShape[2];
    } else if (auto wmmaEnc = dyn_cast<AMDWmmaEncodingAttr>(dotEncoding)) {
      auto instrShape = wmmaEnc.getInstrShape();
      instrM = instrShape[0];
      instrN = instrShape[1];
      instrK = instrShape[2];
    } else if (auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotEncoding)) {
      // For NVIDIA MMA, instruction shape depends on version
      // MMAv2: typically 16x8 or similar
      auto instrShape = mmaEnc.getInstrShape();
      instrM = instrShape[0];
      instrN = instrShape[1];
      // K dimension for MMA is determined by kWidth
      instrK = aKWidth > 0 ? aKWidth : 16;
    }
    LDBG("instrM: " << instrM);
    LDBG("instrN: " << instrN);
    LDBG("instrK: " << instrK);

    //SmallVector<int64_t> minShape =
    //    getMinShapePerSemanticTile(
    //    cast<RankedTensorType>(dot.getResult().getType()));
    //LDBG("minShape[0]: " << minShape[0]);
    //LDBG("minShape[1]: " << minShape[1]);

    // Calculate prefetch widths
    unsigned elementWidthA = aType.getElementTypeBitWidth();
    unsigned elementWidthB = bType.getElementTypeBitWidth();
    
    // K dimension width: Use 8x instruction K width for better tensor core utilization
    if (aKWidth == 0)
      prefetchWidthK = 256 / elementWidthA;
    else
      prefetchWidthK = 8 * aKWidth;

    // Skip prefetching if K dimension is less than prefetch width
    if (kSize < prefetchWidthK)
      continue;

    // M dimension width: Use at least 2x instruction M for 2x2 assembly
    // Also ensure we don't exceed 256 bits or the actual dimension size
    // int64_t targetPrefetchM = 2 * instrM;  // 2x for 2x2 tiling
    prefetchWidthM = mSize / 2; // std::max(mSize, targetPrefetchM);
    
    // N dimension width: Use at least 2x instruction N for 2x2 assembly
    // Also ensure we don't exceed 256 bits or the actual dimension size
    // int64_t targetPrefetchN = 2 * instrN;  // 2x for 2x2 tiling
    prefetchWidthN = nSize / 2; // std::max(nSize, targetPrefetchN);
    LDBG("prefetchWidthM: " << prefetchWidthM);
    LDBG("prefetchWidthN: " << prefetchWidthN);
    LDBG("prefetchWidthK: " << prefetchWidthK);

    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      LDBG("aHeaderDef: " << aHeaderDef);
      LDBG("bHeaderDef: " << bHeaderDef);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aVals[dot] = aVals;
        dot2bVals[dot] = bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = getYieldOperand(aSmem);
        dot2bYield[dot] = getYieldOperand(bSmem);
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
        dot2aVals[dot].back().getDefiningOp(), false, builder);
    FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
        dot2bVals[dot].back().getDefiningOp(), false, builder);
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aPrefetched = generateLocalLoadSlice(
        dot2aHeaderDef[dot], 0, true, dotEncoding, builder,
        failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched = generateLocalLoadSlice(
        dot2bHeaderDef[dot], 1, true, dotEncoding, builder,
        failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
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
        auto dot = dyn_cast<triton::DotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();
      newOp = generateDotsAndNonPrefetchingLocalLoads(dot, dotEncoding, builder, mapping,
                                         newForOp);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    generatePrefetchingLocalLoads(dot, builder, mapping, yieldValues);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {
    LDBG("PrefetchPass");
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
      LDBG("PrefetchPass - Succeeded");
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
