#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

// Include shared C-compatible TDM utilities for warp distribution
#include "../../backend/include/TDMCommon.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
// Validates that the tensor descriptor's strides and shared layout are
// compatible with TDM. Requirements:
//  - The shared order must be [rank-1, rank-2, ..., 0].
//  - All stride-1 dimensions must be consecutive trailing dims.
// Additionally, a single stride-1 dimension may appear at the rank-2
// position (col-major) if the shared order has rank-2 and rank-1 swapped.
LogicalResult validateStridesAndSharedOrder(triton::MakeTensorDescOp op,
                                            Attribute sharedEnc,
                                            ArrayRef<int64_t> shape,
                                            ValueRange strides) {
  int rank = shape.size();
  auto sharedOrder = triton::gpu::getOrder(
      cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shape);

  SmallVector<unsigned> strideOneDims;
  for (auto [dim, strideVal] : llvm::enumerate(strides)) {
    if (getConstantIntValue(getAsOpFoldResult(strideVal)).value_or(0) == 1)
      strideOneDims.push_back(dim);
  }

  if (strideOneDims.empty())
    return op.emitError() << "requires at least one dimension to have stride 1";

  // If the only stride-1 dim is the second-to-last dimension (col-major) we can
  // safely reorder the dimensions during lowering.
  bool isColMajor =
      strideOneDims.size() == 1 && strideOneDims.front() == rank - 2;

  SmallVector<unsigned> expectedOrder(llvm::reverse(llvm::seq<unsigned>(rank)));
  if (isColMajor)
    std::swap(expectedOrder[0], expectedOrder[1]);

  if (sharedOrder != ArrayRef(expectedOrder)) {
    if (isColMajor)
      return op.emitError()
             << "requires shared order [rank-2, rank-1, rank-3, "
                "rank-4, ..., 0] because dim[rank-2] has stride 1";
    return op.emitError() << "requires shared order [rank-1, rank-2, ..., 0]";
  }

  if (strideOneDims.size() > 1) {
    unsigned k = strideOneDims.size();
    unsigned numStride1Dims = strideOneDims.size();
    for (unsigned i = 0; i < numStride1Dims; ++i) {
      if (strideOneDims[i] != rank - numStride1Dims + i)
        return op.emitError() << "requires all stride 1 dimensions to be "
                                 "consecutive starting from the last dimension";
    }
  }

  return success();
}

// Collects all users of the value beyond the basic block boundaries
// defining a given value.
void collectUsers(Value value, llvm::SetVector<Operation *> &users) {
  for (OpOperand &use : value.getUses()) {
    Operation *userOp = use.getOwner();
    if (users.contains(userOp)) {
      // stop recursion; avoid loops
      return;
    }
    users.insert(userOp);
    const unsigned argIdx = use.getOperandNumber();

    if (auto unrealCast = dyn_cast<mlir::UnrealizedConversionCastOp>(userOp)) {
      collectUsers(unrealCast->getResult(argIdx), users);
    }

    if (auto branch = dyn_cast<mlir::BranchOpInterface>(userOp)) {
      auto successors = branch->getSuccessors();
      for (auto [idx, successor] : llvm::enumerate(successors)) {
        auto operands = branch.getSuccessorOperands(idx);
        if (argIdx < operands.size()) {
          collectUsers(successor->getArgument(argIdx), users);
        }
      }
    }
  }
}

Attribute findEncodingFromUsers(Operation *op) {
  llvm::SetVector<Operation *> users;
  for (auto result : op->getResults())
    collectUsers(result, users);

  Attribute sharedEnc;
  for (auto use : users) {
    Attribute userEnc;
    if (auto load = llvm::dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(use)) {
      userEnc = load.getResult().getType().getEncoding();
    } else if (auto store =
                   llvm::dyn_cast<amdgpu::AsyncTDMCopyLocalToGlobalOp>(use)) {
      userEnc = store.getSrc().getType().getEncoding();
    }
    if (!userEnc)
      continue;

    // Assign first encoding found; or error out if different encoding is found
    if (!sharedEnc)
      sharedEnc = userEnc;
    else if (sharedEnc != userEnc) {
      op->emitError("Descriptor is used with different shared encodings.");
      return {};
    }
  }
  if (!sharedEnc)
    op->emitError("Encoding hasn't been found from users.");
  return sharedEnc;
}

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  const TargetInfo &targetInfo;

  MakeTensorDescOpConversion(LLVMTypeConverter &converter,
                             const TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = llvm::to_vector(adaptor.getShape());
    auto tensorStride = llvm::to_vector(adaptor.getStrides());
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto blockTy = tensorDescTy.getBlockType();
    auto sharedEnc = blockTy.getEncoding();
    if (!sharedEnc) {
      // TODO: add an extra pass to assign layout to descriptors
      sharedEnc = findEncodingFromUsers(op);
      if (!sharedEnc)
        return rewriter.notifyMatchFailure(op, "Descriptor has no layout.");
    }
    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (auto padEnc = getPaddedEncoding(sharedEnc)) {
      if (padEnc.getIntervals().size() != 1 || padEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = padEnc.getIntervals()[0];
      padAmount = padEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(blockTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(blockTy.getShape());
    int numWarps = lookupNumWarps(op);
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);

    if (failed(validateStridesAndSharedOrder(op, sharedEnc, shapePerCTA,
                                             tensorStride))) {
      return failure();
    }
    auto sharedOrder = triton::gpu::getOrder(
        cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shapePerCTA);
    bool isRowMajor = sharedOrder[0] == (sharedOrder.size() - 1);

    // Create TDM descriptor for 2D-5D tensors
    auto tdmDesc = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, shapePerCTA, numWarps,
        padInterval, padAmount, tensorShape, tensorStride, basePtr, isRowMajor);

    // Apply per-warp offsets to global_addr and tensor_dim at descriptor
    // creation time.
    {
      auto ctx = rewriter.getContext();
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      size_t numDims = blockShape.size();

      auto hwShapePerCTA = shapePerCTA;
      if (!isRowMajor)
        LLVM::AMD::swapTrailingDims(hwShapePerCTA);

      int warpsArr[5];
      tdmGetWarpDistribution(hwShapePerCTA.data(), numDims, numWarps,
                             warpsArr);
      SmallVector<unsigned> warpsPerCTA(warpsArr, warpsArr + numDims);

      auto smemSpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
      auto memDescTy = triton::gpu::MemDescType::get(
          blockShape, blockTy.getElementType(), sharedEnc,
          /*memorySpace=*/smemSpace, /*mutableMemory=*/true);
      triton::LinearLayout sharedLayout =
          triton::gpu::isPaddedEncoding(sharedEnc)
              ? triton::gpu::paddedLinearLayout(memDescTy)
              : triton::gpu::toLinearLayout(memDescTy);

      triton::LinearLayout tdmViewSharedLayout = sharedLayout;
      if (!isRowMajor && numDims >= 2) {
        auto dimN_2 =
            StringAttr::get(ctx, "dim" + std::to_string(numDims - 2));
        auto dimN_1 =
            StringAttr::get(ctx, "dim" + std::to_string(numDims - 1));
        tdmViewSharedLayout =
            LLVM::AMD::swapOutDimSemantics(sharedLayout, dimN_2, dimN_1);
      }

      auto cgaLayout =
          triton::gpu::SharedLinearEncodingAttr::get(
              ctx, tdmViewSharedLayout, /*layoutAlignment=*/16)
              .getCGALayout()
              .getLinearLayout();

      auto tdmLayout = triton::gpu::getTDMLinearLayout(hwShapePerCTA,
                                                        warpsPerCTA, cgaLayout);

      auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
      auto ctaId = targetInfo.getClusterCTAId(rewriter, loc);

      auto kMessage = StringAttr::get(ctx, "message");
      auto kWarp = StringAttr::get(ctx, "warp");
      auto kBlock = StringAttr::get(ctx, "block");

      auto warpOffset = applyLinearLayout(
          loc, rewriter, tdmLayout,
          {{kMessage, b.i32_val(0)}, {kWarp, warpId}, {kBlock, ctaId}});

      // Compute global address offset from warp distribution
      SmallVector<Value> hwStride(numDims);
      for (size_t i = 0; i < numDims; ++i)
        hwStride[i] = b.trunc(i32_ty, tensorStride[i]);
      if (!isRowMajor)
        LLVM::AMD::swapTrailingDims(hwStride);

      Value baseOffset = b.i32_val(0);
      for (size_t i = 0; i < numDims; ++i) {
        Value dimOffset = b.mul(warpOffset[i].second, hwStride[i]);
        baseOffset = b.add(baseOffset, dimOffset);
      }

      // Advance global_addr in group0[2:3]
      Value curAddrLo = LLVM::AMD::vecGet(b, tdmDesc.group0, 2);
      Value curAddrHi =
          b.and_(LLVM::AMD::vecGet(b, tdmDesc.group0, 3),
                 b.i32_val(0x7FFFFFFF));
      Value curAddr =
          b.or_(b.zext(i64_ty, curAddrLo),
                b.shl(b.zext(i64_ty, curAddrHi), b.i64_val(32)));
      auto elementBitWidth = elementType.getIntOrFloatBitWidth();
      Value byteOffset =
          b.mul(b.sext(i64_ty, baseOffset), b.i64_val(elementBitWidth / 8));
      Value newAddr = b.add(curAddr, byteOffset);
      tdmDesc.group0 = LLVM::AMD::vecSet(
          b, tdmDesc.group0, 2, b.trunc(i32_ty, newAddr));
      tdmDesc.group0 = LLVM::AMD::vecSet(
          b, tdmDesc.group0, 3,
          b.or_(b.trunc(i32_ty, b.lshr(newAddr, b.i64_val(32))),
                b.i32_val(1 << 31)));

      // Adjust tensor_dim by warp offset
      Value v16 = b.i32_val(16);
      {
        Value warpOff = warpOffset[numDims - 1].second;
        Value dimLo = b.lshr(LLVM::AMD::vecGet(b, tdmDesc.group1, 1), v16);
        Value dimHi = b.shl(
            b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 2),
                   b.i32_val(0xFFFF)),
            v16);
        Value dim = b.or_(dimLo, dimHi);
        Value newDim = b.smax(b.i32_val(0), b.sub(dim, warpOff));
        tdmDesc.group1 = LLVM::AMD::vecSet(
            b, tdmDesc.group1, 1,
            b.or_(b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 1),
                         b.i32_val(0xFFFF)),
                  b.shl(newDim, v16)));
        tdmDesc.group1 = LLVM::AMD::vecSet(
            b, tdmDesc.group1, 2,
            b.or_(b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 2),
                         b.i32_val(0xFFFF0000)),
                  b.and_(b.lshr(newDim, v16), b.i32_val(0xFFFF))));
      }
      if (numDims >= 2) {
        Value warpOff = warpOffset[numDims - 2].second;
        Value dimLo = b.lshr(LLVM::AMD::vecGet(b, tdmDesc.group1, 2), v16);
        Value dimHi = b.shl(
            b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 3),
                   b.i32_val(0xFFFF)),
            v16);
        Value dim = b.or_(dimLo, dimHi);
        Value newDim = b.smax(b.i32_val(0), b.sub(dim, warpOff));
        tdmDesc.group1 = LLVM::AMD::vecSet(
            b, tdmDesc.group1, 2,
            b.or_(b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 2),
                         b.i32_val(0xFFFF)),
                  b.shl(newDim, v16)));
        tdmDesc.group1 = LLVM::AMD::vecSet(
            b, tdmDesc.group1, 3,
            b.or_(b.and_(LLVM::AMD::vecGet(b, tdmDesc.group1, 3),
                         b.i32_val(0xFFFF0000)),
                  b.and_(b.lshr(newDim, v16), b.i32_val(0xFFFF))));
      }
    }

    SmallVector<Value> groups = tdmDesc.getAllGroups();

    auto desc =
        packLLElements(loc, getTypeConverter(), groups, rewriter, tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, targetInfo, benefit);
  // NOTE: AdvanceTDMDescOpConversion is registered in
  // populateLoadStoreOpToLLVMPatterns alongside the other TDM ops.
  return;
}
