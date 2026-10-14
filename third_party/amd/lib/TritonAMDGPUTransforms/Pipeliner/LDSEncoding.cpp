#include "LDSEncoding.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

namespace mlir {

// On targets without direct-to-LDS scatter support, a direct-to-LDS copy
// requires enough data per CTA for each lane to write at least 32 bits into
// LDS. The source layout itself may still be suboptimal here; CoalesceAsyncCopy
// can rewrite it before the final direct-to-LDS lowering checks coalescing
// legality.
static bool
hasEnoughCTABytesForDirectToLds(tt::LoadOp loadOp,
                                const tt::AMD::TargetInfo &targetInfo) {
  if (targetInfo.supportsDirectToLdsScatter())
    return true;

  auto srcTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!srcTy)
    return true;

  int64_t ctaTileBits = mlir::product(ttg::getShapePerCTA(srcTy)) *
                        getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
  return ctaTileBits >= 32 * targetInfo.getWarpSize();
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::AMD::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads.
  if (!triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis))
    return false;

  using tt::amdgpu::ISAFamily;
  bool hasDirectToLdsPath =
      llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4, ISAFamily::CDNA5},
                         targetInfo.getISAFamily());

  // On targets without scatter support, reject CTA tiles that are too small to
  // provide one 32-bit direct-to-LDS write per lane. Layout-specific coalescing
  // is checked later, after CoalesceAsyncCopy has a chance to rewrite the src.
  if (hasDirectToLdsPath &&
      !hasEnoughCTABytesForDirectToLds(loadOp, targetInfo))
    return false;

  if (sharedEnc && hasDirectToLdsPath) {
    // Compute the final vecSize we can use for the combination of
    // sourceEncoding and sharedEncoding. We can only use AsyncCopy if the
    // target supports the requested or a smaller vecSize because we cannot
    // stride when loading directly to lds on GFX9
    auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
    auto regLayout = triton::gpu::toLinearLayout(srcTy);
    // It's the allocation so we trim the multibuffer dimension
    auto srcShape = srcTy.getShape();
    triton::LinearLayout sharedLayout;
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(srcShape, sharedEnc);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

    unsigned elemBitWidth = tt::getPointeeBitWidth(srcTy);
    unsigned vecSize = regToSharedLayout.getNumConsecutiveInOut();
    if (paddedEnc)
      vecSize = std::min(vecSize, paddedEnc.getMinInterval());

    if (fitToValidDirectToLdsVecSize(vecSize, elemBitWidth, targetInfo) == 0)
      return false;
  }

  return true;
}

// Shared layout for an f32 FMA dot operand, which composePaddedLayout does not
// cover and upstream leaves unswizzled. Transposing a K-contiguous operand into
// an operand-major buffer can make lanes along the strided dimension revisit
// the same LDS banks. Pick the smallest XOR swizzle that separates them.
static ttg::SharedEncodingTrait
composeSwizzledLayoutForFMA(ttg::TensorOrMemDesc srcTy,
                            ArrayRef<unsigned> order, ttg::CGAEncodingAttr cga,
                            unsigned bitWidth,
                            const tt::AMD::TargetInfo &targetInfo) {
  if (!srcTy.getElementType().isF32()) {
    LDBG("No FMA LDS swizzle: only f32 operands are supported, got "
         << srcTy.getElementType());
    return {};
  }
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  if (!blocked || order.size() != 2) {
    LDBG("No FMA LDS swizzle: need a 2D blocked source layout, got "
         << srcTy.getEncoding() << " with rank " << order.size());
    return {};
  }
  unsigned contig = order[0];
  unsigned strided = order[1];
  // Cap the phases by the per-CTA extents: the buffer is per CTA, so phases
  // counted against the full shape would alias once toLinearLayout wraps them
  // modulo the local column count.
  SmallVector<int64_t> shape =
      ttg::getShapePerCTA(cga.getCTASplitNum(), srcTy.getShape());
  // The swizzle displaces a column by vec * phase, so vec is both the run that
  // stays contiguous through it and the step between phases. It has to cover
  // the widest LDS load/store (128 bits), or a 128-bit access gets reordered
  // internally and split, and the columns one warp spans, or a phase lands back
  // inside the span it was meant to step over and the lanes still share banks.
  int64_t warpSpan = static_cast<int64_t>(blocked.getThreadsPerWarp()[contig]) *
                     blocked.getSizePerThread()[contig];
  int64_t vec =
      std::max<int64_t>(warpSpan, std::max<int64_t>(1, 128 / bitWidth));
  // A row narrower than one complete bank row naturally starts at a different
  // bank. Keep those rows in the same phase and only swizzle when the row
  // stride wraps, while also keeping each lane's rows in the same phase.
  constexpr int64_t bankBitWidth = 32;
  int64_t elemsPerBankRow =
      static_cast<int64_t>(targetInfo.getSharedMemoryBanks()) * bankBitWidth /
      bitWidth;
  int64_t rowsPerBankRow =
      std::max<int64_t>(1, elemsPerBankRow / shape[contig]);
  int64_t perPhase =
      std::max<int64_t>(blocked.getSizePerThread()[strided], rowsPerBankRow);
  int64_t maxPhase =
      std::min<int64_t>({blocked.getThreadsPerWarp()[strided],
                         shape[contig] / vec, shape[strided] / perPhase});
  if (maxPhase <= 1) {
    LDBG("No FMA LDS swizzle: no two lanes share a bank, or the tile is too "
         "small to step over (vec="
         << vec << " perPhase=" << perPhase << ")");
    return {};
  }
  LDBG("FMA LDS swizzle: vec=" << vec << " perPhase=" << perPhase
                               << " maxPhase=" << maxPhase);
  return ttg::SwizzledSharedEncodingAttr::get(srcTy.getContext(), vec, perPhase,
                                              maxPhase, order, cga);
}

std::optional<ttg::SharedEncodingTrait> getSharedEncIfAllUsersAreDotEnc(
    Operation *loadOp, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const TargetFeatures &targetFeatures, const tt::AMD::TargetInfo &targetInfo,
    bool useAsyncCopy) {
  assert(loadOp);
  Value loadedValue = loadOp->getResult(0);
  llvm::SmallVector<ttg::SharedEncodingTrait> sharedEncs;
  // A user that already carries a shared encoding (e.g. a local_alloc feeding
  // memdesc_trans) pins the layout, so the FMA swizzle below must stay off
  bool pinnedByMemDescUser =
      llvm::any_of(loadedValue.getUsers(), [](Operation *u) {
        return u->getNumResults() == 1 &&
               isa<ttg::MemDescType>(u->getResult(0).getType());
      });
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SharedEncodingTrait tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingTrait>(memDesc.getEncoding());
      // If the immediate user is ttg::LocalAllocOp, likely it's created in
      // TritonAMDGPUOptimizeDotOperands. We should just respect it.
      if (!isa<ttg::LocalAllocOp>(user) &&
          !getSharedEncIfAllUsersAreDotEnc(user, axisInfoAnalysis,
                                           targetFeatures, targetInfo,
                                           useAsyncCopy)) {
        return std::nullopt;
      }
      LDBG("Deduced shared encoding candidate from memDesc: " << tempAttr);
      sharedEncs.push_back(tempAttr);
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto cgaLayout = ttg::getCGALayout(srcTy.getEncoding());

      auto order = getOrderForMemory(srcTy);
      if (useAsyncCopy && !targetInfo.supportsDirectToLdsScatter()) {
        // For architectures that don't support scattering into LDS we must
        // ensure that each warp writes a contiguous memory chunk. This requires
        // the shared memory order to follow the thread order, while preserving
        // the fastest dimension from the memory order if it's contiguous > 1 to
        // keep vectorization.
        auto srcTensorTy = cast<RankedTensorType>(srcTy);
        auto regOrder = triton::gpu::getOrder(srcTensorTy);
        auto threadOrder = triton::gpu::getThreadOrder(srcTensorTy);

        SetVector<unsigned> orderSet;

        auto regContig =
            triton::gpu::getContigPerThread(srcTensorTy)[regOrder[0]];
        unsigned elemBitWidth =
            getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
        unsigned finalRegContig =
            fitToValidDirectToLdsVecSize(regContig, elemBitWidth, targetInfo);
        // We only apply the order change if we find a vector size which works
        // for async copy.
        if (finalRegContig > 0) {
          // Preserve the fastest reg dim for contig > 1 to keep vectorization.
          if (finalRegContig > 1)
            orderSet.insert(order[0]);
          orderSet.insert(threadOrder.begin(), threadOrder.end());
          order = orderSet.takeVector();
        }
      }

      unsigned bitWidth = getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOperand conversions support
      // arbitrary shared memory ordering
      if (rank == 3) {
        // Move the batch dimension (dim #0) to be the last so that it will be
        // the slowest varying dimension.
        for (unsigned i = 0; i < rank; ++i)
          if (order[i] != 0)
            sharedOrder.emplace_back(order[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = order;
      }

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        // FMA LDS layout flip (read-side coalescing).
        // sharedOrder is otherwise inherited from the global-load contiguity.
        // When an FMA dot operand arrives K-contiguous in global memory (e.g.
        // A or B is transposed), the dot reads from LDS strided, serializing
        // the loads. Re-orient the shared buffer to operand-major so those
        // reads are coalesced.
        if (rank == 2 && isa<ttg::BlockedEncodingAttr>(dotOpEnc.getParent())) {
          SmallVector<unsigned> operandMajorOrder = ttg::getOrderForDotOperand(
              dotOpEnc.getOpIdx(), /*rank=*/2, /*kContig=*/false);
          if (sharedOrder != operandMajorOrder) {
            LDBG("FMA LDS layout flip: sharedOrder ["
                 << sharedOrder[0] << "," << sharedOrder[1]
                 << "] -> operand-major [" << operandMajorOrder[0] << ","
                 << operandMajorOrder[1]
                 << "] for opIdx=" << dotOpEnc.getOpIdx());
            sharedOrder = std::move(operandMajorOrder);
          }
        }
        // Determine if we can use padded layouts and fallback to swizzled
        // layouts if not
        bool canUseAsyncCopy = false;
        if (useAsyncCopy && isa<tt::LoadOp>(loadOp)) {
          // We pass numBuffers=2 because we assume the schedule will not
          // determine a single buffer (which does not work with AsyncCopy)
          canUseAsyncCopy = canBeConvertedToAsyncLoad(
              2, cast<tt::LoadOp>(loadOp), {}, axisInfoAnalysis, targetInfo);
        }
        tempAttr = composePaddedLayout(targetFeatures, dotOpEnc.getOpIdx(),
                                       dotOpEnc.getKWidth(), srcTy, sharedOrder,
                                       dotOpEnc, canUseAsyncCopy);
        // SwizzledSharedEncodingAttr::get dispatches on the dot's parent and
        // only computes AMD swizzles for MFMA and WMMA. A blocked (FMA) parent
        // falls through unswizzled, so compute one here.
        if (!tempAttr && !pinnedByMemDescUser &&
            isa<ttg::BlockedEncodingAttr>(dotOpEnc.getParent()))
          tempAttr = composeSwizzledLayoutForFMA(srcTy, sharedOrder, cgaLayout,
                                                 bitWidth, targetInfo);
        if (!tempAttr) {
          tempAttr = ttg::SwizzledSharedEncodingAttr::get(
              loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
              cgaLayout, bitWidth, /*needTrans=*/false);
        }
        LDBG("Deduced shared encoding candidate from dot layout: " << tempAttr);
        sharedEncs.push_back(tempAttr);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto mfmaEnc = getDotEncoding<ttg::AMDMfmaEncodingAttr>(
                userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = mfmaEnc.composeSharedLayoutForOperand(
              cgaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from mfma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        } else if (auto dotEnc = getDotEncoding<ttg::AMDWmmaEncodingAttr>(
                       userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = dotEnc.composeSharedLayoutForOperand(
              cgaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from wmma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        }
      }
    }
  }

  auto equalSharedEncIgnoreVec = [](ttg::SwizzledSharedEncodingAttr a,
                                    ttg::SwizzledSharedEncodingAttr b) {
    if (!a || !b)
      return false;
    return (a.getPerPhase() == b.getPerPhase() &&
            a.getMaxPhase() == b.getMaxPhase() &&
            a.getOrder() == b.getOrder() &&
            a.getCGALayout() == b.getCGALayout());
  };
  if (sharedEncs.empty() || !sharedEncs.front())
    return std::nullopt;
  auto maxVecSharedEnc = sharedEncs.front();

  // TODO add support for padded layouts. Right now they will use a separate
  // allocation
  for (auto sharedEnc : llvm::drop_begin(sharedEncs, 1)) {
    auto maybeSwizzShared =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(sharedEnc);
    auto maybeSwizzMaxVec =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(maxVecSharedEnc);

    if (!equalSharedEncIgnoreVec(maybeSwizzShared, maybeSwizzMaxVec)) {
      LDBG("Incompatible shared encodings");
      return std::nullopt;
    }
    if (maybeSwizzShared.getVec() > maybeSwizzMaxVec.getVec()) {
      maxVecSharedEnc = sharedEnc;
    }
  }

  LDBG("Deduced shared encoding: " << maxVecSharedEnc);

  return maxVecSharedEnc;
}

} // namespace mlir
