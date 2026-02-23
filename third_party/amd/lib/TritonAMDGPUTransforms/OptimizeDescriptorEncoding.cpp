#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <numeric>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define DEBUG_TYPE "tritonamdgpu-optimize-descriptor-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::PaddedSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEncPadded(
    Value loadedValue, const triton::AMD::TargetInfo &targetInfo) {
  ttg::PaddedSharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::PaddedSharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::PaddedSharedEncodingAttr>(memDesc.getEncoding());
      auto newAttr =
          getSharedEncIfAllUsersAreDotEncPadded(user->getResult(0), targetInfo);

      if (!newAttr.has_value())
        return std::nullopt;

      auto interval = tempAttr.getIntervals()[0];
      auto padding = tempAttr.getPaddings()[0];
      // Update interval and padding if we find a compatible shared encoding
      // down the chain
      if (newAttr.has_value() && *newAttr) {
        interval = newAttr->getIntervals()[0];
        padding = newAttr->getPaddings()[0];
      }

      tempAttr = ttg::PaddedSharedEncodingAttr::get(
          tempAttr.getContext(), interval, padding,
          tempAttr.getLinearComponent());
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto cgaLayout = ttg::getCGALayout(srcTy.getEncoding());
      auto order = getOrderForMemory(srcTy);
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
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
        // For async descriptor loads, enable padding.
        tempAttr =
            composePaddedLayout(targetInfo, dotOpEnc.getOpIdx(),
                                dotOpEnc.getKWidth(), srcTy, sharedOrder);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto dotEnc = getDotEncoding<ttg::AMDWmmaEncodingAttr>(
                userResult, &opIdx, &vecSize)) {
          tempAttr =
              composePaddedLayout(targetInfo, opIdx, vecSize, srcTy, order);
        }
      }
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

Attribute findEncodingFromUsers(Operation *op) {
  if (auto load = dyn_cast<tt::DescriptorLoadOp>(op)) {
    auto arch = getAMDArch(op->getParentOfType<ModuleOp>());
    auto targetInfo = tt::AMD::TargetInfo(arch.value_or("").str());
    auto paddedEnc = getSharedEncIfAllUsersAreDotEncPadded(load, targetInfo);
    LDBG("findEncodingFromUsers: LoadDesc Op Padded Encoding"
         << (paddedEnc.has_value() ? "found" : "not found"));
    return (paddedEnc.value_or(nullptr));
  } else {
    // Pick the first use that has a shared encoding
    for (auto use : op->getUsers()) {
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(use)) {
        if (auto encoding = dyn_cast<ttg::PaddedSharedEncodingAttr>(
                localAlloc.getType().getEncoding()))
          return encoding;
      } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
        if (auto encoding = dyn_cast<ttg::PaddedSharedEncodingAttr>(
                store.getSrc().getType().getEncoding()))
          return encoding;
      }
    }
  }
  return {};
}

Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                    ttg::CGAEncodingAttr cgaLayout,
                                    ArrayRef<int64_t> useShape,
                                    unsigned numCTAs) {
  auto ctx = tensorType.getContext();
  ArrayRef<int64_t> shape = useShape.empty() ? tensorType.getShape() : useShape;
  unsigned rank = shape.size();
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);
  if (!cgaLayout) {
    SmallVector<unsigned> ctasPerCGA(tensorType.getRank(), 1);
    ctasPerCGA.back() = numCTAs;
    cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                      ctasPerCGA, order);
  } else if (cgaLayout.getRank() != tensorType.getRank()) {
    cgaLayout = ttg::updateCGALayoutForShape(cgaLayout, shape);
  }
  auto blockShapePerCTA =
      triton::gpu::getShapePerCTA(cgaLayout.getCTASplitNum(), shape);

  auto elemWidth = tensorType.getElementType().getIntOrFloatBitWidth();
  unsigned padAmount = 128 / elemWidth;
  // Restrict pad interval (calculated from TDM descriptor's pad interval field)
  // Fallback to swizzled encoding if the interval exceeds this limit.
  // TODO: Query pad interval limit from target info
  unsigned maxPadIntervalElements = 256u * 32 / elemWidth;
  unsigned padInterval = static_cast<unsigned>(blockShapePerCTA[order[0]]);
  if (padInterval > maxPadIntervalElements) {
    return ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, order, cgaLayout);
  }

  return ttg::PaddedSharedEncodingAttr::get(ctx, {{padInterval, padAmount}},
                                            order, shape, cgaLayout);
}

bool isForcedToDefault(Operation *op) {
  return isa<tt::CallOp, tt::ReturnOp>(op);
}

} // anonymous namespace

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDESCRIPTORENCODING
#include "TritonAMDGPUTransforms/Passes.h.inc"

// This pass assigns encoding to each descriptor in the function. Descriptors
// are created using `tl.make_tensor_descriptor` or passed in as arguments to
// the kernel. They are used by TDM load/store/gather/scatter. We assign shared
// memory encoding (e.g., padded) to the descriptors and use it for deriving
// encodings on descriptor ops including load/store/gather/scatter. The pass
// works in two phases: First, we derive a favorable encoding for each
// descriptor based on its uses (e.g., load -> tt.dot) and store it as
// EncodingInfo for each descriptor. The EncodingInfo is propagated to other
// desc descriptors through fixed point iteration. Finally, the computed
// EncodingInfo is fully materialized and assigned to the descriptors.
// Example:
//   %d = tt.make_tensor_descriptor ...   ; no encoding yet
//   %r = scf.for ... iter_args(%di = %d) -> ... {
//     %x = tt.descriptor_load %di ... ; use gives desired encoding for %di
//     %y = tt.dot %x, %b, %c           ; encoding from dot
//     scf.yield %di                    ; same encoding is propagated
//   }
//
class TritonAMDGPUOptimizeDescriptorEncodingPass
    : public impl::TritonAMDGPUOptimizeDescriptorEncodingBase<
          TritonAMDGPUOptimizeDescriptorEncodingPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    ttg::assignMemoryLayouts(m, findEncodingFromUsers,
                             getFallbackSharedEncoding, updateEncodingForShape,
                             isForcedToDefault);
  }
};

} // namespace mlir
