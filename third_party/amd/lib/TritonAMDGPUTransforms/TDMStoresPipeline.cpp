#include "TritonAMDGPUTransforms/TDMStoresPipeline.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace {

// Bookkeeping for one descriptor store / scatter we want to pipeline.
struct TDMStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TDMStore> getTDMStores(scf::ForOp forOp) {
  SmallVector<TDMStore> stores;
  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::DescriptorStoreLikeOpInterface>(op)) {
      stores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
    } else if (isa<scf::ForOp>(op)) {
      // Don't recurse into nested loops; they'd be handled separately.
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return stores;
}

// Lift a single LDS allocation outside the loop, sized like store.src.
static Value createAlloc(scf::ForOp &forOp, const TDMStore &store) {
  OpBuilder builder(forOp);
  RankedTensorType ty = store.src.getType();
  auto encoding = getEncodingFromDescriptor(store.op, ty, store.desc);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      ttg::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, store.op->getLoc(), memdescType);
}

// Replace one descriptor_{store,scatter} with the pipelined async TDM
// sequence:
//
//   amdg.async_tdm_wait num=0          (wait for previous iter's TDM write
//                                       to release the LDS buffer)
//   ttg.local_store src, alloc         (write current iter's data into LDS)
//   amdg.async_tdm_copy_local_to_global  OR  amdg.async_tdm_scatter
//
// This single-buffers the LDS allocation across iterations and lets the
// outgoing async store overlap with the next iteration's compute.
static void createTDMAsyncCopy(scf::ForOp forOp, const TDMStore &store,
                               Value alloc) {
  OpBuilder builder(store.op);
  Location loc = store.op->getLoc();

  ttag::AsyncTDMWait::create(builder, loc, ArrayRef<Value>{}, 0);
  ttg::LocalStoreOp::create(builder, loc, store.src, alloc);

  Value desc = store.desc;
  if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(store.op)) {
    ttag::AsyncTDMCopyLocalToGlobalOp::create(builder, loc, desc,
                                              storeOp.getIndices(), alloc,
                                              /*barrier=*/Value{});
  } else {
    auto scatterOp = cast<tt::DescriptorScatterOp>(store.op);
    // Mirror TensorScatterLowering: the shared TritonToTritonGPU pass gives
    // the indices an NVIDIA-oriented layout, so re-layout them to AMD's TDM
    // gather/scatter index encoding before issuing the async op.
    auto indices = scatterOp.getXOffsets();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto idxEnc = getTDMGatherScatterIndexEncoding(scatterOp, indicesType);
    if (indicesType.getEncoding() != idxEnc) {
      auto newIdxType = RankedTensorType::get(
          indicesType.getShape(), indicesType.getElementType(), idxEnc);
      indices = ttg::ConvertLayoutOp::create(builder, loc, newIdxType, indices);
    }
    ttag::AsyncTDMScatterOp::create(builder, loc, desc, indices,
                                    scatterOp.getYOffset(), alloc,
                                    /*barrier=*/Value{});
  }

  store.op->erase();
}

} // namespace

bool mlir::pipelineTDMStores(scf::ForOp forOp) {
  SmallVector<TDMStore> stores = getTDMStores(forOp);
  if (stores.empty())
    return false;

  // Reuse a single allocation across stores with the same src shape/type.
  // This is safe because every iteration starts with `wait num=0` before
  // the local_store, so the allocation is never live across iterations.
  DenseMap<Operation *, Value> storeToAlloc;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> allocs;
  for (const TDMStore &store : stores) {
    RankedTensorType srcTy = store.src.getType();
    auto key = std::make_pair(srcTy.getShape(), srcTy.getElementType());
    Value &alloc = allocs[key];
    if (!alloc)
      alloc = createAlloc(forOp, store);
    storeToAlloc[store.op] = alloc;
  }

  for (const TDMStore &store : stores)
    createTDMAsyncCopy(forOp, store, storeToAlloc[store.op]);

  // After the loop: drain the last in-flight TDM write, then free the
  // allocation(s).
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  ttag::AsyncTDMWait::create(builder, forOp->getLoc(), ArrayRef<Value>{}, 0);
  for (auto it : storeToAlloc)
    ttg::LocalDeallocOp::create(builder, forOp->getLoc(), it.second);

  return true;
}
