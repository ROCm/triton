#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSTREAMPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace fourStage {
LogicalResult attPipelineLoop(scf::ForOp forOp, int numStages,
                              bool useAsyncCopy);
}

namespace {

static Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                                    Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp->getResult(0).getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp->getOperand(2));
    return ifOp;
  }
  return tt::predicateOp(rewriter, op, pred);
}

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexibility to
// better interleave instructions for better instruction-level parallelism.
//
// The code here creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
//    1.a. User may also specify `global_prefetch=<s>` to set the number of
//         stages between tt.load and ttg.local_store ops.
//    1.b. User may also specify `local_prefetch=<s>` to set the number of
//         stages between ttg.local_load and compute.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule method).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The ttg.local_store
//    will save the global load value to shared memory and the ttg.local_load
//    will load the relevant tiles for the tt.dot. These operations will be
//    scheduled according to various scheduling schemes outlined below in the
//    initSchedule method (see details there).
// 4. Finally the schedule will be passed to the PipelineExpander to rewrite
//    accordingly. The new implementation will consist of:
//    a. Prologue: containing the ramp-up of num_stages-1 stages for
//       iteratorions i=[0, num_stages-1).
//    b. New loop: ordered by cluster and iterated on each operation by
//       `i + (num_stages-op_stage)`.
//    c. Epilogue: ramp-down of the last `num_stages-1` iterations for the
//       ops in stages 1 to last_stage. This must consider that the loop
//       bounds may be shorter than num_stages. In this case, the epilogue
//       iterations must align with the prologue.
//

// Define categories of scheduling details per Operation types.
// The StreamPipeliner schedules 5 types of operations:
// 1. GLOBAL_LOAD: tt.load / ttg.async_copy_global_to_local
// 2. LOCAL_STORE: ttg.local_store
// 3. LOCAL_LOAD:  ttg.local_load
// 4. COMPUTE:     ops that use the loaded data
// 5. ASYNC_WAIT:  ttg.async_wait
// Note that ttg ops mentioned in the above list are created in this pass.
enum SchedType {
  SCHED_GLOBAL_LOAD,
  SCHED_LOCAL_STORE,
  SCHED_LOCAL_LOAD,
  SCHED_COMPUTE,
  SCHED_ASYNC_WAIT,
  SCHED_SIZE
};

struct LoadInfo {
  // Shared layout is used for loads feeding into dot ops.
  ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
  // The distance of this load's stage to its use' stage.
  int distToUse = 0;
  Operation *use = nullptr;
};

} // namespace

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult
initSchedule(int maxDist, int stages[SCHED_SIZE], int numStages,
             int &numBuffers, bool useAsyncCopy,
             std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters,
             tt::CoarseSchedule &schedule) {
  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxDist;

  LDBG(
      "Stage schedule:" << "  GLOBAL_LOAD stage = " << stages[SCHED_GLOBAL_LOAD]
                        << ", LOCAL_STORE stage = " << stages[SCHED_LOCAL_STORE]
                        << ", LOCAL_LOAD stage = " << stages[SCHED_LOCAL_LOAD]
                        << ", COMPUTE stage = " << stages[SCHED_COMPUTE]
                        << ", ASYNC_WAIT stage = " << stages[SCHED_ASYNC_WAIT]
                        << "; total = " << numStages);

  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    LDBG("Invalid stage schedule");
    return failure();
  }

  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy we need one more buffer since we are not using a
  // register buffer
  if (useAsyncCopy) {
    numBuffers += 1;
  }

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  int asyncWaitCluster = 0;

  // If tt.load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else
  //   Initiate ttg.local_store before tt.load
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  // If ttg.local_load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else if they share the buffer
  //   ttg.local_load must come first
  // else
  //   schedule ttg.local_load in the middle
  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  // Make assignments
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusterVec;
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

struct AsyncCopyChainOps {
  ttg::AsyncCopyGlobalToLocalOp copyOp;
  ttg::AsyncCommitGroupOp commitOp;
  ttg::AsyncWaitOp waitOp;
  ttg::LocalLoadOp localLoadOp;
};

AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                  Value extractIdx, scf::ForOp forOp) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Extract local subview from shared allocation
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);

  // If the load is used by an existing local allocation we replace it with the
  // new subview
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, userAlloc, viewLoad);
      allocsToErase.push_back(userAlloc);
    }
  }
  for (auto allocToErase : allocsToErase)
    allocToErase.erase();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loadOp.getLoc(), loadOp.getPtr(), viewLoad, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());

  // Insert synchronization primitives to create barriers during lowering
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));

  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  // Create local load which consumes the async token from the AsyncWait
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad, waitOp);

  return {copyOp, commitOp, waitOp, sharedLoad};
}

void scheduleAsyncCopy(
    const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
    tt::CoarseSchedule &schedule, const int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters) {
  auto [copyOp, commitOp, waitOp, localLoadOp] = asyncOps;
  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schdule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
    schedule.insert(localLoadOp, stages[SCHED_LOCAL_LOAD],
                    clusters[SCHED_LOCAL_LOAD]);

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] &&
      localLoadOp->hasOneUse()) {
    if (auto cvt =
            dyn_cast<ttg::ConvertLayoutOp>(*localLoadOp->getUsers().begin()))
      schedule.insert(cvt, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void createAndScheduleAsyncCopy(
    tt::LoadOp loadOp, Value alloc, Value extractIdx, scf::ForOp forOp,
    tt::CoarseSchedule &schedule, const int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters) {

  auto asyncOps = createAsyncCopy(loadOp, alloc, extractIdx, forOp);
  loadOp->replaceAllUsesWith(ValueRange{asyncOps.localLoadOp});

  scheduleAsyncCopy(asyncOps, loadOp, schedule, stages, clusters);

  schedule.erase(loadOp);
  loadOp.erase();
}

struct StreamCopyChainOps {
  tt::LoadOp copyOp;
  ttg::MemDescSubviewOp subviewOp;
  ttg::LocalStoreOp localStoreOp;
  ttg::LocalLoadOp localLoadOp;
};

StreamCopyChainOps createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                    Value extractIdx, scf::ForOp forOp) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  tt::LoadOp copy = cast<tt::LoadOp>(builder.clone(*loadOp));

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto subviewOp =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  // Clean up old local caches.
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, userAlloc,
                                      subviewOp.getResult());
      allocsToErase.push_back(userAlloc);
    }
  }
  for (auto allocToErase : allocsToErase)
    allocToErase.erase();

  // Prefetch load ahead of the dot stage if is used by the dot.
  auto storeOp = builder.create<ttg::LocalStoreOp>(loc, copy, subviewOp);

  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), subviewOp);

  return {copy, subviewOp, storeOp, sharedLoad};
}

void scheduleStreamCopy(
    const StreamCopyChainOps &streamOps, tt::LoadOp loadOp,
    tt::CoarseSchedule &schedule, const int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters) {
  auto [copyOp, subviewOp, localStoreOp, localLoadOp] = streamOps;
  auto [stage, cluster] = schedule[loadOp];
  schedule.insert(copyOp, stage, cluster);

  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
    schedule.insert(localLoadOp, stages[SCHED_LOCAL_LOAD],
                    clusters[SCHED_LOCAL_LOAD]);

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] &&
      localLoadOp->hasOneUse()) {
    if (auto cvt =
            dyn_cast<ttg::ConvertLayoutOp>(*localLoadOp->getUsers().begin()))
      schedule.insert(cvt, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void createAndScheduleStreamCopy(
    tt::LoadOp loadOp, Value alloc, Value extractIdx, scf::ForOp forOp,
    tt::CoarseSchedule &schedule, const int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters) {

  auto streamOps = createStreamCopy(loadOp, alloc, extractIdx, forOp);
  loadOp->replaceAllUsesWith(ValueRange{streamOps.localLoadOp});

  scheduleStreamCopy(streamOps, loadOp, schedule, stages, clusters);

  schedule.erase(loadOp);
  loadOp.erase();
}

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// with which dot operand |inputValue| is fed into if possible.
static ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue,
                                               unsigned *opIdx) {
  if (!llvm::hasSingleElement(inputValue.getUses()))
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }
  return getDotEncoding(user->getResult(0), opIdx);
}

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value loadedValue) {
  ttg::SwizzledSharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SwizzledSharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SwizzledSharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(userResult).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
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
        tempAttr = ttg::SwizzledSharedEncodingAttr::get(
            loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
            ctaLayout, bitWidth, /*needTrans=*/false);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        if (auto dotEnc = getDotEncoding(userResult, &opIdx)) {
          unsigned vecSize = llEnc.getLinearLayout().getNumConsecutiveInOut();
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = dotEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
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

LogicalResult scheduleLoads(
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo, int maxDist,
    int numStages, int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters,
    tt::CoarseSchedule &schedule) {
  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads = llvm::divideCeil(numStages - 2, maxDist + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, info] : loadToInfo) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(info.use))
      schedule.insert(info.use, stages[SCHED_COMPUTE], clusters[SCHED_COMPUTE]);
  }

  // Assign stages to the loads.
  for (auto [loadOp, info] : loadToInfo) {
    int stage = (maxDist - info.distToUse) * stagesBetweenLoads;
    schedule.insert(loadOp, stages[stage], clusters[SCHED_GLOBAL_LOAD]);
  }

  return success();
}

namespace {
bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               Value alloc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  // Compute the final vecSize we can use for the combination of sourceEncoding
  // and sharedEncoding. We can only use AsyncCopy if the width is >= 32 bit
  auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
  auto dstTy = cast<ttg::MemDescType>(alloc.getType());
  auto regLayout = triton::gpu::toLinearLayout(srcTy);
  // It's the allocation so we can pass the srcTy shape
  auto srcShape = srcTy.getShape();
  auto sharedLayout =
      triton::gpu::toLinearLayout(srcShape, dstTy.getEncoding(), srcShape);
  auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
  unsigned loadContig = regToSharedLayout.getNumConsecutiveInOut();
  unsigned width = loadContig * dstTy.getElementTypeBitWidth();
  if (width < 32)
    return false;

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads
  return triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis);
}
} // namespace

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
SmallVector<std::pair<Operation *, Value>> createAndScheduleStreamOps(
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo, scf::ForOp &forOp,
    const int &numBuffers, bool useAsyncCopy, tt::CoarseSchedule &schedule,
    const int stages[SCHED_SIZE],
    const std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> &clusters,
    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp.getContext());
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, loadOp->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    loadToAllocs.emplace_back(loadOp, alloc);
  }

  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Replace tt.loads with async copies or stream copies
  for (auto &[op, alloc] : loadToAllocs) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (useAsyncCopy && canBeConvertedToAsyncLoad(numBuffers, loadOp, alloc,
                                                    axisInfoAnalysis)) {
        createAndScheduleAsyncCopy(loadOp, alloc, extractIdx, forOp, schedule,
                                   stages, clusters);
      } else {
        createAndScheduleStreamCopy(loadOp, alloc, extractIdx, forOp, schedule,
                                    stages, clusters);
      }
    }
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});

  return loadToAllocs;
}

LogicalResult preprocessLoopAndBuildSchedule(scf::ForOp &forOp, int numStages,
                                             int stages[SCHED_SIZE],
                                             bool useAsyncCopy,
                                             tt::PipeliningOption &options) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(
      forOp->getParentOfType<ModuleOp>());
  int numBuffers = 1;
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusters;
  tt::CoarseSchedule schedule(numStages);

  auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
  triton::AMD::ISAFamily isaFamily = triton::AMD::ISAFamily::Unknown;
  if (arch)
    isaFamily = triton::AMD::deduceISAFamily(*arch);

  bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
  bool filterSmallVectors = isaFamily != triton::AMD::ISAFamily::CDNA4;
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      triton::gpu::loadOpsToIndirectionLevel(forOp, pipelineWithoutDot,
                                             axisInfoAnalysis, numStages,
                                             filterSmallVectors);

  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevel.size() << " loads to pipeline:");
    for (const auto &[l, i] : loadOpToIndLevel) {
      LDBG("  - load: " << *l);
      LDBG("    at distance: " << i.first);
      LDBG("    used by op: " << *i.second);
    }
  });

  if (loadOpToIndLevel.empty()) {
    LDBG("couldn't find any pipeline-able loads:\n" << *forOp);
    return failure();
  }

  llvm::MapVector<Operation *, LoadInfo> loadToInfo;
  int maxDist = -1;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    auto sharedEncoding =
        getSharedEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
    loadToInfo[load] = {sharedEncoding, distance, use};
    maxDist = std::max(maxDist, distance);
  }

  auto dumpSchedule = [&](llvm::StringRef msg) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      LDBG(msg);
      schedule.dump();
    });
  };

  if (failed(initSchedule(maxDist, stages, numStages, numBuffers, useAsyncCopy,
                          clusters, schedule)))
    return failure();

  if (failed(scheduleLoads(loadToInfo, maxDist, numStages, stages, clusters,
                           schedule)))
    return failure();
  dumpSchedule("Coarse schedule loads only:");

  // Convert the loads into shared memory allocations and loads from them.
  SmallVector<std::pair<Operation *, Value>> sharedMemAllocs =
      createAndScheduleStreamOps(loadToInfo, forOp, numBuffers, useAsyncCopy,
                                 schedule, stages, clusters, axisInfoAnalysis);
  dumpSchedule("Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule("Final coarse schedule:");

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  return success();
}

LogicalResult streamPipelineLoop(scf::ForOp forOp, int numStages,
                                 int globalPrefetch, int localPrefetch,
                                 bool useAsyncCopy) {

  int lastStage = numStages - 1;
  int stages[SCHED_SIZE];
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  tt::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  options.predicateFn = streamPredication;

  // Annotate loadOp in prologue for further moving up
  options.annotateFn = [](Operation *op,
                          tt::PipeliningOption::PipelinerPart part,
                          unsigned stage) {
    if (part != tt::PipeliningOption::PipelinerPart::Prologue)
      return;

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      loadOp->setAttr("amd.pipeliner_part",
                      StringAttr::get(op->getContext(), "prologue"));
    }
  };

  if (failed(preprocessLoopAndBuildSchedule(forOp, numStages, stages,
                                            useAsyncCopy, options)))
    return failure();
  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}

namespace fourStage {
// Four Stage pipeliner to create a schedule if we have 2 chained dots with ops
// in between. The goal of this pipeliner is to interleave the alu ops with the
// two dots in the loop. To achieve this it places the dots on consecutive
// stages and double buffers the loads feeding the dots. The coarse schedule
// looks like:
//   Stage0: (distance==1) loads for dot1
//   Stage1: (distance==1) loads for dot2, local loads for dot1
//   Stage2: dot1, interleaved ops part1
//   Stage3: local loads for dot2, dot2, interleaved ops part2
//
// Local writes for dot1 are scheduled placed in stage1 and for dot2 into stage2
// if we are not using async copies
//
// To optimize the interleaving of mfma and alu ops on AMD hardware (co-issue)
// we cluster all ops into 2 memory and 2 computation clusters and schedule them
// in the following order:
//   ComputeCluster1: dot1, interleaved ops part2
//   MemoryCluster1: LocalWrite2, LocalRead1, Loads2
//   ComputeCluster2: dot2, interleaved ops part1
//   MemoryCluster2: LocalWrite1, LocalRead2, Loads1
// In the implementation we split the clusters further to ensure consistent op
// scheduling, e.g. AsyncWait at the top of the memory cluster

enum FS_CLUSTERS {
  // Cluster0
  DOT1,
  ALU2,
  // Cluster1
  ASYNCWAIT2,
  LWRITE1,
  LLOAD2,
  LOAD1,
  // Cluster2
  DOT2,
  ALU1,
  // Cluster3
  ASYNCWAIT1,
  LWRITE2,
  LLOAD1,
  LOAD2,

  COUNT
};

enum FS_STAGES {
  STAGE_DOT1 = 2,
  STAGE_ALU1 = 2,
  STAGE_LOAD1 = 0,
  STAGE_LWRITE1 = 1,
  STAGE_LLOAD1 = 1,

  STAGE_DOT2 = 3,
  STAGE_ALU2 = 3,
  STAGE_LOAD2 = 1,
  STAGE_LWRITE2 = 2,
  STAGE_LLOAD2 = 3,
};

void fourStageCreateAndScheduleAsyncCopy(
    tt::LoadOp loadOp, Value alloc, Value extractIdx, scf::ForOp forOp,
    tt::CoarseSchedule &schedule,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT>
        &clusters) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Extract local subview from shared allocation
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);

  // If the load is used by an existing local allocation we replace it with the
  // new subview
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, userAlloc, viewLoad);
      allocsToErase.push_back(userAlloc);
    }
  }
  for (auto allocToErase : allocsToErase)
    allocToErase.erase();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loadOp.getLoc(), loadOp.getPtr(), viewLoad, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());

  // Insert synchronization primitives to create barriers during lowering
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));

  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  // Create local load which consumes the async token from the AsyncWait
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad, waitOp);

  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.erase(loadOp);
  // Schedule new ops
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage == FS_STAGES::STAGE_LOAD1) {
    schedule.insert(waitOp, FS_STAGES::STAGE_LLOAD1,
                    clusters[FS_CLUSTERS::ASYNCWAIT1]);
    schedule.insert(sharedLoad, FS_STAGES::STAGE_LLOAD1,
                    clusters[FS_CLUSTERS::LLOAD1]);
  } else {
    schedule.insert(waitOp, FS_STAGES::STAGE_LLOAD2,
                    clusters[FS_CLUSTERS::ASYNCWAIT2]);
    schedule.insert(sharedLoad, FS_STAGES::STAGE_LLOAD2,
                    clusters[FS_CLUSTERS::LLOAD2]);
  }

  loadOp->replaceAllUsesWith(ValueRange{sharedLoad});
  if (auto cvt =
          dyn_cast<ttg::ConvertLayoutOp>(*sharedLoad->getUsers().begin())) {
    auto [localLoadStage, localLoadCluster] = schedule[sharedLoad];
    schedule.insert(cvt, localLoadStage, localLoadCluster);
  }

  loadOp.erase();
}

void fourStageCreateAndScheduleStreamCopy(
    tt::LoadOp loadOp, Value alloc, Value extractIdx, scf::ForOp forOp,
    tt::CoarseSchedule &schedule,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT>
        &clusters) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  Operation *copy = builder.clone(*loadOp);

  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, loadStage, loadCluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  // Clean up old local caches.
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, userAlloc, viewLoad.getResult());
      allocsToErase.push_back(userAlloc);
    }
  }
  for (auto allocToErase : allocsToErase)
    allocToErase.erase();

  // Prefetch load ahead of the dot stage if is used by the dot.
  auto storeOp =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);

  // Create local load
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
  Value result = sharedLoad.getResult();
  // if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
  //   schedule.insert(sharedLoad, stages[SCHED_LOCAL_LOAD],
  //                   clusters[SCHED_LOCAL_LOAD]);

  if (loadStage == FS_STAGES::STAGE_LOAD1) {
    schedule.insert(storeOp, FS_STAGES::STAGE_LWRITE1,
                    clusters[FS_CLUSTERS::LWRITE1]);
    schedule.insert(sharedLoad, FS_STAGES::STAGE_LLOAD1,
                    clusters[FS_CLUSTERS::LLOAD1]);
  } else {
    schedule.insert(storeOp, FS_STAGES::STAGE_LWRITE2,
                    clusters[FS_CLUSTERS::LWRITE2]);
    schedule.insert(sharedLoad, FS_STAGES::STAGE_LLOAD2,
                    clusters[FS_CLUSTERS::LLOAD2]);
  }

  loadOp->replaceAllUsesWith(ValueRange{result});

  if (auto cvt =
          dyn_cast<ttg::ConvertLayoutOp>(*sharedLoad->getUsers().begin())) {
    auto [localLoadStage, localLoadCluster] = schedule[sharedLoad];
    schedule.insert(cvt, localLoadStage, localLoadCluster);
  }

  loadOp.erase();
}

LogicalResult fourStageScheduleDots(
    std::array<tt::DotOp, 2> dotOps,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT> &clusters,
    tt::CoarseSchedule &schedule) {
  schedule.insert(dotOps[0], STAGE_DOT1, clusters[FS_CLUSTERS::DOT1]);
  schedule.insert(dotOps[1], STAGE_DOT2, clusters[FS_CLUSTERS::DOT2]);

  return success();
}

LogicalResult fourStageScheduleLoads(
    std::array<tt::DotOp, 2> dotOps,
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT> &clusters,
    tt::CoarseSchedule &schedule) {
  for (auto [load, info] : loadToInfo) {
    if (info.use == dotOps[0]) {
      schedule.insert(load, STAGE_LOAD1, clusters[FS_CLUSTERS::LOAD1]);
    } else if (info.use == dotOps[1]) {
      schedule.insert(load, STAGE_LOAD2, clusters[FS_CLUSTERS::LOAD2]);
    }
  }
  return success();
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
SmallVector<std::pair<Operation *, Value>> fourStageCreateAndScheduleStreamOps(
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo, scf::ForOp &forOp,
    const int &numBuffers, bool useAsyncCopy, tt::CoarseSchedule &schedule,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT> &clusters,
    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp.getContext());
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    // Create an allocation that can hold distance nu/betweember of loadOp
    // shapes.
    builder.setInsertionPoint(forOp);
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    SmallVector<int64_t> bufferShape(ty.getShape());
    bufferShape.insert(bufferShape.begin(), numBuffers);
    Type memdescType =
        ttg::MemDescType::get(bufferShape, ty.getElementType(),
                              info.sharedEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    Value alloc =
        builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
    assert(alloc && "Failed to create alloc for the async load.");
    loadToAllocs.emplace_back(loadOp, alloc);
  }

  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Replace tt.loads with async copies or stream copies
  for (auto &[op, alloc] : loadToAllocs) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (useAsyncCopy && canBeConvertedToAsyncLoad(numBuffers, loadOp, alloc,
                                                    axisInfoAnalysis)) {
        fourStageCreateAndScheduleAsyncCopy(loadOp, alloc, extractIdx, forOp,
                                            schedule, clusters);
      } else {
        fourStageCreateAndScheduleStreamCopy(loadOp, alloc, extractIdx, forOp,
                                             schedule, clusters);
      }
    }
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});

  return loadToAllocs;
}

LogicalResult fourStageScheduleOpsBetweenDots(
    scf::ForOp forOp, std::array<tt::DotOp, 2> dotOps,
    tt::CoarseSchedule &schedule,
    const std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT>
        &clusters) {
  SetVector<Operation *> dot0Slice;
  getForwardSlice(Value(dotOps[0]), &dot0Slice);

  if (!dot0Slice.contains(dotOps[1])) {
    LDBG("Dot1 does not feed into Dot2");
    return failure();
  }

  // For each operand of the second dot we go back the def-chain if it's part of
  // the forward slice of the first dot. We want to find a good point to split
  // the def-chain into 2 separate schedule stages and cluster. The idea is to
  // find a expand_dim or broadcast op and split at the next alu/math op. This
  // should reduce the values we are loop carrying and helps with register
  // pressure.
  // There is one heuristic which ignored trucnf ops since we get bad codegen if
  // we move it to the second half
  for (auto operand : dotOps[1]->getOperands()) {
    auto operandDefOp = operand.getDefiningOp();

    // Skip if the op is not part of the forward slice
    if (!operandDefOp || !dot0Slice.contains(operand.getDefiningOp()))
      continue;

    // Schedule the ops directly feeding the second dot as ALU2
    schedule.insertIfAbsent(operandDefOp, FS_STAGES::STAGE_ALU2,
                            clusters[FS_CLUSTERS::ALU2]);

    LDBG("Check dot operand: " << operand);
    // DFS-like traversal of the def-chain. For each search item we store a bool
    // to signal if we already passed an broadcast/expand_dim op to signal that
    // we split on the next alu op.
    struct SearchItem {
      Value v;
      bool splitOnAlu{false};
    };
    llvm::SmallVector<SearchItem> queue;
    queue.push_back({operand, false});

    while (!queue.empty()) {
      auto [v, splitOnAlu] = queue.pop_back_val();

      // Abort path if we hit a blockarg or left the forward slice of dot0
      auto defOp = v.getDefiningOp();
      if (!defOp)
        continue;
      if (!dot0Slice.contains(defOp)) {
        LDBG("Found unrelated op to previous dot: " << v);
        continue;
      }

      bool isAluOp = defOp->getDialect()->getNamespace() ==
                         arith::ArithDialect::getDialectNamespace() &&
                     !isa<arith::TruncFOp>(defOp);
      isAluOp = isAluOp || defOp->getDialect()->getNamespace() ==
                               math::MathDialect::getDialectNamespace();

      // If the op has already a schedule we do not split here
      if (schedule.count(defOp) != 0) {
        LDBG("Found op with previous schedule: " << v);
        splitOnAlu = false;
      }

      // If the op is an alu op and we passed an expand/broadcast we split here
      if (splitOnAlu && isAluOp) {
        LDBG("Found alu op schedule to first alu cluster: " << *defOp);
        schedule.insert(defOp, FS_STAGES::STAGE_ALU1,
                        clusters[FS_CLUSTERS::ALU1]);
        continue;
      }
      LDBG("Skip non alu op: " << *defOp);
      // Follow def chain
      for (Value op2 : defOp->getOperands()) {
        queue.push_back({op2, splitOnAlu || !isAluOp});
      }
    }
  }

  // Schedule ops using dot1 but not feeding into dot2 to overlap with dot2
  auto yield = forOp.getBody()->getTerminator();
  for (auto yieldOperand : yield->getOperands()) {
    auto defOp = yieldOperand.getDefiningOp();
    if (!defOp || !dot0Slice.contains(defOp))
      continue;

    schedule.insertIfAbsent(defOp, FS_STAGES::STAGE_ALU2,
                            clusters[FS_CLUSTERS::ALU2]);
  }

  return success();
}

LogicalResult
fourStagePreprocessLoopAndBuildSchedule(scf::ForOp &forOp, int numStages,
                                        bool useAsyncCopy,
                                        tt::PipeliningOption &options) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(
      forOp->getParentOfType<ModuleOp>());
  tt::CoarseSchedule schedule(numStages);

  auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
  triton::AMD::ISAFamily isaFamily = triton::AMD::ISAFamily::Unknown;
  if (arch)
    isaFamily = triton::AMD::deduceISAFamily(*arch);

  bool filterSmallVectors = isaFamily != triton::AMD::ISAFamily::CDNA4;
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      triton::gpu::loadOpsToIndirectionLevel(
          forOp, /*pipelineWithoutDot=*/false, axisInfoAnalysis, numStages,
          filterSmallVectors);

  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevel.size() << " loads to pipeline:");
    for (const auto &[l, i] : loadOpToIndLevel) {
      LDBG("  - load: " << *l);
      LDBG("    at distance: " << i.first);
      LDBG("    used by op: " << *i.second);
    }
  });

  if (loadOpToIndLevel.empty()) {
    LDBG("couldn't find any pipeline-able loads:\n" << *forOp);
    return failure();
  }

  if (llvm::any_of(loadOpToIndLevel,
                   [](auto it) { return it.second.first != 0; })) {
    LDBG("Does not support indirect loads yet\n");
    return failure();
  }

  llvm::MapVector<Operation *, LoadInfo> loadToInfo;
  int maxDist = -1;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    auto sharedEncoding =
        getSharedEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
    loadToInfo[load] = {sharedEncoding, distance, use};
    maxDist = std::max(maxDist, distance);
  }

  std::array<tt::CoarseSchedule::Cluster, FS_CLUSTERS::COUNT> clusters;
  std::generate(clusters.begin(), clusters.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  auto dotOpsVec = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
  if (dotOpsVec.size() != 2) {
    LDBG("Does only work with 2 dots");
    return failure();
  }
  std::array<tt::DotOp, 2> dotOps = {dotOpsVec[0], dotOpsVec[1]};

  auto dumpSchedule = [&](llvm::StringRef msg) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      LDBG(msg);
      schedule.dump();
    });
  };

  if (failed(fourStageScheduleDots(dotOps, clusters, schedule)))
    return failure();
  if (failed(fourStageScheduleLoads(dotOps, loadToInfo, clusters, schedule)))
    return failure();
  dumpSchedule("Coarse schedule load and dots only:");

  // Convert the loads into shared memory allocations and loads from them.
  if (failed(
          fourStageScheduleOpsBetweenDots(forOp, dotOps, schedule, clusters))) {
    return failure();
  }
  dumpSchedule("Coarse schedule after schedule ops between dots:");

  // Convert the loads into shared memory allocations and loads from them.
  int numBuffers = 2;
  SmallVector<std::pair<Operation *, Value>> sharedMemAllocs =
      fourStageCreateAndScheduleStreamOps(loadToInfo, forOp, numBuffers,
                                          useAsyncCopy, schedule, clusters,
                                          axisInfoAnalysis);
  dumpSchedule("Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule("Final coarse schedule:");

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  // Explicitly deallocate created allocations.
  for (auto [_load, alloc] : sharedMemAllocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);

  return success();
}

LogicalResult attPipelineLoop(scf::ForOp forOp, int numStages,
                              bool useAsyncCopy) {
  // Note that we already checked general things (like no barriers, distances
  // etc.)
  // Check if we can 4-stage pipeline loop
  auto dotCount = llvm::range_size(forOp.getBody()->getOps<tt::DotOp>());
  if (dotCount != 2) {
    LDBG("Does only support 2 dots");
    return failure();
  }

  if (numStages != 4) {
    LDBG("Only works with num_stages==4");
    return failure();
  }

  // Four stage pipeliner

  LDBG("Use four stage pipeliner");

  tt::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  options.predicateFn = streamPredication;

  // Annotate loadOp in prologue for further moving up
  options.annotateFn = [](Operation *op,
                          tt::PipeliningOption::PipelinerPart part,
                          unsigned stage) {
    if (part != tt::PipeliningOption::PipelinerPart::Prologue)
      return;

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      loadOp->setAttr("amd.pipeliner_part",
                      StringAttr::get(op->getContext(), "prologue"));
    }
  };

  if (failed(fourStagePreprocessLoopAndBuildSchedule(forOp, numStages,
                                                     useAsyncCopy, options)))
    return failure();
  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  if (failed(tt::pipelineForLoop(rewriter, forOp, options))) {
    assert(false);
  }
  return success();
}

} // namespace fourStage

struct PipelinePass : impl::TritonAMDGPUStreamPipelineBase<PipelinePass> {
  using impl::TritonAMDGPUStreamPipelineBase<
      PipelinePass>::TritonAMDGPUStreamPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check numStages
    if (globalPrefetch < 0 || globalPrefetch >= numStages) {
      moduleOp.emitError("global prefetch control must be in [0, ")
          << numStages << "); " << globalPrefetch << " is out of range";
      return signalPassFailure();
    }

    if (localPrefetch < 0 || localPrefetch >= numStages) {
      moduleOp.emitError("local prefetch control must be in [0, ")
          << numStages << "); " << localPrefetch << " is out of range";
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (tt::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!triton::gpu::isSafeToPipeline(forOp)) {
        LDBG("Loop not safe to pipeline:\n" << *forOp);
        continue;
      }
      if (succeeded(fourStage::attPipelineLoop(
              forOp, tt::getNumStagesOrDefault(forOp, numStages),
              useAsyncCopy))) {
        continue;
      }
      (void)streamPipelineLoop(forOp,
                               tt::getNumStagesOrDefault(forOp, numStages),
                               globalPrefetch, localPrefetch, useAsyncCopy);
    }

    if (useAsyncCopy) {
      llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
      moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
      tt::combineRedundantWaitOps(waitOps);
    }
  }
};

} // namespace mlir
