#include "FourStagePipeliner.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Support/LLVM.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-four-stage-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

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

FourStagePipeliner::FourStagePipeliner(scf::ForOp _forOp, int _numStages,
                                       int _globalPrefetch, int _localPrefetch,
                                       bool _useAsyncCopy)
    : forOp(_forOp), numStages(_numStages), numBuffers(1),
      useAsyncCopy(_useAsyncCopy), schedule(numStages),
      axisInfoAnalysis(forOp->getParentOfType<ModuleOp>()) {
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = _globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - _localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  options.predicateFn = streamPredication;
}

bool FourStagePipeliner::checkPrecondition(scf::ForOp forOp, int numStages) {
  // Skip the second loop (causual loop)
  static bool isFirst = true;
  if (!isFirst)
    return false;
  isFirst = false;

  unsigned dotCount{};
  unsigned reduceCount{};

  if (tt::getNumStagesOrDefault(forOp, numStages) != 4)
    return false;

  if (!forOp.getBody())
    return false;

  for (auto &op : *forOp.getBody()) {
    if (isa<triton::DotOpInterface>(op)) {
      dotCount++;
    } else if (isa<triton::ReduceOp>(op)) {
      reduceCount++;
    }
  }
  return dotCount == 2 && reduceCount == 2;
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult FourStagePipeliner::initSchedule(int maxIndirectionLevel) {
  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxIndirectionLevel;

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
  numBuffers = 2;

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

  // Create clusters in order of 4-stage pipeliner. You can swap lines below to
  // change the schedule of the loop. Not all combination are valid, e.g. if a
  // consumer and producer from the same stage are in the wrong cluster order
  // the loop expander will silently fail

  // DOT1
  dotClusters[0] = schedule.clusters.newAtBack();
  // SM2,
  softmaxClusters[0] = schedule.clusters.newAtBack();
  // Wait for V, LRV
  localReadClusters[0] = schedule.clusters.newAtBack();
  // ACK
  asyncCopyClusters[0] = schedule.clusters.newAtBack();
  // DOT2
  dotClusters[1] = schedule.clusters.newAtBack();
  // SM1
  softmaxClusters[1] = schedule.clusters.newAtBack();
  // Wait for K, LRK
  localReadClusters[1] = schedule.clusters.newAtBack();
  // ACV
  asyncCopyClusters[1] = schedule.clusters.newAtBack();

  // ATTENTION 4-stage (not used)
  clusters[SCHED_GLOBAL_LOAD] = softmaxClusters[1];
  clusters[SCHED_LOCAL_STORE] = asyncCopyClusters[0];
  clusters[SCHED_LOCAL_LOAD] = asyncCopyClusters[0];
  clusters[SCHED_ASYNC_WAIT] = asyncCopyClusters[0];
  clusters[SCHED_COMPUTE] = softmaxClusters[0];
  // Make assignments
  // std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusterVec;
  // std::generate(clusterVec.begin(), clusterVec.end(),
  //               [&]() { return schedule.clusters.newAtBack(); });

  // clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  // clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  // clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  // clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  // clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

bool FourStagePipeliner::createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                         Value extractIdx) {
  assert(useAsyncCopy);
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers == 1)
    return false;

  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  Value src = loadOp.getPtr();
  auto srcTy = cast<triton::gpu::TensorOrMemDesc>(src.getType());

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  auto sharedEncodingAttr =
      cast<ttg::SwizzledSharedEncodingAttr>(allocTy.getEncoding());

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
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, alloc, viewLoad);
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loadOp.getLoc(), src, viewLoad, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

  // Insert synchronization primitives to create barriers during lowering
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));

  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  // Create local load which consumes the async token from the AsyncWait
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad, waitOp);

  auto [loadStage, loadCluster] = schedule[loadOp];
  // Schedule new ops
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
  // Disable for FA
  // if (loadStage != stages[SCHED_LOCAL_LOAD])
  //   scheduleOp(waitOp, SCHED_ASYNC_WAIT);

  // if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
  //   scheduleOp(sharedLoad, SCHED_LOCAL_LOAD);

  loadOp->replaceAllUsesWith(ValueRange{sharedLoad});

  // 4-stage pipeliner scheduleing
  auto localLoadStage = loadStage == 0 ? 1 : 3;
  auto localLoadCluster = loadStage == 0 ? 1 : 0;
  schedule.insert(sharedLoad, localLoadStage,
                  localReadClusters[localLoadCluster]);
  schedule.insert(waitOp, localLoadStage, localReadClusters[localLoadCluster]);

  // Make sure that a possible cvt is in the same stage or otherwise it will not
  // get folded
  if (sharedLoad->hasOneUse()) {
    if (auto cvt =
            dyn_cast<ttg::ConvertLayoutOp>(*sharedLoad->getUsers().begin())) {
      LDBG("Change cvt layout stage and cluster");
      schedule.insert(cvt, localLoadStage, localReadClusters[localLoadCluster]);
    }
  }

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] &&
      sharedLoad->hasOneUse()) {
    if (auto cvt =
            dyn_cast<ttg::ConvertLayoutOp>(*sharedLoad->getUsers().begin()))
      scheduleOp(cvt, SCHED_LOCAL_LOAD);
  }

  // Delete old loadOp
  schedule.erase(loadOp);
  loadOp.erase();
  return true;
}

void FourStagePipeliner::createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                          Value extractIdx) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  Operation *copy = builder.clone(*loadOp);

  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

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
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  // Prefetch load ahead of the dot stage if is used by the dot.
  auto storeOp =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);
  scheduleOp(viewLoad, SCHED_LOCAL_STORE);
  scheduleOp(storeOp, SCHED_LOCAL_STORE);

  // Create local load
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
  Value result = sharedLoad.getResult();
  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
    scheduleOp(sharedLoad, SCHED_LOCAL_LOAD);

  // If the currently processed `LoadOp` is labeled with an index regarding
  // to which `DotOp` operand the corresponding data belongs to, then label the
  // expanded `LocalStoreOp` with the same index. This is required for
  // instruction scheduling hints to correctly count the emitted `ds_write`
  // instructions for each GEMM tile.
  if (auto attr = loadOp->getAttr(tt::amdgpu::OpIdxAttr::getMnemonic())) {
    storeOp->setAttr(tt::amdgpu::OpIdxAttr::getMnemonic(), attr);
  }

  loadOp->replaceAllUsesWith(ValueRange{result});

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] && result.hasOneUse()) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*result.getUsers().begin()))
      scheduleOp(cvt, SCHED_LOCAL_LOAD);
  }

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

// Create a map from load ops to their indirection levels and the final uses
// of the load op (another load op, or a dot op).
//
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
void FourStagePipeliner::computeLoadOpsToIndirectionLevelAndUse() {
  DenseSet<Operation *> seen;

  // Recursively visit the given op and its operands to discover all load ops
  // and collect their indirection levels and uses.
  std::function<void(Operation *, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        // Skip previously visited load ops.
        if (!seen.insert(op).second)
          return;

        if (isa<tt::LoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.emplace_back(op, distance, use);
          use = op;
          ++distance;
        }
        for (Value operand : op->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::DotOpInterface>(op))
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }
}

// Goes through all load ops to identify those that can be pipelined and assign
// layout to them.
void FourStagePipeliner::assignMemoryLayouts() {
  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(op))
      // TODO: We'd need to verify that the distance is the same.
      continue;

    auto loadOp = cast<tt::LoadOp>(op);
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy) {
      LDBG("Skip non-tensor load " << loadOp);
      continue;
    }

    auto pointeeTy =
        cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
    unsigned width = vec * pointeeTy.getIntOrFloatBitWidth();

    LDBG("assign memory layouts (width=" << width << ") for load " << loadOp);
    LoadInfo loadInfo;
    if (isa<tt::DotOpInterface>(use)) {
      // Only use shared memory when feeding into a dot op.
      loadInfo.usedByDot = true;
      // If the max continugous bits we can read is < 32, buffer in registers.
      if (width >= 32) {
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0)).value_or(nullptr);
      }
    } else if (auto useOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadToInfo already, it means that the use is not valid for pipelining
      // for some reason. We should skip this loadOp, too.
      //
      // Note that we have an assumption that the use of this loadOp has already
      // be processed in a previous loop iteration. This assumption is held by
      // how loadOpsToIndirectionLevelAndUse recursively collects
      // loadOpToIndLevelAndUse using DFS.
      if (loadToInfo.count(useOp) == 0) {
        continue;
      }
    }

    loadToInfo[op] = loadInfo;
  }
}

LogicalResult
FourStagePipeliner::scheduleLoads(DenseSet<Operation *> &rootUsers) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  computeLoadOpsToIndirectionLevelAndUse();
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return failure();

  // Check which loads are good for pipelining, and assign them memory layouts.
  assignMemoryLayouts();
  if (loadToInfo.empty())
    return failure();

  // Filter out load ops that cannot be pipelined.
  int resize = 0;
  for (int i = 0, e = loadOpToIndLevelAndUse.size(); i < e; ++i) {
    auto [loadOp, distance, use] = loadOpToIndLevelAndUse[i];
    if (loadToInfo.count(loadOp) != 0)
      loadOpToIndLevelAndUse[resize++] = loadOpToIndLevelAndUse[i];
  }
  loadOpToIndLevelAndUse.resize(resize);

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse)
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);

  LDBG("maxIndirectionLevel = " << maxIndirectionLevel);
  if (maxIndirectionLevel >= numStages)
    return failure();

  if (failed(initSchedule(maxIndirectionLevel)))
    return failure();

  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads =
      llvm::divideCeil(numStages - 2, maxIndirectionLevel + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Assign stages to the loads.
  // FA:
  //  Load1: Stage=0, cluster=1
  //  Load2: Stage=1, cluster=3
  int i{};
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    if (schedule.count(loadOp) > 0)
      continue;
    schedule.insert(loadOp, i, asyncCopyClusters[i == 0 ? 0 : 1]);
    i++;
  }

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(use)) {
      auto loadStage = schedule[loadOp].first;
      schedule.insert(use, loadStage + 2, dotClusters[loadStage == 0 ? 0 : 1]);
      // scheduleOp(use, SCHED_COMPUTE);
      rootUsers.insert(use);
    }
  }

  // Calculate distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }

  LLVM_DEBUG({
    LDBG("Chosen loads to pipeline:");
    for (const auto &[load, info] : loadToInfo) {
      LDBG("  - load: " << *load);
      LDBG("    distToUse: " << info.distToUse);
      LDBG("    usedByDot: " << info.usedByDot);
    }
  });

  return success();
}

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void FourStagePipeliner::scheduleDependencies() {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; ++stage) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      auto depCluster = cluster;
      bool override = false;
      if (llvm::isa<triton::DotOpInterface>(op) && stage == 3) {
        depCluster = softmaxClusters[0];
        override = true;
      }

      auto moveStages = [this, stage, cluster = cluster,
                         depCluster = depCluster, override](Operation *op) {
        if (llvm::isa<ttg::ConvertLayoutOp>(op)) {
          return std::make_pair(stage, cluster);
        }
        return std::make_pair(stage, depCluster);
      };
      schedule.insertDepsOfOp(op, false, false, moveStages);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void FourStagePipeliner::scheduleDistanceOneDependencies() {
  auto getNestedOperands = [](Operation *op) {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<tt::CoarseSchedule::Cluster *, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getArgNumber() == 0 || arg.getOwner() != op.getBlock())
        continue;
      auto yieldOp = op.getBlock()->getTerminator();
      Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
      Operation *defOp = v.getDefiningOp();
      if (!defOp || schedule.count(defOp) != 0)
        continue;
      if (isa<tt::LoadOp>(defOp)) {
        // Exception: schedule loads with a distance of 1 together with the
        // current op.
        schedule.insertIfAbsent(defOp, stage, cluster);
        schedule.insertDepsOfOp(defOp, stage, cluster, true);
      } else {
        if (dist1Cluster.count(&cluster) == 0) {
          dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
        }
        schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
        schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster], true);
      }
    }
  }
}

void FourStagePipeliner::scheduleRemainingToLastStage() {
  int lastStage = numStages - 1;
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  auto cluster = clusters[SCHED_COMPUTE];
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      opToCluster[&op] = cluster;
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == lastStage) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        tt::CoarseSchedule::Cluster userCluster = opToCluster[user];
        tt::CoarseSchedule::Cluster opCluster = schedule[op].second;
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, lastStage, cluster);
  }
}

// Create an allocation that can hold distance number of loadOp shapes.
Value FourStagePipeliner::createAlloc(
    Operation *loadOp, ttg::SwizzledSharedEncodingAttr sharedEnc) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), numBuffers);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  auto alloc = builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
  sharedMemAllocs.push_back(alloc);
  return alloc;
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
void FourStagePipeliner::createStreamOps() {
  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding || info.isAsync)
      continue;

    Value alloc = createAlloc(loadOp, info.sharedEncoding);
    assert(alloc && "Failed to create alloc for the async load.");
    loadToAllocs.emplace_back(loadOp, alloc);
  }

  IRRewriter builder(forOp.getContext());
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
  (void)addIterArgsToLoop(builder, forOp, {extractIdx});

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
      if (useAsyncCopy && createAsyncCopy(loadOp, alloc, extractIdx))
        continue;
      createStreamCopy(loadOp, alloc, extractIdx);
    }
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});
}

LogicalResult FourStagePipeliner::preprocessLoopAndBuildSchedule() {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  if (failed(scheduleLoads(rootUsers)))
    return failure();
  if (loadToInfo.empty())
    return failure();

  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    schedule.dump();
  });

  // Convert the loads into shared memory allocations and loads from them.
  createStreamOps();
  LLVM_DEBUG({
    LDBG("Coarse schedule with replaced laod ops:");
    schedule.dump();
  });

  // Schedule reductions
  int c = 2;
  for (auto reduceOp : forOp.getBody()->getOps<tt::ReduceOp>()) {
    schedule.insert(reduceOp, c, softmaxClusters[c == 2 ? 1 : 0]);
    c++;
  }

  for (auto exp2Op : forOp.getBody()->getOps<mlir::math::Exp2Op>()) {
    schedule.insert(exp2Op, 2, softmaxClusters[1]);
  }
  LLVM_DEBUG({
    LDBG("Coarse schedule after schedule reduction:");
    schedule.dump();
  });

  scheduleDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    schedule.dump();
  });

  scheduleDistanceOneDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    schedule.dump();
  });

  scheduleRemainingToLastStage();
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    schedule.dump();
  });

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
  for (auto alloc : sharedMemAllocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);

  return success();
}

LogicalResult FourStagePipeliner::pipelineLoop() {
  if (failed(preprocessLoopAndBuildSchedule()))
    return failure();
  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}
