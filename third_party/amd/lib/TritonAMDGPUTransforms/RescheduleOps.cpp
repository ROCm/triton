#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/SchedulerMachineModel.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef LLVM_DEBUG
#define LLVM_DEBUG(X) X

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-reschedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

/******************************************************************************
  Reschedule ttgir after refine-ops-pass to interleave refined ops at the ttgir
  level, and thereby improve llir order which will improve
  scheduling and regalloc of backend compiler.

  The goals of rescheduling the basic block are:
  (1) Triton scheduling only needs to compliment LLVM scheduler,
      not redo same heuristics.
  (2) To create a better op order before LLIR to help LLVM backend scheduler.
  (3) Inject sched.barriers into op order to provide scheduling guard rails
      to backend scheduler to constrain pre-RA and post-RA scheduling.

  Scheduling consists of multiple passes controlled by SchedManager:
  (1) Analyse the current op sequence.
  (2) Create additional sets of dependencies and priorities in SchedDag.
  (3) Reschedule based on ready-list,
      (a) Select best node based on heuristic.
      (b) Remove node and dependencies from nodes in SchedDag.
      (c) Update ready-list.
  (4) Results in a new op sequence.
******************************************************************************/

namespace {

// TODO (ravil): Note, took function from `SchedInstructions.cpp`.
// we need to combine these two implementations
Operation *createSchedBarrier(OpBuilder &rewriter, Location loc,
                              uint32_t maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

/*
  Bitmasks for sched.barrier(mask). Adding bits to mask allows that op type
  to cross the barrier. So this mask allows everything to cross.
  Remove bits later will prevent crossing the barrier.
  Name mask based on what barrier blocks.
  This mask has all possible bits turned on except for the non_mem_non_sideffect
  because it overrides many other bits.
*/
uint32_t schedBarMaskBlockNone =
    0
    //|static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::none)
    //|static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::non_mem_non_sideffect)
    | static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::valu) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::salu) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_vmem) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_read) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_write) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_ds) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_read) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_write) |
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::transcendental);

uint32_t schedBarMaskBlockAll = 0;

uint32_t schedBarMaskBlockDot =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma);
uint32_t schedBarMaskBlockDsRead =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_read);
uint32_t schedBarMaskBlockDsWrite =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_write);

uint32_t schedBarMaskBlockDotLds =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_ds) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_read) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_write);

uint32_t schedBarMaskBlockDotLdsGlobal =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_ds) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_read) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::ds_write) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_vmem) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_read) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_write);

uint32_t schedBarMaskBlockGlobal =
    schedBarMaskBlockNone ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::all_vmem) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_read) ^
    static_cast<uint32_t>(mlir::amdgpu::sched_barrier_opt_enum::vmem_write);

enum class SchedDirection { TopDown, BottomUp };

/******************************************************************************
  Note: Dependencies point from child to parent; parents must preceed children
  in op order. Therefore dependencies point upward in dag.

  parent / dependee / dst / scheduled before
    ^
    |
  child / dependent / src / scheduled after
******************************************************************************/

/*
  This stores a priority for scheduling one node vs another.
  This is an optional parameter which a PriorityCalculator sets
  and a SchedulingHeuristic might use.

  For example, we want to get to a dot asap, so we want to analyze which are on
  critical path. Also, we want to set the relative order of LocalLoadOps.
*/
enum class SchedDagNodePriorityType : uint32_t {
  DotCriticalPath = 0,
  LocalStoreCriticalPath = 1,
  Size // Keep as last to know size().
};
using SchedDagNodePriorityDataType = int32_t;
using SchedDagNodePriority =
    SmallVector<SchedDagNodePriorityDataType,
                static_cast<uint32_t>(SchedDagNodePriorityType::Size)>;
constexpr SchedDagNodePriorityDataType schedDagNodePriorityUnset =
    std::numeric_limits<SchedDagNodePriorityDataType>::lowest();
StringRef toString(SchedDagNodePriorityType type) {
  switch (type) {
  case SchedDagNodePriorityType::DotCriticalPath:
    return "DotCriticalPath";
  case SchedDagNodePriorityType::LocalStoreCriticalPath:
    return "LocalStoreCriticalPath";
  case SchedDagNodePriorityType::Size:
    return "Size";
  default:
    return "ERROR";
  }
}

/******************************************************************************
  SchedDagNode contains op, parents and children dependencies.
  Before scheduling nodes are created and dependencies are added.
  During scheduling, each node scheduled removes it from the dag,
  and that node's deps are removed.
  Nodes without parents are ready to be scheduled if Direction=TopDown.
  Nodes without children are ready to be scheduled if Direction=BottomUp.
******************************************************************************/
struct SchedDagNode {
  SchedDagNode(Operation *op)
      : op(op), priority(static_cast<uint32_t>(SchedDagNodePriorityType::Size),
                         schedDagNodePriorityUnset) {
    static int32_t serialId = 0;
    id = serialId++;
    opStr = op->getName().getStringRef();
    resetPriorities();
  }

  SchedDagNode(const SchedDagNode &node)
      : op(node.op), id(node.id), opStr(node.opStr), children(node.children),
        parents(node.parents), priority(node.priority) {}

  // For DenseMapInfo to create empty/tombstone entries.
  SchedDagNode(int32_t i) : op(nullptr), id(i), opStr("") {}

  void addChild(SchedDagNode *node) { children.insert(node); }
  void addParent(SchedDagNode *node) { parents.insert(node); }
  Operation *getOp() { return op; }
  bool hasChildren() { return !children.empty(); }
  bool hasParents() { return !parents.empty(); }
  int32_t numChildren() { return children.size(); }
  int32_t numParents() { return parents.size(); }

  /*
    TopDown: node is ready to schedule when it has no parents.
    BottomUp: node is ready to schedule when it has no children.
  */
  template <SchedDirection Direction> bool isReady() {
    if constexpr (Direction == SchedDirection::TopDown) {
      return parents.empty();
    } else {
      return children.empty();
    }
  }

  const llvm::SetVector<SchedDagNode *> &getChildren() { return children; }
  const llvm::SetVector<SchedDagNode *> &getParents() { return parents; }

  bool removeChild(SchedDagNode *node) {
    if (children.contains(node)) {
      children.remove(node);
      return true;
    }
    return false;
  }

  bool removeParent(SchedDagNode *node) {
    if (parents.contains(node)) {
      parents.remove(node);
      return true;
    }
    return false;
  }

  void clearDeps() {
    children.clear();
    parents.clear();
  }

  // Manipulate the node's priorities.
  void setPriority(SchedDagNodePriorityType type,
                   SchedDagNodePriorityDataType value) {
    priority[static_cast<uint32_t>(type)] = value;
  }
  SchedDagNodePriorityDataType
  getPriority(SchedDagNodePriorityType type) const {
    return priority[static_cast<uint32_t>(type)];
  }
  bool hasPriority(SchedDagNodePriorityType type) const {
    return getPriority(type) != schedDagNodePriorityUnset;
  }
  void resetPriorities() {
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++i) {
      setPriority(static_cast<SchedDagNodePriorityType>(i),
                  schedDagNodePriorityUnset);
    }
  }
  void propagateHigherPriorityToParents(SchedDagNodePriorityType type) {
    assert(hasPriority(type));
    for (SchedDagNode *parent : parents) {
      // First time setting priorities of this type.
      if (!parent->hasPriority(type) ||
          getPriority(type) > parent->getPriority(type)) {
        parent->setPriority(type, getPriority(type));
        parent->propagateHigherPriorityToParents(type);
      } else if (getPriority(type) == parent->getPriority(type)) {
        // Second+ time setting priorities of this type.
        parent->propagateHigherPriorityToParents(type);
      }
    }
  }
  void propagateLowerPriorityToChildren(SchedDagNodePriorityType type) {
    assert(hasPriority(type));
    for (SchedDagNode *child : children) {
      if (!child->hasPriority(type) ||
          getPriority(type) < child->getPriority(type)) {
        child->setPriority(type, getPriority(type));
        child->propagateLowerPriorityToChildren(type);
      } else if (getPriority(type) == child->getPriority(type)) {
        child->propagateLowerPriorityToChildren(type);
      }
    }
  }

  Operation *op;
  // Unique node id.
  int32_t id;
  StringRef opStr;

  // Children depend on this node; this node must be scheduled before children.
  llvm::SetVector<SchedDagNode *> children;

  // This node depends on parents; this node must be scheduled after parents.
  llvm::SetVector<SchedDagNode *> parents;
  // Priorities are enumerated and optionally set by
  // and used by SchedulingHeuristics.
  SchedDagNodePriority priority;
};

struct SchedDagNodeDenseMapInfo : public llvm::DenseMapInfo<SchedDagNode> {
  static inline SchedDagNode getEmptyKey() {
    return SchedDagNode(DenseMapInfo<int32_t>::getEmptyKey());
  }
  static inline SchedDagNode getTombstoneKey() {
    return SchedDagNode(DenseMapInfo<int32_t>::getTombstoneKey());
  }
  static unsigned getHashValue(const SchedDagNode &node) {
    return DenseMapInfo<int32_t>::getHashValue(node.id);
  }
  static bool isEqual(const SchedDagNode &lhs, const SchedDagNode &rhs) {
    return DenseMapInfo<int32_t>::isEqual(lhs.id, rhs.id);
  }
};

// Format: [@nodeId opName p={parent nodes} c={child nodes}]
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDagNode &node) {
  out << "[@" << node.id << " " << node.opStr;
  out << " p={";
  for (auto p : node.getParents()) {
    out << "@" << p->id << " ";
  }
  out << "} c={";
  for (auto c : node.getChildren()) {
    out << "@" << c->id << " ";
  }
  out << "} pr={";
  for (uint32_t i = 0;
       i < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++i) {
    SchedDagNodePriorityType type = static_cast<SchedDagNodePriorityType>(i);
    if (node.hasPriority(type)) {
      out << node.getPriority(type) << " ";
    } else {
      out << "? ";
    }
  }
  out << "}]";
  return out;
}

using SchedDagNodeList = SmallVector<SchedDagNode *>;
// Format NodeList.
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDagNodeList &nodes) {
  for (auto node : nodes) {
    out << *node << "\n";
  }
  return out;
}
llvm::raw_ostream &
operator<<(llvm::raw_ostream &out,
           llvm::SmallVector<std::shared_ptr<SchedDagNode>> &nodes) {
  for (auto node : nodes) {
    out << *(node.get()) << "\n";
  }
  return out;
}

/******************************************************************************
  Categories of ops to facilitate scheduling.
******************************************************************************/
bool nodeCategoryDot(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::DotOp>(op);
}

bool nodeCategoryLocalLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::gpu::LocalLoadOp>(op);
}

bool nodeCategoryLocalStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::gpu::LocalStoreOp>(op);
}

bool nodeCategoryLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::LoadOp, triton::gpu::LocalLoadOp,
                   triton::amdgpu::BufferLoadOp>(op);
}

bool nodeCategoryStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::StoreOp, triton::gpu::LocalStoreOp>(op);
}

bool nodeCategoryMem(SchedDagNode *node) {
  return nodeCategoryLoad(node) || nodeCategoryStore(node);
}

bool nodeCategoryGlobalLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(op);
}

bool nodeCategoryGlobalStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::StoreOp>(op);
}

bool nodeCategoryGlobal(SchedDagNode *node) {
  Operation *op = node->getOp();
  return nodeCategoryGlobalLoad(node) || nodeCategoryGlobalStore(node);
}

bool nodeCategoryNop(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::gpu::MemDescSubviewOp, triton::gpu::MemDescTransOp,
                   tt::amdgpu::ExtractSliceOp, tt::amdgpu::ConcatOp>(op);
}

bool nodeCategoryBarrier(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<mlir::gpu::BarrierOp, ROCDL::SchedBarrier, ROCDL::SetPrioOp>(
      op);
}

std::string getNodeColor(SchedDagNode *node) {
  Operation *op = node->getOp();
  if (llvm::isa<DotOp>(op)) {
    return "deepskyblue";
  } else if (llvm::isa<triton::gpu::LocalLoadOp>(op)) {
    return "yellow";
  } else if (llvm::isa<triton::gpu::LocalStoreOp>(op)) {
    return "orange";
  } else if (nodeCategoryGlobalLoad(node)) {
    return "red";
  } else if (nodeCategoryGlobalStore(node)) {
    return "green";
  } else if (nodeCategoryBarrier(node)) {
    return "magenta";
  } else if (nodeCategoryNop(node)) {
    return "none";
  } else {
    return "gray80";
  }
}

using OpNodeMap = llvm::MapVector<Operation *, SchedDagNode *>;

struct SchedDep {
  SchedDagNode *parent;
  SchedDagNode *child;

  SchedDep() = default;
  SchedDep(SchedDagNode *p, SchedDagNode *c) : parent(p), child(c) {}

  llvm::raw_ostream &dump(llvm::raw_ostream &out) const {
    out << "p=" << parent->id << " <- c=" << child->id;
    return out;
  }

}; // SchedDep

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const SchedDep &dep) {
  return dep.dump(out);
}

/*
  Need to compare SchedDeps based on contents of the SchedDagNode
  and not just based he pointers to the nodes.
  However, the empty and tombstone deps need to have
  dummy pointers which can't be dereferenced.
  Therefore the comparison operations need to first check
  if the pointers are empty/tombstone before dereferencing.
*/
struct SchedDepDenseMapInfo : llvm::DenseMapInfo<SchedDep> {

  // These represent additional nodes are illegal to dereference.
  static const SchedDagNode *emptyNode;
  static const SchedDagNode *tombstoneNode;

  static inline SchedDep getEmptyKey() {
    return SchedDep(DenseMapInfo<SchedDagNode *>::getEmptyKey(),
                    DenseMapInfo<SchedDagNode *>::getEmptyKey());
  }
  static inline SchedDep getTombstoneKey() {
    return SchedDep(DenseMapInfo<SchedDagNode *>::getTombstoneKey(),
                    DenseMapInfo<SchedDagNode *>::getTombstoneKey());
  }
  // Hash parent and child ids.
  // can I de-reference d.parent, it will it sometimes be empty or tombstone
  // key?
  static unsigned getHashValue(const SchedDep &d) {
    return llvm::detail::combineHashValue(
        SchedDagNodeDenseMapInfo::getHashValue(*d.parent),
        SchedDagNodeDenseMapInfo::getHashValue(*d.child));
  }

  static bool isEqual(const SchedDagNode *lhs, const SchedDagNode *rhs) {
    if (lhs == emptyNode) {
      if (rhs == emptyNode)
        return true;
      return false;
    }
    // know lhs not empty
    if (lhs == tombstoneNode) {
      if (rhs == tombstoneNode)
        return true;
      return false;
    }
    // know lhs not empty nor tombstone
    if (rhs == emptyNode || rhs == tombstoneNode) {
      return false;
    }
    // know neither lhs nor rhs are empty nor tombstone
    // now it is safe to dereference them.
    return SchedDagNodeDenseMapInfo::isEqual(*lhs, *rhs);
  }

  // Equal if parent and child ids are equal.
  static bool isEqual(const SchedDep &lhs, const SchedDep &rhs) {
    return isEqual(lhs.parent, rhs.parent) && isEqual(lhs.child, rhs.child);
  }
};
const SchedDagNode *SchedDepDenseMapInfo::emptyNode =
    DenseMapInfo<SchedDagNode *>::getEmptyKey();
const SchedDagNode *SchedDepDenseMapInfo::tombstoneNode =
    DenseMapInfo<SchedDagNode *>::getTombstoneKey();

using DepSet = DenseSet<SchedDep, SchedDepDenseMapInfo>;
using DepMap = DenseMap<StringRef, DepSet>;

/******************************************************************************
  SchedDag consists of nodes and deps.
  Because the scheduling process will remove deps from nodes,
  there are 2 copies of dependencies;
  one is on the nodes themselves (parents, children),
  the other in is DepMap.
  After scheduling, the dependencies are restored to the nodes.
******************************************************************************/
struct SchedDag {

  SchedDag(Block *block) {
    // Create a new SchedDag.
    for (auto it = block->begin(); it != block->end(); ++it) {
      Operation *op = &(*it);
      addOp(op);
    }
  }

  // Shallow copy constructor.
  SchedDag(const SchedDag &dag)
      : nodesHeap(dag.nodesHeap), nodeList(dag.nodeList), deps(dag.deps),
        nodeMap(dag.nodeMap) {
    LDBG("SchedDag::CopyConstructor(shallow)");
  }

  void addOp(Operation *op) {
    std::shared_ptr<SchedDagNode> node = std::make_shared<SchedDagNode>(op);
    nodeMap.insert({op, node.get()});
    nodeList.push_back(node.get());
    nodesHeap.push_back(std::move(node));
  }

  void addDeps(StringRef depTypeName, const DepSet &depSet) {
    deps[depTypeName] = depSet;
    applyDeps(depTypeName);
  }

  void applyDeps() {
    for (auto &depType : deps) {
      StringRef depTypeName = depType.getFirst();
      DepSet &depSet = depType.getSecond();
      for (auto dep : depSet) {
        dep.child->addParent(dep.parent);
        dep.parent->addChild(dep.child);
      }
    }
  }

  void applyDeps(StringRef depTypeName) {
    DepSet &depSet = deps[depTypeName];
    for (auto dep : depSet) {
      dep.child->addParent(dep.parent);
      dep.parent->addChild(dep.child);
    }
  }

  void clearDeps() {
    for (auto *node : nodeList) {
      node->clearDeps();
    }
  }

  void resetDeps() {
    clearDeps();
    applyDeps();
  }

  // Removes dep from all depTypes
  int32_t removeDep(const SchedDep &dep) {
    int32_t count = 0;
    for (auto &depType : deps) {
      StringRef depTypeName = depType.getFirst();
      DepSet &depSet = depType.getSecond();
      int32_t erased = depSet.erase(dep);
      count += erased;
    }
    return count;
  }

  // Remove node from nodeList and remove deps from nodes.
  // Leaves heapNodes alone.
  void removeNodeAndDeps(SchedDagNode *node) {
    for (auto child : node->getChildren()) {
      child->removeParent(node);
      int32_t n = removeDep(SchedDep(node, child));
    }
    for (auto parent : node->getParents()) {
      parent->removeChild(node);
      int32_t n = removeDep(SchedDep(parent, node));
    }
    readyNodes.remove(node);
    for (auto it = nodeList.begin(); it != nodeList.end(); it++) {
      SchedDagNode *n = *it;
      if (n == node) {
        nodeList.erase(it, it + 1);
        return;
      }
    }
  }

  /*
    Same as removeNodeRemoveDeps, except convey depenencies.
    Before there are 3 parents and 3 children.
    p0 p1 p2
     \ | /
      node
     / | \
    c0 c1 c2

    After there are 9 dependencies.
    p0 p1 p2
     \ | /
       *
     / | \
    c0 c1 c2
  */
  void removeNodeCascadeDeps(SchedDagNode *node) {
    // Add new dependencies first.
    for (auto parent : node->getParents()) {
      for (auto child : node->getChildren()) {
        parent->addChild(child);
        child->addParent(parent);
        // Add this dep to other list.
        SchedDep dep(parent, child);
        deps["Other"].insert(dep);
      }
    }
    // Remove node and old dependencies.
    removeNodeAndDeps(node);
  }

  template <SchedDirection Direction> void initReadyNodes() {
    readyNodes.clear();
    for (auto *node : nodeList) {
      if (node->isReady<Direction>()) {
        readyNodes.insert(node);
      }
    }
  }

  bool finished() { return readyNodes.empty(); }

  /*
    After scheduling a node top-down, mark it's children as
    dependency-fulfilled. After scheduling a node bottom-up, mark it's parents
    as dependency-fulfilled.
  */
  template <SchedDirection Direction>
  void removeScheduledNode(SchedDagNode *node) {
    assert(node->isReady<Direction>());
    readyNodes.remove(node);

    if constexpr (Direction == SchedDirection::TopDown) {
      for (auto child : node->getChildren()) {
        child->removeParent(node);
        if (child->isReady<Direction>()) {
          readyNodes.insert(child);
        }
      }
    } else {
      for (auto parent : node->getParents()) {
        parent->removeChild(node);
        if (parent->isReady<Direction>()) {
          readyNodes.insert(parent);
        }
      }
    }
  }

  SetVector<SchedDagNode *> &getReadyNodes() { return readyNodes; }

  void dumpNodes(llvm::raw_ostream &out) {
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      auto node = *it;
      out << *node << "\n";
    }
  }

  void dumpDeps(llvm::raw_ostream &out) {
    for (auto &depType : deps) {
      StringRef depTypeName = depType.getFirst();
      DepSet &depSet = depType.getSecond();
      for (auto dep : depSet) {
        out << depTypeName << ": " << dep << "\n";
      }
    }
  }

  llvm::raw_ostream &dumpDotFormat(llvm::raw_ostream &out) {
    out << "digraph \"dep-dag\" {\n";
    out << "rankdir=\"BT\"\n";

    // Dump nodes.
    out << "\n// Op nodes.\n";
    int32_t numRefined = 0;
    SchedDagNode *firstRefined = nullptr;
    for (auto node : nodeList) {
      Operation *op = node->getOp();
      std::string color = getNodeColor(node);
      std::string addr = std::to_string(reinterpret_cast<intptr_t>(node));
      out << addr << "\t[label=\"[@" << node->id << " " << node->opStr << "]\""
          << ", style=filled, fillcolor=" << color << "]\n";

      // Count refined ops.
      if (isa<DotOp>(op) &&
          op->hasAttr(triton::amdgpu::RefinedOpAttr::getMnemonic())) {
        if (!firstRefined) {
          firstRefined = node;
        }
      }
    }

    // Dump refined-ops subgraphs; dots only.
    if (firstRefined && false) {
      out << "\n// Clusters for refined dots.\n";
      int32_t serial = 0;
      int32_t prevUnrefinedId = -1;
      out << "\nsubgraph ref_" << serial << " {\n";
      out << "  cluster=true;\n";
      out << "  color = \"" << getNodeColor(firstRefined) << "\";\n";
      out << "  label = \"refined[" << serial << "]\";\n  ";

      for (auto node : nodeList) {
        Operation *op = node->getOp();
        if (isa<DotOp>(op) &&
            op->hasAttr(triton::amdgpu::RefinedOpAttr::getMnemonic())) {
          auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpAttr>(
              triton::amdgpu::RefinedOpAttr::getMnemonic());
          int32_t idUnrefinedOp = attr.getIdUnrefinedOp();
          if (idUnrefinedOp == prevUnrefinedId || prevUnrefinedId < 0) {
            // Same unrefined op.
            std::string addr = std::to_string(reinterpret_cast<intptr_t>(node));
            out << "\"" << addr << "\" ";
          } else {
            // New unrefined op.
            // Close previous subgraph.
            out << ";\n}\n\n";

            // Begin next subgraph.
            serial++;
            out << "\nsubgraph ref_" << serial << " {\n";
            out << "  cluster=true;\n";
            out << "  color = \"" << getNodeColor(node) << "\";\n";
            out << "  label = \"refined[" << serial << "]\";\n  ";
            std::string addr = std::to_string(reinterpret_cast<intptr_t>(node));
            out << "\"" << addr << "\" ";
          }
          prevUnrefinedId = idUnrefinedOp;
        }
      }
      out << ";\n}\n\n";
    }

    std::map<StringRef, std::pair<std::string, std::string>> format;
    // Assume later deps are more important to visualize b/c complex.
    format["Barrier"] = std::make_pair("gray90", "dotted");
    format["RefinedOrder"] = std::make_pair("gray50", "dotted");
    format["Data"] = std::make_pair("black", "dotted");
    format["LocalLoadTypeOrder"] = std::make_pair("darkgreen", "solid");
    format["LocalStoreTypeOrder"] = std::make_pair("darkgreen", "solid");
    format["GlobalLoadCategoryOrder"] = std::make_pair("darkgreen", "solid");
    format["MemOrder"] = std::make_pair("blue", "solid");
    format["MemInterleave"] = std::make_pair("red", "solid");

    for (auto &depType : deps) {
      StringRef depTypeName = depType.getFirst();
      DepSet &depSet = depType.getSecond();
      std::string color = "black";
      std::string style = "solid";
      if (format.find(depTypeName) != format.end()) {
        color = format[depTypeName].first;
        style = format[depTypeName].second;
      }
      out << "\n// DepType: " << depTypeName << ".\n";
      for (auto dep : depSet) {
        std::string parentAddr =
            std::to_string(reinterpret_cast<intptr_t>(dep.parent));
        std::string childAddr =
            std::to_string(reinterpret_cast<intptr_t>(dep.child));
        out << childAddr << " -> " << parentAddr << " [color=" << color
            << ", style=" << style << "]\n";
      }
    }
    out << "}\n";
    return out;
  }

  // SchedDagNodes as shared_ptrs for dealloc.
  llvm::SmallVector<std::shared_ptr<SchedDagNode>> nodesHeap;
  // SchedDagNodes as simple ptrs for everything else.
  SchedDagNodeList nodeList;
  DepMap deps;
  OpNodeMap nodeMap;
  SetVector<SchedDagNode *> readyNodes;
};

/******************************************************************************
  Each DependencyCalculator gets to see the current nodeList
  as well as all previously applied deps.
******************************************************************************/
struct DependencyCalculator {
  DependencyCalculator(StringRef name) : depTypeName(name) {}

  virtual void calcDeps() = 0;

  void addDepsToDag(SchedDag *d) {
    LDBG("DependencyCalculator<" << depTypeName << ">::addDepsToDag()");
    dag = d;
    depSet.clear();
    // Populates depSet.
    calcDeps();
    dag->addDeps(depTypeName, depSet);
  }
  virtual ~DependencyCalculator() = default;

  StringRef depTypeName;
  SchedDag *dag;
  DepSet depSet;
};

/******************************************************************************
  Create data dependencies based on def-use chains.
  Also creates dependencies based on various barriers.
  This class represents the minimum set of dependencies needed for correctness.
******************************************************************************/
struct DataDependencyCalculator : DependencyCalculator {
  DataDependencyCalculator() : DependencyCalculator("Data") {}

  // Add data deps for operands.
  void calcDeps() {
    for (auto it = dag->nodeList.begin(); it != dag->nodeList.end(); ++it) {
      SchedDagNode *node = (*it);
      for (auto operandValue : node->getOp()->getOperands()) {
        auto operandDefOp = operandValue.getDefiningOp();
        SchedDagNode *parentNode = dag->nodeMap[operandDefOp];
        if (parentNode) {
          SchedDep dep;
          dep.parent = parentNode;
          dep.child = node;
          depSet.insert(dep);
        }
      }
    }
  }
};

/******************************************************************************
  Creates dependencies based on various barriers.
  TODO(dtanner) need to add support for below?
  triton::gpu::AsyncWaitOp
  triton::nvidia_gpu::TMAStoreWaitOp
  triton::nvidia_gpu::ArriveBarrierOp
  RegionBranchOpInterface ?
  MemoryEffects::Write
  MemoryEffects::Read
  triton::CallOp
  triton::gpu::LocalAllocOp
  triton::gpu::LocalDeallocOp
******************************************************************************/
struct BarrierDependencyCalculator : DependencyCalculator {
  BarrierDependencyCalculator() : DependencyCalculator("Barrier") {}

  // There is an implied data dependency between LDS ops and GPUBarrier.
  template <SchedDirection Direction> void calcDepsLdsGpuBar() {

    auto fwIt = dag->nodeList.begin();
    auto bkIt = dag->nodeList.rbegin();
    auto next = [&]() -> SchedDagNode * {
      if constexpr (Direction == SchedDirection::TopDown) {
        if (fwIt == dag->nodeList.end())
          return nullptr;
        return *(fwIt++);
      }
      if constexpr (Direction == SchedDirection::BottomUp) {
        if (bkIt == dag->nodeList.rend())
          return nullptr;
        return *(bkIt++);
      }
      return nullptr;
    };

    llvm::SmallVector<SchedDagNode *> ldsOpsNodes;
    while (SchedDagNode *node = next()) {
      auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(node->getOp());
      auto localStore = dyn_cast<triton::gpu::LocalStoreOp>(node->getOp());
      auto localAlloc = dyn_cast<triton::gpu::LocalAllocOp>(node->getOp());
      if (localLoad || localStore || localAlloc) {
        ldsOpsNodes.push_back(node);
      }
      auto gpuBarrier = dyn_cast<mlir::gpu::BarrierOp>(node->getOp());
      if (gpuBarrier) {
        SchedDagNode *barrierNode = node;
        for (auto ldsOpNode : ldsOpsNodes) {
          if constexpr (Direction == SchedDirection::TopDown) {
            SchedDep dep;
            dep.parent = ldsOpNode;
            dep.child = barrierNode;
            depSet.insert(dep);
          }
          if constexpr (Direction == SchedDirection::BottomUp) {
            SchedDep dep;
            dep.parent = barrierNode;
            dep.child = ldsOpNode;
            depSet.insert(dep);
          }
        }
        ldsOpsNodes.clear();
      }
    }
  }

  // GpuBar can't be reordered across themselves.
  // While this may be logically superfluous, it's fine to leave it for clarity.
  void calcDepsGpuBarGpuBar() {
    SchedDagNode *prevBar = nullptr;
    for (auto it = dag->nodeList.begin(); it != dag->nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      auto bar = dyn_cast<mlir::gpu::BarrierOp>(node->getOp());
      if (bar) {
        if (prevBar) {
          SchedDep dep;
          dep.parent = prevBar;
          dep.child = node;
          depSet.insert(dep);
        }
        prevBar = node;
      }
    }
  }

  // Nodes without results still must come before cf.br.
  void calcDepsCfBr() {
    SchedDagNode *lastNode = (*(dag->nodeList.rbegin()));
    for (auto it = std::next(dag->nodeList.rbegin());
         it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = (*it);
      if (node->getOp()->getNumResults() == 0) {
        SchedDep dep;
        dep.parent = node;
        dep.child = lastNode;
        depSet.insert(dep);
      }
    }
  }

  // Which ops allowed to cross the barrier.
  enum SchedBarOpType : int32_t {
    None = 0, // No ops can cross barrier.
    All = 1,  // All, non-memory, non-side-effect producing
    Valu = 2,
    Salu = 4,
    Mfma = 8,
    VmemAll = 16,
    VmemRead = 32,
    VmemWrite = 64,
    LdsAll = 128,
    LdsRead = 256,
    LdsWrite = 512,
    Trans = 1024,
  };

  // Return true if node matches sched bar op type;
  // doing so means this op is allowed to cross the barrier.
  // Returning false means this nop isn't allowed to cross the barrier.
  bool isaSchedBarOpType(SchedDagNode *node, SchedBarOpType sbTy) {

    if (sbTy == SchedBarOpType::None) {
      return false;
    }

    // Mfma
    if (isa<triton::DotOp>(node->getOp())) {
      return (sbTy == SchedBarOpType::All || sbTy == SchedBarOpType::Mfma);

      // Lds Read
    } else if (isa<triton::gpu::LocalLoadOp>(node->getOp())) {
      return (sbTy == SchedBarOpType::LdsRead ||
              sbTy == SchedBarOpType::LdsAll);

      // Lds Write
    } else if (isa<triton::gpu::LocalStoreOp, triton::gpu::LocalAllocOp>(
                   node->getOp())) {
      return (sbTy == SchedBarOpType::LdsWrite ||
              sbTy == SchedBarOpType::LdsAll);

      // Global Load
    } else if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(
                   node->getOp())) {
      return (sbTy == SchedBarOpType::VmemRead ||
              sbTy == SchedBarOpType::VmemAll);

      // Global Store
    } else if (isa<triton::StoreOp, triton::amdgpu::BufferStoreOp>(
                   node->getOp())) {
      return (sbTy == SchedBarOpType::VmemWrite ||
              sbTy == SchedBarOpType::VmemAll);

      // Transcendental
    } else if (isa<math::ExpOp, math::Exp2Op>(node->getOp())) {
      return (sbTy == SchedBarOpType::All || sbTy == SchedBarOpType::Valu ||
              sbTy == SchedBarOpType::Salu || sbTy == SchedBarOpType::Trans);

      // Alu
    } else if (isa<math::SqrtOp, math::RsqrtOp, arith::DivFOp, triton::ReduceOp,
                   arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp,
                   arith::SIToFPOp, triton::FpToFpOp, triton::PreciseSqrtOp,
                   math::SqrtOp, arith::SubIOp, arith::AddIOp, arith::MulIOp,
                   arith::DivSIOp, arith::DivUIOp, arith::RemFOp,
                   arith::RemSIOp, arith::RemUIOp, arith::AndIOp, arith::OrIOp,
                   arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
                   arith::MinNumFOp, arith::MaxNumFOp, arith::MinSIOp,
                   arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp,
                   arith::AddFOp, arith::SubFOp, arith::MulFOp,
                   arith::MaximumFOp, arith::MinimumFOp,
                   triton::gpu::ConvertLayoutOp>(node->getOp())) {
      return (sbTy == SchedBarOpType::All || sbTy == SchedBarOpType::Valu ||
              sbTy == SchedBarOpType::Salu);

    } else {
      // Unrecognized ops, assume they're simple alu.
      return (sbTy == SchedBarOpType::All || sbTy == SchedBarOpType::Valu ||
              sbTy == SchedBarOpType::Salu);
    }
  }

  // Returns true if there are any sched.barriers
  // non-zero masks; more complicated to create barriers.
  bool hasSchedBarMasks() {
    for (auto node : dag->nodeList) {
      Operation *op = node->getOp();
      if (isa<ROCDL::SchedBarrier>(node->getOp())) {
        IntegerAttr maskAttr = op->getAttrOfType<IntegerAttr>("mask");
        int32_t mask = maskAttr.getInt();
        if (mask != 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Sched.bars block ops based on type.
  void calcDepsSchedBarMasks() {
    // For each node and type, track which nodes don't match the type.
    // This means any sched.bar, for each bit in the mask,
    // create deps between the bit
    DenseMap<SchedBarOpType, SchedDagNodeList> visitedNodes;
    DenseMap<SchedBarOpType, SchedDagNode *> visitedBarriers;

    visitedNodes[SchedBarOpType::None] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::All] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::Valu] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::Salu] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::Mfma] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::VmemAll] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::VmemRead] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::VmemWrite] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::LdsAll] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::LdsRead] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::LdsWrite] = SchedDagNodeList();
    visitedNodes[SchedBarOpType::Trans] = SchedDagNodeList();
    // Visit every node top-down.
    for (auto node : dag->nodeList) {
      LDBG("Visiting " << *node);
      Operation *op = node->getOp();
      if (isa<ROCDL::SchedBarrier>(node->getOp())) {
        IntegerAttr maskAttr = op->getAttrOfType<IntegerAttr>("mask");
        int32_t mask = maskAttr.getInt();
        LDBG("Found sched.barrier w/ mask=" << mask);
        // Add deps for all prev matching nodes before sched.bar.
        for (auto entry : visitedNodes) {
          SchedBarOpType sbType = entry.getFirst();
          SchedDagNodeList visitedList = entry.getSecond();
          if (!(mask & sbType)) {
            LDBG("Adding deps for sbType=" << sbType);
            for (auto v : visitedNodes[sbType]) {
              SchedDep schedDep;
              schedDep.parent = v;
              schedDep.child = node;
              depSet.insert(schedDep);
            }
            // Make this barrier the only thing in this visited list.
            visitedNodes[sbType].clear();
            visitedNodes[sbType].push_back(node);
          }
        }
        // Update this as most recent barrier visited.
        for (auto entry : visitedBarriers) {
          SchedBarOpType sbType = entry.getFirst();
          if (!(mask & sbType)) {
            visitedBarriers[sbType] = node;
          }
        }
      } else {
        // Non Sched.Barrier
        // Add op to every matching visiting list.
        for (auto entry : visitedNodes) {
          SchedBarOpType sbType = entry.getFirst();
          if (!isaSchedBarOpType(node, sbType)) {
            visitedNodes[sbType].push_back(node);
          }
        }
        // Add dep for node after prev sched.bar.
        for (auto entry : visitedBarriers) {
          SchedBarOpType sbType = entry.getFirst();
          SchedDagNode *prevBar = entry.getSecond();
          if (prevBar) {
            if (!isaSchedBarOpType(node, sbType)) {
              SchedDep schedDep;
              schedDep.parent = prevBar;
              schedDep.child = node;
              depSet.insert(schedDep);
            }
          }
        }
      }
    }
    // Now that we went through the list top-down, we need to go from
    // the last sched.bar to the last region.
  }

  // Interpret any OpType as full scheduling barrier that no ops can cross.
  void calcDepsFullBars() {
    SchedDagNode *prevBar = nullptr;
    SchedDagNodeList prevNodes;

    for (auto node : dag->nodeList) {
      if (isa<ROCDL::SchedBarrier, ROCDL::SetPrioOp>(node->getOp())) {
        // prevNodes must be before this barrier.
        for (auto p : prevNodes) {
          SchedDep schedDep;
          schedDep.parent = p;
          schedDep.child = node;
          depSet.insert(schedDep);
        }
        // Barrier becomes only previous node.
        prevNodes.clear();
        prevNodes.push_back(node);
        prevBar = node;
      } else {
        prevNodes.push_back(node);
        if (prevBar) {
          SchedDep schedDep;
          schedDep.parent = prevBar;
          schedDep.child = node;
          depSet.insert(schedDep);
        }
      }
    }
  }

  void calcDeps() {
    calcDepsLdsGpuBar<SchedDirection::BottomUp>();
    calcDepsLdsGpuBar<SchedDirection::TopDown>();
    calcDepsGpuBarGpuBar();
    calcDepsCfBr();
    calcDepsFullBars();
    if (hasSchedBarMasks()) {
      calcDepsSchedBarMasks();
    }
  }
};

/******************************************************************************
  Create order dependencies between ops which were refined from same original
  op.
******************************************************************************/
struct RefinedOpDependencyCalculator : DependencyCalculator {
  RefinedOpDependencyCalculator() : DependencyCalculator("RefinedOrder") {}

  /*
    Assume all dots are in ideal order, due to user placing unrefined dots
    in ideal order, and refinement creating refined dots in ideal order.
    Therefore the dot order before any rescheduling is a source of truth.
    Deps were already created between refined dots, here we place deps
    between the unrefined ops.
  */
  void calcDepsDot() {
    SchedDagNode *prevDot = nullptr;
    int32_t prevId = -1;
    for (auto it = dag->nodeList.begin(); it != dag->nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      if (DotOp op = dyn_cast<DotOp>(node->getOp())) {
        if (auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpAttr>(
                triton::amdgpu::RefinedOpAttr::getMnemonic())) {
          int32_t id = attr.getIdUnrefinedOp();
          if (prevDot && id != prevId) {
            SchedDep dep;
            dep.parent = prevDot;
            dep.child = node;
            depSet.insert(dep);
          }
          prevDot = node;
          prevId = id;
        }
      }
    }
  }

  /*
    Add dependencies between ops which were refined from the same op.
    It is assumed that the ideal relative order of refines ops is the order
    created during refinement.
  */
  void calcDepsRefinedOp() {
    SchedDagNode *prevNode = nullptr;
    int32_t prevId = -1;
    for (auto it = dag->nodeList.begin(); it != dag->nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      Operation *op = node->getOp();
      if (auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpAttr>(
              triton::amdgpu::RefinedOpAttr::getMnemonic())) {
        int32_t id = attr.getIdUnrefinedOp();
        if (prevNode && id == prevId) {
          SchedDep dep;
          dep.parent = prevNode;
          dep.child = node;
          depSet.insert(dep);
        }
        prevNode = node;
        prevId = id;
      }
    }
  }

  void calcDeps() {
    calcDepsRefinedOp();
    calcDepsDot();
  }
};

/*
  Likely don't want this as it incorrectly ordered GL1 LS1 GL0 LS0
  And enforced the order with dependencies.
  Keeping around in case want to adapt the logic to something different later.
*/
struct PriorityOrderDependencyCalculator : DependencyCalculator {
  PriorityOrderDependencyCalculator() : DependencyCalculator("PriorityOrder") {}
  void calcDeps() {

    for (uint32_t priorityIdx = 0;
         priorityIdx < static_cast<uint32_t>(SchedDagNodePriorityType::Size);
         ++priorityIdx) {

      SchedDagNodePriorityType priorityType =
          static_cast<SchedDagNodePriorityType>(priorityIdx);
      SchedDagNodePriorityDataType prevPriority = schedDagNodePriorityUnset;
      SchedDagNodeList prevNodes;
      SchedDagNodePriorityDataType currPriority = schedDagNodePriorityUnset;
      SchedDagNodeList currNodes;

      for (auto node : dag->nodeList) {
        if (node->hasPriority(priorityType)) {
          auto nodePriority = node->getPriority(priorityType);
          if (currPriority == schedDagNodePriorityUnset) {
            currPriority = currPriority;
          }
          if (nodePriority == currPriority) {
            // We're still on the same priority, add deps and node.
            for (auto n : prevNodes) {
              SchedDep dep;
              dep.parent = n;
              dep.child = node;
              depSet.insert(dep);
            }
            currNodes.push_back(node);
          } else { // maintain order even when we dropped priority b/c other
                   // deps.
            // We're on a new priority; add all deps between prev and curr.
            for (auto p : prevNodes) {
              for (auto c : currNodes) {
                SchedDep dep;
                dep.parent = p;
                dep.child = c;
                depSet.insert(dep);
              }
            }
            // Shift sets forward.
            prevPriority = currPriority;
            prevNodes = currNodes;
            currNodes.clear();
            currNodes.push_back(node);
            currPriority = nodePriority;
          }
        }
      }
      // Done with all nodes; add last group of deps.
      for (auto p : prevNodes) {
        for (auto c : currNodes) {
          SchedDep dep;
          dep.parent = p;
          dep.child = c;
          depSet.insert(dep);
        }
      }
    }
  }
};

/*
  Simple DependencyCalculators based on op types (or category) and serializes
  all ops of that type.
*/
template <typename OpType>
void calcDepsOpType(SchedDagNodeList *nodeList, DepSet &depSet) {
  SchedDagNode *prevNode = nullptr;
  int32_t prevId = -1;
  for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
    SchedDagNode *node = *it;
    if (llvm::isa<OpType>(node->getOp())) {
      if (prevNode) {
        SchedDep dep;
        dep.parent = prevNode;
        dep.child = node;
        depSet.insert(dep);
      }
      prevNode = node;
    }
  }
}

void calcDepsOpCategory(SchedDagNodeList *nodeList, DepSet &depSet,
                        std::function<bool(SchedDagNode *)> category) {
  SchedDagNode *prevNode = nullptr;
  int32_t prevId = -1;
  for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
    SchedDagNode *node = *it;
    if (category(node)) {
      if (prevNode) {
        SchedDep dep;
        dep.parent = prevNode;
        dep.child = node;
        depSet.insert(dep);
      }
      prevNode = node;
    }
  }
}

struct DotOrderDependencyCalculator : DependencyCalculator {
  DotOrderDependencyCalculator() : DependencyCalculator("DotTypeOrder") {}
  void calcDeps() { calcDepsOpType<triton::DotOp>(&dag->nodeList, depSet); }
};

struct LocalLoadOrderDependencyCalculator : DependencyCalculator {
  LocalLoadOrderDependencyCalculator()
      : DependencyCalculator("LocalLoadTypeOrder") {}
  void calcDeps() {
    calcDepsOpType<triton::gpu::LocalLoadOp>(&dag->nodeList, depSet);
  }
};

struct LocalStoreOrderDependencyCalculator : DependencyCalculator {
  LocalStoreOrderDependencyCalculator()
      : DependencyCalculator("LocalStoreTypeOrder") {}
  void calcDeps() {
    calcDepsOpType<triton::gpu::LocalStoreOp>(&dag->nodeList, depSet);
  }
};

struct GlobalLoadOrderDependencyCalculator : DependencyCalculator {
  GlobalLoadOrderDependencyCalculator()
      : DependencyCalculator("GlobalLoadCategoryOrder") {}
  void calcDeps() {
    calcDepsOpCategory(&dag->nodeList, depSet, nodeCategoryGlobalLoad);
  }
};

/******************************************************************************
  Create order dependencies between ops which were refined from same original
  op.
******************************************************************************/
struct DotLdsOrderDependencyCalculator : DependencyCalculator {
  DotLdsOrderDependencyCalculator() : DependencyCalculator("DotLdsOrder") {}

  void calcDeps() {
    SchedDagNode *prevDot = nullptr;
    SchedDagNode *prevLds = nullptr;

    for (auto node : dag->nodeList) {
      if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(
              node->getOp())) {
        if (prevDot) {
          SchedDep dep;
          dep.parent = prevDot;
          dep.child = node;
          depSet.insert(dep);
        }
        prevLds = node;
      } else if (isa<triton::DotOp>(node->getOp())) {
        if (prevLds) {
          SchedDep dep;
          dep.parent = prevLds;
          dep.child = node;
          depSet.insert(dep);
        }
        prevDot = node;
      }
    }
  }
};

/******************************************************************************
  For kernels with increasing prefetching, there will be many memory
  ops which can be re-ordered. This begins the analysis of what ordering
  still need to be figured out.
  E.g., for FlashAttention how to order the GL, LS for K and V.
******************************************************************************/
struct MemOrderDependencyCalculator : DependencyCalculator {
  MemOrderDependencyCalculator() : DependencyCalculator("MemOrder") {}

  // get memory ops only from the graph; keep them in order.
  void createMemDag(SchedDag *memDag) const {

    LDBG("Removing non-mem nodes.");
    SchedDagNodeList listCopy = memDag->nodeList;
    for (SchedDagNode *node : listCopy) {
      if (!nodeCategoryMem(node) && !isa<triton::DotOp>(node->op)) {
        memDag->removeNodeCascadeDeps(node);
      }
    }
    LDBG("Removing non-unique refinement ids.");

    // We are correctly removing the nodes, but some dependencies are staying in
    // the graph.
    SetVector<int32_t> refinedIds;
    listCopy = memDag->nodeList;
    for (SchedDagNode *node : listCopy) {
      Operation *op = node->getOp();
      if (auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpAttr>(
              triton::amdgpu::RefinedOpAttr::getMnemonic())) {
        int32_t id = attr.getIdUnrefinedOp();
        if (refinedIds.contains(id)) {
          memDag->removeNodeCascadeDeps(node);
        } else {
          refinedIds.insert(id);
        }
      } else {
        LDBG("WARNING memory op has no RefineOpAttr: " << *op);
      }
    }
  }

  /*
    Determine which memory ops can be co-scheduled
    with which other memory ops, or need a strict order.
  */
  void calcDeps() {
    LLVM_DEBUG(dag->dumpDeps(llvm::dbgs()));

    SchedDag memDag = *dag;
    createMemDag(&memDag);

    LDBG("Simplified Graph of MemNodes");
    LLVM_DEBUG(memDag.dumpNodes(llvm::dbgs()));
    LLVM_DEBUG(memDag.dumpDotFormat(llvm::dbgs()));
  }
};

/******************************************************************************
  Library of comparison functions for scheduling heuristics.
  prefer*() functions return true if a is preferred over b
  and if we want the preferred op.
  This is a strict a > b type of comparison, so they return false if a <= b.
  The parameter prefer is typically set to Direction==TopDown so that we can
  specify something like schedule LoadOp closer to top of loop,
  and these comparison will prefer the LoadOp for TopDown
  and dis-prefer it for BottomUp (which will push them toward top of loop).
  One exception is preferMachineState should have prefer=true since
  it already account for ideal machine state for the two different
  scheduling direction, i.e. at no point do we prefer a bad machine state.

  The findPreferred*() functions return a or b if one is strictly preferred
  over the other or nullptr if not. It is often the case that a==b
  for some of these comparisons, in which case the SchedHeuristic will
  then move on to the next comparison.
******************************************************************************/

// Prefer based on node priority.
bool preferPriority(SchedDagNode *a, SchedDagNode *b,
                    SchedDagNodePriorityType priorityType, bool prefer = true) {
  if (a->hasPriority(priorityType)) {
    if (b->hasPriority(priorityType)) {
      return (a->getPriority(priorityType) > b->getPriority(priorityType)) ==
             prefer;
    }
    return prefer;
  }
  if (b->hasPriority(priorityType)) {
    return !prefer;
  }
  return false;
}
SchedDagNode *findPreferredPriority(SchedDagNode *a, SchedDagNode *b,
                                    SchedDagNodePriorityType priorityType,
                                    bool prefer = true) {
  if (preferPriority(a, b, priorityType, prefer)) {
    return a;
  } else if (preferPriority(b, a, priorityType, prefer)) {
    return b;
  }
  return nullptr;
}

// Prefer based on op type.
template <typename OpType>
bool preferOpType(SchedDagNode *a, SchedDagNode *b, bool prefer = true) {
  return (llvm::isa<OpType>(a->getOp()) and !llvm::isa<OpType>(b->getOp())) ==
         prefer;
}
template <typename OpType>
SchedDagNode *findPreferredOpType(SchedDagNode *a, SchedDagNode *b,
                                  bool prefer = true) {
  if (preferOpType<OpType>(a, b, prefer)) {
    return a;
  } else if (preferOpType<OpType>(b, a, prefer)) {
    return b;
  }
  return nullptr;
}

// Prefer based on op category (functions).
bool preferOpCategory(SchedDagNode *a, SchedDagNode *b,
                      bool (*category)(SchedDagNode *), bool prefer = true) {
  return (category(a) and !category(b)) == prefer;
}
SchedDagNode *findPreferredOpCategory(SchedDagNode *a, SchedDagNode *b,
                                      bool (*category)(SchedDagNode *),
                                      bool prefer = true) {
  if (preferOpCategory(a, b, category, prefer)) {
    return a;
  } else if (preferOpCategory(b, a, category, prefer)) {
    return b;
  }
  return nullptr;
}

// Prefer based on machine state.
bool preferMachineState(SchedDagNode *a, SchedDagNode *b, MachineState *machine,
                        bool prefer = true) {
  return (machine->getCyclesUntilOpReady(a->getOp()) <
          machine->getCyclesUntilOpReady(b->getOp())) == prefer;
}
SchedDagNode *findPreferredMachineState(SchedDagNode *a, SchedDagNode *b,
                                        MachineState *machine,
                                        bool prefer = true) {
  if (preferMachineState(a, b, machine, prefer)) {
    return a;
  } else if (preferMachineState(b, a, machine, prefer)) {
    return b;
  }
  return nullptr;
}

// Returns ops in original mlir block order.
template <SchedDirection Direction>
SchedDagNode *getOriginalOrder(SchedDagNode *a, SchedDagNode *b) {
  return (a->id < b->id) == (Direction == SchedDirection::TopDown) ? a : b;
}

/******************************************************************************
  Each PriorityCalculator gets to see the current dag
  and set scheduling priorities to nodes.
******************************************************************************/
struct PriorityCalculator {
  PriorityCalculator(SchedDagNodePriorityType type) : priorityType(type) {}

  virtual void calcPriorities() = 0;

  void addPrioritiesToDag(SchedDag *d) {
    LDBG("PriorityCalculator<" << toString(priorityType)
                               << ">::addPrioritiesToDag()");
    dag = d;
    calcPriorities();
  }
  virtual ~PriorityCalculator() = default;

  SchedDagNodePriorityType priorityType;
  SchedDag *dag;
};

/******************************************************************************
  Add DotCriticalPath weights to nodes to correctly schedule.
  Priority = 1->N starting at the bottom of the block.
  Then priorities are propagated (unmodified) to parents and children.
  Therefore when scheduling TopDown, we know from all ops in the ready list
  which are needed to get to DotOP(priority=9) asap, and we schedule those
  first. Therefore we can schedule the critical path to the first DotOp.

  We only want to populate the dag with def/use Data dependencies
  to highlight the flow of data.
  Later we can add Barrier deps to the dag and re-propagate priorities
  to also get the flow of execution.

  Second, propagate priorities.
  LocalLoadOp a0 = 9 <- want to schedule asap.
  LocalLoadOp a1 = 6
  LocalLoadOp a2 = 3
  LocalLoadOp b0 = 9 <- want to schedule asap.
  LocalLoadOp b1 = 8
  LocalLoadOp b2 = 7

  First, assign priorities to DotOps.
        AB
  DotOp 00 = 9 <- want to schedule asap.
  DotOp 01 = 8
  DotOp 02 = 7
  DotOp 10 = 6
  DotOp 11 = 5
  DotOp 12 = 4
  DotOp 20 = 3
  DotOp 21 = 2
  DotOp 22 = 1
******************************************************************************/
struct DotCriticalPathPriorityCalculator : public PriorityCalculator {

  DotCriticalPathPriorityCalculator()
      : PriorityCalculator(SchedDagNodePriorityType::DotCriticalPath) {}

  void calcPriorities() {
    SchedDagNodePriorityDataType dotPriority = 1;
    // Prioritize DotOps.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::DotOp>(node->getOp())) {
        node->setPriority(SchedDagNodePriorityType::DotCriticalPath,
                          dotPriority);
        dotPriority += 1;
      }
    }

    // Propagate high priorities up for critical path.
    for (auto *node : dag->nodeList) {
      if (isa<triton::DotOp>(node->getOp())) {
        node->propagateHigherPriorityToParents(
            SchedDagNodePriorityType::DotCriticalPath);
      }
    }

    // Propagate low priorities down for critical path.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::DotOp>(node->getOp())) {
        node->propagateLowerPriorityToChildren(
            SchedDagNodePriorityType::DotCriticalPath);
      }
    }
  }
};

/******************************************************************************
  Ideally we want the DotCriticalPath to also be able to label the
  LocalStoreOps. However LDS semantics make this hard; do we have aliasing
  information so I can query which LocalStoreOps are needed for a LocalLoadOp.

  Since gpu.barriers enforce that all LocalLoadOps must complete,
  we can assume that LocalStoreOps are already correctly ordered relative to
  LocalLoadOps. Here we just want to continue the critical path to specify what
  is the optimal order of LocalStoreOps, and by consequence the optimal order
  of LoadOps. Therefore we just want to ensure that LoadOps have the same order.

  TODO(dtanner) Expand this to direct-to-lds.
  TODO(dtanner) Update this for removing gpu.barriers.
    For this we will take a LocalLoadOp, determine the dependent LocalStoreOps
    and continue that propagation.
******************************************************************************/
struct LocalStoreCriticalPathPriorityCalculator : public PriorityCalculator {

  LocalStoreCriticalPathPriorityCalculator()
      : PriorityCalculator(SchedDagNodePriorityType::LocalStoreCriticalPath) {}

  void calcPriorities() {
    SchedDagNodePriorityDataType priority = 1;
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->setPriority(SchedDagNodePriorityType::LocalStoreCriticalPath,
                          priority);
        priority += 1;
      }
    }

    // Propagate high priorities up for critical path.
    for (auto *node : dag->nodeList) {
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->propagateHigherPriorityToParents(
            SchedDagNodePriorityType::LocalStoreCriticalPath);
      }
    }

    // Propagate low priorities down for critical path.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->propagateLowerPriorityToChildren(
            SchedDagNodePriorityType::LocalStoreCriticalPath);
      }
    }
  }
};

/*
  Abstract Base class for scheduling heuristics.
  E.g. TopDown, schedule LoadOps early and LocalStoreOps late.
*/
template <SchedDirection Direction> struct SchedHeuristic {
  SchedHeuristic(StringRef n) : name(n) {}
  // Scheduler prints the name of heuristic.
  StringRef getName() const { return name; }
  // Scheduler calls before beginning a scheduling pass.
  virtual void begin() {};
  // Scheduler calls comparison functor to evaluate ready list.
  virtual SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) = 0;
  // Scheduler calls this printer right before evaluating the ready list,
  // for debugging when the heuristic has state.
  virtual void dump(llvm::raw_ostream &out) {};
  // Scheduler allows heuristic to update state based on scheduled op.
  virtual void selectedOp(SchedDagNode *) {};
  // Scheduler notifies scheduling is done, for debugging to print final state.
  virtual void end() {};
  virtual ~SchedHeuristic() = default;
  StringRef name;
};

/******************************************************************************
  Schedule ops based on priority, which are DotOp and LocalStoreOp critical
  paths. Priorities themselves have an priority order, Dot is p[0] and
  LocalStore is p[1]. Therefore, when scheduling TopDown, we first compare
  based on p[0], and only if they're the same do we examine p[1].
  This also means that when we schedule BottomUp, not only are we looking for
  the lowest priority ops first, but we also want to find the lowest
  p[1] before we look for the lowest p[0].
******************************************************************************/
template <SchedDirection Direction>
struct SchedHeuristicPriority : public SchedHeuristic<Direction> {

  SchedHeuristicPriority() : SchedHeuristic<Direction>("Priority") {}

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    uint32_t start = 0;
    uint32_t stop = static_cast<uint32_t>(SchedDagNodePriorityType::Size);
    uint32_t incr = 1;
    if (Direction == SchedDirection::BottomUp) {
      // Reverse priorities for reversed direction.
      start = static_cast<uint32_t>(SchedDagNodePriorityType::Size) - 1;
      stop = -1;
      incr = -1;
    }
    for (uint32_t i = start; i < stop; i += incr) {
      SchedDagNodePriorityType priorityType =
          static_cast<SchedDagNodePriorityType>(i);
      if (auto selected = findPreferredPriority(
              a, b, priorityType, Direction == SchedDirection::TopDown)) {
        return selected;
      }
    }

    // Fallback to original order.
    return getOriginalOrder<Direction>(a, b);
  }
};

/******************************************************************************
  Schedule based on MachineModel to reflect resource pipes and data latencies.
  Note, unlike other heuristics, this one employs prefer=true
  (rather than prefer=Direction==TopDown) because we always want a good
  machine state regardless of direction.

  Fallback comparison is Priority so that we also stick to the DotOp critical
  path.
******************************************************************************/
template <SchedDirection Direction>
struct SchedHeuristicMachineModel : public SchedHeuristic<Direction> {

  SchedHeuristicMachineModel()
      : SchedHeuristic<Direction>("MachineModel"),
        model(std::make_shared<MachineModelGFX942>()),
        machine(model.get(), Direction == SchedDirection::TopDown) {}

  void begin() {
    machine.reset();
    scheduleCycles.clear();
  };

  /*
    First priority is to get to dot.
  */
  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    // Prefer based on machine state; will select if one cooled down and other
    // isn't.
    if (auto selected = findPreferredMachineState(a, b, &machine, true)) {
      return selected;
    }

    SchedHeuristicPriority<Direction> shp;
    return shp(a, b);
  }

  void selectedOp(SchedDagNode *node) {
    machine.scheduleOp(node->getOp());
    scheduleCycles.push_back(std::make_pair(node, machine.getCurrentCycle()));
    // For anything else (non def/use) that waits, set data dependencies.
    if constexpr (Direction == SchedDirection::TopDown) {
      if (nodeCategoryMem(node)) {
        for (auto child : node->getChildren()) {
          if (llvm::isa<mlir::gpu::BarrierOp, mlir::cf::BranchOp>(
                  child->getOp())) {
            machine.updateOpDataReady(child->getOp(), node->getOp());
          }
        }
      }
    } else {
      if (llvm::isa<mlir::gpu::BarrierOp, mlir::cf::BranchOp>(node->getOp())) {
        for (auto parent : node->getParents()) {
          if (nodeCategoryMem(parent)) {
            machine.updateOpDataReady(parent->getOp(), node->getOp());
          }
        }
      }
    }
  }

  void dump(llvm::raw_ostream &out) { out << "MachineState: " << machine; };

  void end() {
    LDBG("Machine Schedule Cycles");
    for (auto entry : scheduleCycles) {
      LDBG("t=" << entry.second << " " << *entry.first);
    }
  }

  std::shared_ptr<MachineModel> model;
  MachineState machine;
  SmallVector<std::pair<SchedDagNode *, int32_t>> scheduleCycles;
};

/******************************************************************************
  Schedule original order (insomuch as dependencies allowed).
  Also hoist nops (which groups together) for ease of reading.
******************************************************************************/
struct SchedHeuristicOriginalOrder
    : public SchedHeuristic<SchedDirection::TopDown> {

  SchedHeuristicOriginalOrder()
      : SchedHeuristic<SchedDirection::TopDown>("OriginalOrder") {}

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    if (auto selected = findPreferredOpCategory(a, b, nodeCategoryNop)) {
      return selected;
    }
    return getOriginalOrder<SchedDirection::TopDown>(a, b);
  }
};

/*
  After scheduling the mlirBlock, we place sched.barriers(mask) to convey
  certain scheduling constraints to LLVM. E.g., we know one of the Deps passes
  added deps between LocalLoads and Dots, therefore we iterate through
  all the adjacent Dots and LocalLoads, and if there exists a dependency for
  them, we add a sched.barrier(mask=dot|Load); this is only valid since we know
  that the algorithm which added deps between dots and LocalLoads
  was ordering ALL dots and ALL local loads, and the sched.barrier
  prevents ALL from crossing the barrier.
  The existence of a dep alone does not justify a barrier since a dep
  only contraints the order of a single parent/child op pair, while
  a sched.barrier contrains ALL ops above and below, therefore we can
  only add sched.barriers with knowledge of the AddDeps passes.
*/
struct ApplySchedBarriers {
  ApplySchedBarriers(SchedDag &d, OpBuilder &b) : dag(d), builder(b) {}

  /*
    Verify parentIter is a parentCategory, then find next node matching
    childCategory. If parent/child is in deps, then we know we want them
    ordered. Returns whether a schedBarrier was added.
  */
  bool maybeAddSchedBarrier(SchedDagNode **parentIter,
                            std::function<bool(SchedDagNode *)> parentCategory,
                            std::function<bool(SchedDagNode *)> childCategory,
                            const StringRef depTypeName,
                            uint32_t schedBarMask) {
    SchedDagNode *parentNode = *parentIter;
    Operation *op = parentNode->getOp();
    if (!parentCategory(parentNode)) {
      return false;
    }
    // Find next dot/lds.
    for (SchedDagNode *const *it = parentIter + 1; it != dag.nodeList.end();
         it++) {
      SchedDagNode *childNode = *it;
      Operation *childOp = childNode->getOp();
      // Now check for dot/lds
      if (childCategory(childNode)) {
        // Check if dep exists.
        SchedDep dep(parentNode, childNode);
        if (dag.deps.at(depTypeName).contains(dep)) {
          auto loc = op->getLoc();
          builder.setInsertionPointAfter(op);
          Operation *schedBarOp =
              createSchedBarrier(builder, loc, schedBarMask);
          std::shared_ptr<SchedDagNode> schedBarNode =
              std::make_shared<SchedDagNode>(schedBarOp);
          dag.nodesHeap.push_back(schedBarNode);
          // Insert *after* node.
          LDBG("cap=" << dag.nodeList.capacity()
                      << ", size=" << dag.nodeList.size()
                      << ", max=" << dag.nodeList.max_size());
          LDBG("Inserting parentIter: " << **parentIter);
          LDBG("Inserting parentIt+1: " << **(parentIter + 1));
          parentIter = dag.nodeList.insert(parentIter + 1, schedBarNode.get());
          LDBG("Inserting parentIter: " << **(parentIter) << " (done)");
          // Skip analyzing the sched.barrier.
          ++parentIter;
          return true;
        } else {
          // The next child isn't a direct dependent.
          return false;
        }
      }
    }
    return true;
  }

  /*
    For each op, check what sched.barriers should go after it based on the
    parent/child nodes matching certain categories, and whether the parent/child
    dep is in the DepSet. Since the masks of sched.barriers can intersect, we
    first try to use masks with more constraints before falling back to masks
    which fewer constraints. For example first try adding mask=mfma|ds_read, and
    if that isn't valid try adding mask=mfma or mask=ds_read
  */
  void insertSchedBarriers() {
    // TODO(dtanner) Had a problem with list getting resizes while inserting;
    // it caused iterators to become invalidated, even though the implementation
    // seemed like it should have handled that. Therefore, reserve extra space
    // for inserting sched.barriers before trying to iterate. The number of
    // sched.barriers should be < number of ops.
    dag.nodeList.reserve(dag.nodeList.size() * 3);

    for (auto it = dag.nodeList.begin(); it != dag.nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      bool added;
      // TODO(dtanner) track what category the barrier types adds to add all
      // needed types but not superfluous.

      // Dot / Lds
      added = maybeAddSchedBarrier(it, nodeCategoryDot, nodeCategoryLocalLoad,
                                   "DotLdsOrder", schedBarMaskBlockDotLds);
      if (!added)
        added = maybeAddSchedBarrier(it, nodeCategoryLocalLoad, nodeCategoryDot,
                                     "DotLdsOrder", schedBarMaskBlockDotLds);
      if (!added)
        added =
            maybeAddSchedBarrier(it, nodeCategoryDot, nodeCategoryLocalStore,
                                 "DotLdsOrder", schedBarMaskBlockDotLds);
      if (!added)
        added =
            maybeAddSchedBarrier(it, nodeCategoryLocalStore, nodeCategoryDot,
                                 "DotLdsOrder", schedBarMaskBlockDotLds);

      // Dot / Dot
      if (!added)
        added = maybeAddSchedBarrier(it, nodeCategoryDot, nodeCategoryDot,
                                     "DotTypeOrder", schedBarMaskBlockDot);
      // LocalLoad / LocalLoad
      if (!added)
        added = maybeAddSchedBarrier(
            it, nodeCategoryLocalLoad, nodeCategoryLocalLoad,
            "LocalLoadTypeOrder", schedBarMaskBlockDsRead);
      // LocalStore / LocalStore
      if (!added)
        added = maybeAddSchedBarrier(
            it, nodeCategoryLocalStore, nodeCategoryLocalStore,
            "LocalStoreTypeOrder", schedBarMaskBlockDsWrite);

      // GlobalLoad / GlobalLoad
      if (!added)
        added = maybeAddSchedBarrier(it, nodeCategoryGlobal, nodeCategoryGlobal,
                                     "GlobalLoadCategoryOrder",
                                     schedBarMaskBlockGlobal);
      // TODO(dtanner) anchor GlobalLoads with Dots?
    }
    LDBG("insertSchedBarriers() - DONE");
  }

  SchedDag &dag;
  OpBuilder &builder;
};

/******************************************************************************
  SchedManager is top-level struct which manipulates the SchedDag:
  - Add deps and priorities to dag.
  - Prepared dag for scheduling.
  - Runs scheduling heuristic on the dag.
******************************************************************************/
struct SchedManager {
  SchedManager(Block *block, OpBuilder &b)
      : dag(block), rescheduleId(0), builder(b) {}

  // Calculate new deps based on op order and previously determined deps.
  // Insert new deps into dep map and apply them to dat.
  void addDeps(std::unique_ptr<DependencyCalculator> depCalc) {
    depCalc->addDepsToDag(&dag);
  }

  // Calculate new node priorities based on dag.
  void addPriorities(std::unique_ptr<PriorityCalculator> prioCalc) {
    prioCalc->addPrioritiesToDag(&dag);
  }

  // Reschedule the dag.
  template <SchedDirection Direction>
  void reschedule(SchedHeuristic<Direction> *heuristic) {
    LDBG("SchedManager::reschedule("
         << rescheduleId << "), Direction="
         << ((Direction == SchedDirection::TopDown) ? "TopDown" : "BottomUp")
         << ", Heuristic=" << heuristic->getName());
    // Reset deps right before rescheduling b/c analysis passes may have altered
    // them.
    dag.resetDeps();
    LDBG("SchedDag before reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(dag.dumpDotFormat(llvm::dbgs()));

    // Node readiness is based on direction.
    dag.initReadyNodes<Direction>();
    heuristic->begin();

    // Schedule the dag; this process removes deps from nodes.
    // Store nodes in newly scheduled order.
    SchedDagNodeList rescheduledNodes;
    const bool printDetails = rescheduleId > 0;
    for (int iter = 0; !dag.finished(); ++iter) {
      scheduleNextOp<Direction>(heuristic, rescheduledNodes);
    }
    heuristic->end();

    // After scheduling, re-apply deps to prepare for adding additional deps.
    dag.resetDeps();

    // Update nodeList after rescheduling.
    if constexpr (Direction == SchedDirection::TopDown) {
      LDBG("dag.nodeList = rescheduledNodes");
      dag.nodeList = rescheduledNodes;
    } else {
      LDBG("dag.nodeList = reversed(rescheduledNodes)");
      dag.nodeList.clear();
      for (auto it = rescheduledNodes.rbegin(); it != rescheduledNodes.rend();
           ++it) {
        auto &node = *it;
        dag.nodeList.push_back(node);
      }
    }

    LDBG("NodeList after reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(dag.dumpNodes(llvm::dbgs()));

    LDBG("SchedManager::reschedule(" << rescheduleId << ") - DONE");
    rescheduleId++;
  }

  // Single pass of rescheduling the dag.
  template <SchedDirection Direction>
  void scheduleNextOp(SchedHeuristic<Direction> *heuristic,
                      SchedDagNodeList &rescheduledNodes) {
    const bool printDetails = rescheduleId > 0;
    // Print ReadyNodes and HeuristicState
    const auto &readyNodes = dag.getReadyNodes();
    if (printDetails) {
      LDBG("Iter: " << rescheduledNodes.size());
      LDBG("Ready List:");
      for (auto node : readyNodes) {
        LDBG("    " << *node);
      }
      LLVM_DEBUG(heuristic->dump(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }

    // Select
    SchedDagNode *selectedNode = selectFromReadyNodes(readyNodes, heuristic);
    if (printDetails) {
      LDBG("Selected: " << *selectedNode);
    }
    heuristic->selectedOp(selectedNode);

    // Place selected node in list, remove it from dag which updates
    // readyList.
    rescheduledNodes.push_back(selectedNode);
    dag.removeScheduledNode<Direction>(selectedNode);
  }

  template <SchedDirection Direction>
  SchedDagNode *selectFromReadyNodes(SetVector<SchedDagNode *> readyNodes,
                                     SchedHeuristic<Direction> *heuristic) {
    SchedDagNode *selected = readyNodes.front();
    for (auto it = std::next(readyNodes.begin()); it != readyNodes.end();
         ++it) {
      SchedDagNode *node = *it;
      selected = (*heuristic)(selected, node);
    }
    return selected;
  }

  void insertSchedBarriers() {
    ApplySchedBarriers asb(dag, builder);
    asb.insertSchedBarriers();
  }

  SmallVector<Operation *> getOpList(bool applySchedBarriers = true) {
    SmallVector<Operation *> opList;
    for (auto node : dag.nodeList) {
      Operation *op = node->getOp();
      opList.push_back(op);
    }
    return opList;
  }

  SchedDag dag;
  // For debugging only.
  int32_t rescheduleId;
  OpBuilder &builder;
}; // SchedManager

/******************************************************************************
  TritonAMDGPURescheduleOps::applyReschedulingPasses()
  Top-level scheduling pass for a single block.
******************************************************************************/
struct TritonAMDGPURescheduleOps
    : public TritonAMDGPURescheduleOpsBase<TritonAMDGPURescheduleOps> {
  explicit TritonAMDGPURescheduleOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  LogicalResult verify(Block *mlirBlock) {
    // make sure that a block gets terminated with `cf::BranchOp`
    if (!dyn_cast<mlir::cf::BranchOp>(&(mlirBlock->back()))) {
      return failure();
    }

    // don't schedule if there is not enough operations in a block
    constexpr int NumMinOpsInBlock = 3;
    if (mlirBlock->getOperations().size() < NumMinOpsInBlock)
      return failure();
    return success();
  }

  void applyReschedulingPasses(Block *mlirBlock) {
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses()");
    MLIRContext *context = &getContext();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(mlirBlock);
    SchedManager schedManager(mlirBlock, builder);
    bool dumpGraphs = false;

    /*
      Scheduling Pass 0
      - Dependencies: Data, DotOrder, LocalStoreOrder, Barriers
      - Priorities: DotCriticalPath, LocalStoreCriticalPath
      - Heuristic: Priority

      Dependencies accomplish:
      - Enforced dot relative order.
      - Enforced LocalStoreOp relative order.
      Scheduling in order of priority (critical path) accomplishes:
      - Note that LoadOp are close to LocalStoreOp on purpose.
      - Determines relative order of LocalLoadOps.
      - Determines order of tertiary ops to get to the DotOps asap.
      - Determines the order of LoadOps to match LocalStoreOps.
    */
    // Add Data deps based on def-use chains.
    schedManager.addDeps(std::make_unique<DataDependencyCalculator>());
    if (dumpGraphs) {
      LLVM_DEBUG(LDBG("Dag: data");
                 schedManager.dag.dumpDotFormat(llvm::dbgs()););
    }
    // Assume dots are in ideal order and propagate their critical path.
    schedManager.addDeps(std::make_unique<DotOrderDependencyCalculator>());
    // Add initial priorities while only data deps exist, before adding other
    // deps.
    schedManager.addPriorities(
        std::make_unique<DotCriticalPathPriorityCalculator>());
    // Assume LocalStoreOps are in ideal order and propagate their critical
    // path.
    schedManager.addDeps(
        std::make_unique<LocalStoreOrderDependencyCalculator>());
    schedManager.addPriorities(
        std::make_unique<LocalStoreCriticalPathPriorityCalculator>());
    // Add dependencies for barriers (gpu.barrier, sched.barrier, setprio...).
    schedManager.addDeps(std::make_unique<BarrierDependencyCalculator>());
    // After creating dependencies for barriers, repeat priority analysis.
    // This won't override any previous priorities, but for tertiary ops it
    // determines which are needed to unblock barriers which block other
    // priorities.
    schedManager.addPriorities(
        std::make_unique<DotCriticalPathPriorityCalculator>());
    schedManager.addPriorities(
        std::make_unique<LocalStoreCriticalPathPriorityCalculator>());
    if (dumpGraphs) {
      LLVM_DEBUG(LDBG("Dag: data, dot:dot, ls:ls, bar:*");
                 schedManager.dag.dumpDotFormat(llvm::dbgs()););
    }
    SchedHeuristicPriority<SchedDirection::TopDown> shp;
    schedManager.reschedule<SchedDirection::TopDown>(&shp);

    /*
      Scheduling Pass 1
      - Dependencies: LocalLoadOrder, GlobalLoadOrder
      - Priorities: none
      - Heuristic: MachineModel<BottomUp>

      Dependencies accomplish:
      - Enforce LocalLoadOp relative order.
      - Enforce LoadOp relative order.
      Scheduled in order of MachineModel<BottomUp> accomplishes:
      - Determines LocalStoreOps before barriers and end of loop.
      - Spreads LocalStoreOps out.
      - Determines LocalLoadOps prefetched before DotOps.
      - Spreads LocalLoadOps out.
      - Lifts LoadOps as high as possible, which is likely top of block.
    */
    // Preserve memory op order determined by critical paths above.
    schedManager.addDeps(
        std::make_unique<LocalLoadOrderDependencyCalculator>());
    schedManager.addDeps(
        std::make_unique<GlobalLoadOrderDependencyCalculator>());
    if (dumpGraphs) {
      LLVM_DEBUG(LDBG("Dag: data, dot:dot, ls:ls, bar:*, ll:ll, gl:gl");
                 schedManager.dag.dumpDotFormat(llvm::dbgs()););
    }
    // Now that we've ordered lds ops, spread them out with machine model.
    // This comes first because we want LL and LS as late as possible.
    // This pass pushes the GL up as high as possible.
    SchedHeuristicMachineModel<SchedDirection::BottomUp> sh1;
    schedManager.reschedule<SchedDirection::BottomUp>(&sh1);

    /*
      Scheduling Pass 2
      - Dependencies: DotLdsOrder; PriorityOrder
      - Priorities: 0
      - Heuristic: EarlyGlobalLoads

      Dependencies accomplish:
      - Enforce LocalLoadOp relative to DotOps.
      - Enforce LocalStoreOp relative to DotOps.
      Scheduled in order of MachineModel<TopDown> accomplishes:
      - Delays LoadOps until issuing them is hidden by DotOps.
      - Spreads out LoadOps.
    */
    schedManager.addDeps(std::make_unique<DotLdsOrderDependencyCalculator>());
    if (dumpGraphs) {
      LLVM_DEBUG(
          LDBG(
              "Dag: data, dot:dot, ls:ls, bar:*, ll:ll, gl:gl, ll:dot, ls:dot");
          schedManager.dag.dumpDotFormat(llvm::dbgs()););
    }
    SchedHeuristicMachineModel<SchedDirection::TopDown> sh2;
    schedManager.reschedule<SchedDirection::TopDown>(&sh2);

    /*
      TODO(dtanner) - flash-attention (or anything with multiple GL/LS)
      will need to have some anti-deps to keep some GL after LS so they
      can use the same registers. This should probably be added after
      DotLdsOrder dependencies and before the machine model top-down, so that it
      just delays the LoadOps even further until after the anti-dep.
      schedManager.addDeps(std::make_unique<LocalStoreGlobalLoadAntiDepsDependencyCalculator>());
      This is already the beginning of working on this.
      schedManager.addDeps(std::make_unique<MemOrderDependencyCalculator>());
    */

    schedManager.insertSchedBarriers();

    // Copy scheduled order to basic block.
    SmallVector<Operation *> rescheduledOps = schedManager.getOpList();
    for (auto it = rescheduledOps.rbegin(); it != rescheduledOps.rend(); ++it) {
      (*it)->moveBefore(mlirBlock, mlirBlock->begin());
    }
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses() - DONE");
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder b(&getContext());
    ModuleOp mod = getOperation();
    llvm::SmallVector<Block *> blocks;
    mod.walk([&](triton::amdgpu::InstructionSchedHint hint) {
      if (hint.getVariant() == triton::amdgpu::SchedHint::refine_ops) {
        blocks.push_back(hint->getBlock());
        hint->erase();
      }
    });

    for (auto block : blocks) {
      if (succeeded(verify(block))) {
        LDBG("OpList before applyReschedulingPasses()");
        for (auto it = block->begin(); it != block->end(); ++it) {
          LLVM_DEBUG((*it).print(llvm::dbgs()); llvm::dbgs() << "\n";);
        }
        applyReschedulingPasses(block);
        LDBG("OpList after applyReschedulingPasses()");
        for (auto it = block->begin(); it != block->end(); ++it) {
          LLVM_DEBUG((*it).print(llvm::dbgs()); llvm::dbgs() << "\n";);
        }
      }
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURescheduleOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURescheduleOps>(targetArch);
}
} // namespace mlir
