#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/SchedulerMachineModel.h"

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
  level.

  The goals of rescheduling the basic block are:
  (1) Triton scheduling only needs to compliment LLVM scheduler,
      not redo same heuristics.
  (2) To create a better op order before LLIR to help LLVM backend scheduler.
  (3) Inject sched.barriers into op order to provide scheduling guard rails
      to backend scheduler to constrain pre-RA and post-RA scheduling.

  Scheduling consists of multiple passes controlled by SchedManager:
  (1) Analyse the current op sequence.
  (2) Create an additional sets of dependencies in SchedDag.
  (3) Reschedule based on ready-list,
      (a) Select best node based on heuristic.
      (b) Remove node and dependencies from nodes in SchedDag.
      (c) Update ready-list.
  (4) Results in a new op sequence. Also, restore deps in SchedDag.
******************************************************************************/
namespace {

// TODO (ravil): Note, took function from `SchedInstructions.cpp`.
// we need to combine these two implementations
Operation *createSchedBarrier(OpBuilder &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

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

  For example, we want to get to a dot asap, so we want to analyze which are on critical path.
  Also, we want to set the relative order of local loads.

*/

enum class SchedDagNodePriorityType : uint32_t {
  DotCriticalPath = 0,
  LocalStoreCriticalPath = 1,
  Size // Keep as last to know size().
};
using SchedDagNodePriorityDataType = float;
using SchedDagNodePriority = SmallVector<SchedDagNodePriorityDataType, static_cast<uint32_t>(SchedDagNodePriorityType::Size)>;
constexpr SchedDagNodePriorityDataType schedDagNodePriorityUnset = std::numeric_limits<SchedDagNodePriorityDataType>::lowest();
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
  SchedDagNode(Operation *op) : op(op),
      priority(static_cast<uint32_t>(SchedDagNodePriorityType::Size),
      schedDagNodePriorityUnset) {
    static int32_t serialId = 0;
    id = serialId++;
    opStr = op->getName().getStringRef();
    resetPriorities();
  }

  // Copy constructor.
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
    When scheduling top-down, a node is ready to schedule when it has no
    parents. When scheduling bottom-up, a node is ready to schedule when it has
    no children.
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

  // Returns whether this is ancestor (by recursively checking children) of
  // other; however this is too expensive for production.
  bool isAncestor(SchedDagNode *other) const {
    for (SchedDagNode *child : children) {
      if (child == other) {
        // This is ancestor of other if other is this' child.
        return true;
      } else if (child->isAncestor(other)) {
        // This is ancestor of other if one of this' children are ancestor of
        // other.
        return true;
      }
    }
    return false;
  }

  // Returns whether this is posterity (by recursively checking parents) of
  // other; however this is too expensive for production.
  bool isPosterity(SchedDagNode *other) const {
    for (SchedDagNode *parent : parents) {
      if (parent == other) {
        // This is posterity of other if other is this' parent.
        return true;
      } else if (parent->isPosterity(other)) {
        // This is posterity of other if one of its parents are posterity of
        // other.
        return true;
      }
    }
    return false;
  }

  void setPriority(SchedDagNodePriorityType type, SchedDagNodePriorityDataType value) {
    priority[static_cast<uint32_t>(type)] = value;
  }
  SchedDagNodePriorityDataType getPriority(SchedDagNodePriorityType type) const {
    return priority[static_cast<uint32_t>(type)];
  }
  bool hasPriority(SchedDagNodePriorityType type) const {
    return getPriority(type) != schedDagNodePriorityUnset;
  }
  void resetPriorities() {
    for (uint32_t i = 0; i < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++i) {
      setPriority(static_cast<SchedDagNodePriorityType>(i), schedDagNodePriorityUnset);
    }
  }
  void propagateHigherPriorityToParents(SchedDagNodePriorityType type) {
    assert(hasPriority(type));
    for (SchedDagNode *parent : parents) {
      if (!parent->hasPriority(type) || getPriority(type) > parent->getPriority(type)) {
        parent->setPriority(type, getPriority(type));
        parent->propagateHigherPriorityToParents(type);
      }
    }
  }
  void propagateLowerPriorityToChildren(SchedDagNodePriorityType type) {
    assert(hasPriority(type));
    for (SchedDagNode *child : children) {
      if (!child->hasPriority(type) || getPriority(type) < child->getPriority(type)) {
        child->setPriority(type, getPriority(type));
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
  // Priorities are enumerated and optionally set by DependencyCalculators
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
  for (uint32_t i = 0; i < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++i) {
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
bool opCategoryLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::LoadOp, triton::gpu::LocalLoadOp,
                   triton::amdgpu::BufferLoadOp>(op);
}

bool opCategoryStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::StoreOp, triton::gpu::LocalStoreOp>(op);
}

bool opCategoryMem(SchedDagNode *node) {
  return opCategoryLoad(node) || opCategoryStore(node);
}

bool opCategoryGlobalLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(op);
}

bool opCategoryGlobalStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::StoreOp>(op);
}

bool opCategoryNop(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<triton::gpu::MemDescSubviewOp, triton::gpu::MemDescTransOp,
                   tt::amdgpu::ExtractSliceOp, tt::amdgpu::ConcatOp>(op);
}

bool opCategoryBarrier(SchedDagNode *node) {
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
  } else if (opCategoryGlobalLoad(node)) {
    return "red";
  } else if (opCategoryGlobalStore(node)) {
    return "green";
  } else if (opCategoryBarrier(node)) {
    return "magenta";
  } else if (opCategoryNop(node)) {
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
  TODO(dtanner) - isEqual() might be simplifiable.
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
    LDBG("removeNodeCascadeDeps() " << *node);
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
    LDBG("calcDepsSchedBar()");
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
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
    calcDepsRefinedOp();
    calcDepsDot();
  }
};

struct PriorityOrderDependencyCalculator : DependencyCalculator {
  PriorityOrderDependencyCalculator()
      : DependencyCalculator("PriorityOrder") {}
  void calcDeps() {
    
    for (uint32_t priorityIdx = 0; priorityIdx < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++priorityIdx) {
      
      SchedDagNodePriorityType priorityType = static_cast<SchedDagNodePriorityType>(priorityIdx);
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
          } else { // maintain order even when we dropped priority b/c other deps.
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
  DotOrderDependencyCalculator()
      : DependencyCalculator("DotTypeOrder") {}
  void calcDeps() {
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
    calcDepsOpType<triton::DotOp>(&dag->nodeList, depSet);
  }
};

struct LocalLoadOrderDependencyCalculator : DependencyCalculator {
  LocalLoadOrderDependencyCalculator()
      : DependencyCalculator("LocalLoadTypeOrder") {}
  void calcDeps() {
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
    calcDepsOpType<triton::gpu::LocalLoadOp>(&dag->nodeList, depSet);
  }
};

struct LocalStoreOrderDependencyCalculator : DependencyCalculator {
  LocalStoreOrderDependencyCalculator()
      : DependencyCalculator("LocalStoreTypeOrder") {}
  void calcDeps() {
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
    calcDepsOpType<triton::gpu::LocalStoreOp>(&dag->nodeList, depSet);
  }
};

struct GlobalLoadOrderDependencyCalculator : DependencyCalculator {
  GlobalLoadOrderDependencyCalculator()
      : DependencyCalculator("GlobalLoadCategoryOrder") {}
  void calcDeps() {
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
    calcDepsOpCategory(&dag->nodeList, depSet, opCategoryGlobalLoad);
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
      if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(node->getOp())) {
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
Create high-level dependencies between the different memory ops where there
aren't data deps. A basic block has a collection of refined memory ops, GL, LS,
LL.

for num_stages=2 without local_prefetch
the GL and LS have data dependencies, and both can be co-scheduled with the
local_loads. So we need to specify that the GL should overlap the LL. The LS
can't overlap LL because of anti-dep enforced by gpu.bar.

Example:
GLA01
GLB01
LLA0123
LLB0123
LSA01
LSB01

Could be serialized like

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

or

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

https://github.com/ROCm/triton-internal/issues/736
Determine Memory Op Order & Co-Scheduling
We need to determine the relative ordering between or overlap of different
memory op types; e.g. if global_loads and local_loads can both be scheduled,
which should have higher priority. The ordering of memory ops have implications
for performance and register use. This applies to loops where data can be
prefetched, and also to regions with multiple dots. First, data dependencies
determine order. Second, critical-path analysis determines order, meaning the
memory ops which need to be scheduled to get to the mfmas sooner have highest
priority. Note that for local_loads, some may be on the critical path to get us
to the first 1-2 dot-tiles, but after other dot-tiles' local_loads can be
delayed Third, based on scheduling mode: Min-Vgpr mode: preferred order is
local_store, global_load, local_load. Max-Latency mode: preferred order is
global_load, local_load, local_store.
========================================================================
This applies to loops where data can be prefetched.
This should also apply to regions with multiple dots.
First, def-use dependencies kick in to determine order
If LS overlaps LL (both prefetched, 011, 111)
  min-vgpr: LS, LL to save vgprs
  max-latency: LL, LS to maximize GL latency hiding
  ever want to co-schedule? - no
If GL overlaps LL (GL and (LS or LL) prefetched; 101, 110, 111)
  min-vgpr: GL, LL
  max-latency: GL, LL (same)
If GL overlaps LS
  min-vgpr: LS before GL to save vgprs
  max-latency: GL before LS to maximize latency hiding
If GL overlaps LS overlaps LL
  min-vgpr: LS, GL, LL to save vgprs
  max-latency: GL, LL, LS
Need to verify the above doesn’t create any cyclic dependencies?
Need to clarify that some of these “after” are actually “interleave”.
Need to distinguish between vgpr vs latency for global separate from LDS ops?

The above logic is hyper-focused on gemm, and we need to abstract much more.

==========================================================================
This is actually mostly determined during StreamPipeliner via clustering and
num_stages. We need to decide the optimal memoy op ordering (via cluster) during
StreamPipeliner and covey it forward via the IR. Additionally, we also need
StreamPipeliner to specify whether op A should fully preceed op B, or whether op
A and op B can overlap and therefore be interleaved after refining.
==========================================================================



This first issue has mostly to do with when everything is prefetched, what order
should they be in.



AddDeps: Order deps between different memory types #738
https://github.com/ROCm/triton-internal/issues/738
Memory Opr Order Deps

With this ordering of ops, we can create order dependencies between op types so
we only schedule the ones at a time that we want. E.g. for local prefetch,
scheduler shouldn’t even see local_loads until very late in the loop.

Insert order dependencies to serialize certain memory ops.

This may need information to be communicated from the stream-pipeliner (or other
places) to the scheduler. Some examples of what this task means: (1) A gemm with
num_stages=2 wants local_loads close to the top of the kernel and global_loads
to come after the first few local_loads but interleaved among later local_loads.
Whereas enabling local_prefetch puts all the local_loads at the bottom of the
loop after the global_loads and local_stores. The scheduler needs to know "delay
the local_loads as much as possible" after most mfmas which have freed registers
which the local_local loads can then use. The backend gets this wrong because it
schedules the local_loads early and uses too many vgprs.

(2) For FA num_stages=2 maxDepth=1, the memory structure looks like
global_load
local_store <-- these must come before
global_load <-- these to save vgprs
local_store

This second issue takes

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

and further narrows it to

LLA0
LLB0
LLB1
LLA1
GLA0
LLA2
GLA1
LLB2
GLB0
LLB3
GLB1
LLA3
LSA0
LSA1
LSB0
LSB1

Later will AddDeps between the mem ops and


global_load before/after local_loads
global_loads before/after local_stores
local_load[A] before/after local_load[B]
global_load[K] before/after global_load[K]


Determines which memory ops should overlap other memory ops (vs being
co-scheduled) when data dependencies allow them to be.

if we have all dots in order, and we do a scheduling with delaying all local
loads then we can just grab the order of local_loads from that. Can we just keep
the relative order of local stores as being final? Then have the relative order
of global loads to match (with loop wrap around).
******************************************************************************/
struct MemOrderDependencyCalculator : DependencyCalculator {
  MemOrderDependencyCalculator() : DependencyCalculator("MemOrder") {}

  // get memory ops only from the graph; keep them in order.
  void createMemDag(SchedDag *memDag) const {

    LDBG("Removing non-mem nodes.");
    SchedDagNodeList listCopy = memDag->nodeList;
    for (SchedDagNode *node : listCopy) {
      if (!opCategoryMem(node) && !isa<triton::DotOp>(node->op)) {
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
    LDBG("DependencyCalculator<" << depTypeName << ">::calcDeps()");
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
******************************************************************************/

// Prefer op with higher priority.
bool preferPriority(SchedDagNode *a, SchedDagNode *b,
                    SchedDagNodePriorityType priorityType,
                    bool prefer = true) {
  if (a->hasPriority(priorityType)) {
    if (b->hasPriority(priorityType)) {
      return (a->getPriority(priorityType) > b->getPriority(priorityType)) == prefer;
    }
    return prefer;
  }
  return !prefer;
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

// Prefer op based on type.
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

// Prefer op based on category (e.g. opCategoryLoad()).
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

bool preferLocalLoad(SchedDagNode *a, SchedDagNode *b, bool prefer = true) {
  return (llvm::isa<triton::gpu::LocalLoadOp>(a->getOp()) and
          !llvm::isa<triton::gpu::LocalLoadOp>(b->getOp())) == prefer;
}
#if 0
bool preferMachineState(SchedDagNode *a, SchedDagNode *b, MachineState *machine, bool prefer = true) {
  return (machine->pipeReadyScore(a->getOp()) > machine->pipeReadyScore(b->getOp())) == prefer;
}
SchedDagNode *findPreferredMachineState(SchedDagNode *a, SchedDagNode *b, MachineState *machine,
                                      bool prefer = true) {
  if (preferMachineState(a, b, machine, prefer)) {
    LDBG("MM preferMachineState() -> a");
    return a;
    LDBG("MM preferMachineState() -> b");
  } else if (preferMachineState(b, a, machine, prefer)) {
    return b;
  }
  return nullptr;
}
#endif
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
    LDBG("PriorityCalculator<" << toString(priorityType) << ">::addPrioritiesToDag()");
    dag = d;
    calcPriorities();
  }
  virtual ~PriorityCalculator() = default;

  SchedDagNodePriorityType priorityType;
  SchedDag *dag;
};

/******************************************************************************
  Add DotCriticalPath weights to nodes to correctly schedule
  local load order.

la0 = 9
la1 = 6
la2 = 3
lb0 = 9
lb1 = 8
lb2 = 7

dot00 = 9
dot01 = 8
dot02 = 7
dot10 = 6
dot11 = 5
dot12 = 4
dot20 = 3
dot21 = 2
dot22 = 1

Only use Data dependencies as parent/children since we want to
prioritiese the flow of data.
We don't want barriers to give all ops the same priorities.
******************************************************************************/
struct DotCriticalPathPriorityCalculator
    : public PriorityCalculator {

  DotCriticalPathPriorityCalculator() : PriorityCalculator(SchedDagNodePriorityType::DotCriticalPath) {}
  
  void calcPriorities() {
    dag->resetDeps();
    dag->applyDeps("Data");
    SchedDagNodePriorityDataType dotPriority = 1;
    // Dots have priority N -> 1.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::DotOp>(node->getOp())) {
        node->setPriority(SchedDagNodePriorityType::DotCriticalPath, dotPriority);
        dotPriority += 1;
      }
    }

    // Propagate high priorities up for critical path.
    for (auto *node : dag->nodeList) {
      if (isa<triton::DotOp>(node->getOp())) {
        node->propagateHigherPriorityToParents(SchedDagNodePriorityType::DotCriticalPath);
      }
    }

    // Propagate low priorities down for critical path.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::DotOp>(node->getOp())) {
        node->propagateLowerPriorityToChildren(SchedDagNodePriorityType::DotCriticalPath);
      }
    }
  }
};

/******************************************************************************
  Ideally we want the DotCriticalPath to also be able to label the local stores.
  However local memory semantics make this hard, do we have aliasing information
  so I can query which local_stores are needed for a local_read.
  Since gpu.barriers enforce that all local_loads must complete,
  we can assume that local_stores are already correctly ordered relative to local_loads.
  Now we want to continue the critical path to specify what is the optimal order
  of local stores, and by consequence the optimal order of global loads.
  Therefore we just want to ensure that global_loads have the same order.
  TODO(dtanner) expand this to direct-to-lds.
******************************************************************************/
struct LocalStoreCriticalPathPriorityCalculator
    : public PriorityCalculator {

  LocalStoreCriticalPathPriorityCalculator() : PriorityCalculator(SchedDagNodePriorityType::LocalStoreCriticalPath) {}
  
  void calcPriorities() {
    dag->resetDeps();
    dag->applyDeps("Data");
    SchedDagNodePriorityDataType priority = 1;
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->setPriority(SchedDagNodePriorityType::LocalStoreCriticalPath, priority);
        priority += 1;
      }
    }

    // Propagate high priorities up for critical path.
    for (auto *node : dag->nodeList) {
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->propagateHigherPriorityToParents(SchedDagNodePriorityType::LocalStoreCriticalPath);
      }
    }

    // Propagate low priorities down for critical path.
    for (auto it = dag->nodeList.rbegin(); it != dag->nodeList.rend(); ++it) {
      SchedDagNode *node = *it;
      if (isa<triton::gpu::LocalStoreOp>(node->getOp())) {
        node->propagateLowerPriorityToChildren(SchedDagNodePriorityType::LocalStoreCriticalPath);
      }
    }
  }
};

/*
  Abstract Base class for a scheduling heuristic recipe.
  E.g. TopDown, schedule global_loads early and local_stores late.
*/
template <SchedDirection Direction> struct SchedHeuristic {
  virtual SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) = 0;
  virtual StringRef name() = 0;
  virtual void reset() {};
  virtual void notifySelected(SchedDagNode *) {};
  virtual void dump(llvm::raw_ostream &out) {};
  virtual ~SchedHeuristic() = default;
};

/******************************************************************************
  Delay LocalLoads as much as possible so they're adjacent to the dot which
  needs them. This will then allow for placing deps between local loads.
  Also delay other memory ops so that order deps can be placed between them too.
  SchedDirection = BottomUp
******************************************************************************/
template <SchedDirection Direction>
struct SchedHeuristicPriority
    : public SchedHeuristic<Direction> {

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {

    for (uint32_t i = 0; i < static_cast<uint32_t>(SchedDagNodePriorityType::Size); ++i) {
      SchedDagNodePriorityType priorityType = static_cast<SchedDagNodePriorityType>(i);
      if (auto selected = findPreferredPriority(a, b, priorityType,
          Direction==SchedDirection::TopDown)) {
        return selected;
      }
    }

    // Final comparison based on orig order.
    return getOriginalOrder<Direction>(a, b);
  }

  StringRef name() { return "Priority"; }
};

/******************************************************************************
  Employ MachineModel to capture data latencies and issue rate latencies.
  SchedDirection = BottomUp
  Memory ops are already ordered.
  Goals:
   - Local loads correct, therefore asap according to machine model.
   - Local stores correct, therefore asap according to machine model.
   - Global loads as late as possible to not hinder the above.
  Change UpdateState to NotifyScheduled before node is removed.
  We will examine it's parents and if they are memroy ops,
  we'll store when they're allowed to be issued according to data latencies
  Only need to store single value for global load, local load, local store
******************************************************************************/
template <SchedDirection Direction> 
struct SchedHeuristicLdsOps
    : public SchedHeuristic<Direction> {

  SchedHeuristicLdsOps() : model(std::make_shared<MachineModelGFX942>()),
      machine(model.get(), Direction==SchedDirection::TopDown) {
    machine.reset();
  }
  
  bool preferMachineState(SchedDagNode *a, SchedDagNode *b, MachineState *machine, bool prefer = true) {
    return (machine->getCyclesUntilOpReady(a->getOp())
        < machine->getCyclesUntilOpReady(b->getOp())) == prefer;
  }

  SchedDagNode *findPreferredMachineState(SchedDagNode *a, SchedDagNode *b, MachineState *machine,
                                        bool prefer = true) {
    if (preferMachineState(a, b, machine, prefer)) {
      return a;
    } else if (preferMachineState(b, a, machine, prefer)) {
      return b;
    }
    return nullptr;
  }

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    LDBG("MM comparing " << *a << " and " << *b);
    // Prefer based on machine state; will select if one cooled down and other isn't.
    if (auto selected = findPreferredMachineState(a, b, &machine)) {
      LDBG("Selected based on MachineState");
      return selected;
    }

    // Schedule others that could be in the way of local loads
    // in the order of data flowing to dots.
    if (auto selected = findPreferredOpCategory(a, b, opCategoryNop,
        Direction==SchedDirection::TopDown)) {
      return selected;
    }
    if (auto selected = findPreferredOpType<triton::gpu::LocalLoadOp>(a, b,
      Direction==SchedDirection::TopDown)) {
      LDBG("Selected based on LocalLoadOp");
      return selected;
    }
    if (auto selected = findPreferredOpType<triton::gpu::LocalStoreOp>(a, b,
      Direction==SchedDirection::TopDown)) {
      LDBG("Selected based on LocalStoreOp");
      return selected;
    }
    if (auto selected = findPreferredOpCategory(a, b, opCategoryBarrier,
      Direction==SchedDirection::TopDown)) {
      return selected;
    }
    if (auto selected = findPreferredOpCategory(a, b, opCategoryGlobalLoad,
      Direction==SchedDirection::TopDown)) {
      LDBG("Selected based on GlobalLoad");
      return selected;
    }

    // Final comparison based on orig order.
    return getOriginalOrder<Direction>(a, b);
  }

  StringRef name() { return "Lds"; }

  void reset() {
    machine.reset();
  };

  // node still has dependencies.
  void notifySelected(SchedDagNode *node) {
    LDBG("notifySelected()" << *node);

    // MachineModelOpProperties properties = machine.machineModel->getOpProperties(node->getOp());
    machine.scheduleOp(node->getOp());
    // For anything else (non def/use) that waits, set data dependencies.
    if constexpr (Direction==SchedDirection::TopDown) {
      assert(false);
    } else {
      if (llvm::isa<mlir::gpu::BarrierOp, mlir::cf::BranchOp>(node->getOp())) {
        for (auto parent : node->getParents()) {
          if (opCategoryMem(parent)) {
            LDBG("machine.updateOpDataReady() for " << *parent << ", " << *node);
            machine.updateOpDataReady(parent->getOp(), node->getOp());
          }
        }
      }
    }
  }

  void dump(llvm::raw_ostream &out) {
    out << "MachineState: " << machine;
  };

  std::shared_ptr<MachineModel> model;
  MachineState machine;
};

/******************************************************************************
  After establishing where LocalStoreOp must go,
  now we can lift global loads early.
  TODO(dtanner) need to add antideps from buffer_loads and local_stores for FA
******************************************************************************/
template <SchedDirection Direction>
struct SchedHeuristicGlobalLoadsEarly
    : public SchedHeuristic<Direction> {

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {

    if (auto selected = findPreferredOpCategory(a, b, opCategoryGlobalLoad,
      Direction==SchedDirection::TopDown)) {
      LDBG("Selected based on GlobalLoad");
      return selected;
    }

    // Final comparison based on orig order.
    return getOriginalOrder<Direction>(a, b);
  }

  StringRef name() { return "GlobalLoadsEarly"; }
};

/******************************************************************************
  Schedule original order (insomuch as dependencies allowed).
  Also hoist nops (which groups together) for ease of reading.
******************************************************************************/
struct SchedHeuristicOriginalOrder
    : public SchedHeuristic<SchedDirection::TopDown> {

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    if (auto selected = findPreferredOpCategory(a, b, opCategoryNop)) {
      return selected;
    }
    return getOriginalOrder<SchedDirection::TopDown>(a, b);
  }

  StringRef name() { return "OriginalOrder"; }
};

/******************************************************************************
  SchedManager
  - Tracks dependencies.
  - Prepared dag for scheduling.
  - Runs scheduler.
******************************************************************************/
struct SchedManager {
  SchedManager(Block *block) : dag(block), rescheduleId(0) {
    LDBG("SchedManager()");
  }

  // Calculate new deps based on op order and previously determined deps.
  // Insert new deps into dep map and apply them to dat.
  void addDeps(std::unique_ptr<DependencyCalculator> depCalc) {
    depCalc->addDepsToDag(&dag);
  }

  // Calculate new node priorities based on dag.
  void addPriorities(std::unique_ptr<PriorityCalculator> prioCalc) {
    prioCalc->addPrioritiesToDag(&dag);
  }

  template <SchedDirection Direction>
  void reschedule(SchedHeuristic<Direction> *heuristic) {
    LDBG("SchedManager::reschedule("
         << rescheduleId << "), Direction="
         << ((Direction == SchedDirection::TopDown) ? "TopDown" : "BottomUp")
         << ", Heuristic=" << heuristic->name());
    // Reset deps right before rescheduling b/c analysis passes
    // may have altered them.
    LDBG("dag.resetDeps() before rescheduling");
    dag.resetDeps();
    LDBG("NodeList before reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(dag.dumpNodes(llvm::dbgs()));
    LDBG("SchedDag before reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(dag.dumpDotFormat(llvm::dbgs()));

    // Node readiness is based on direction.
    dag.initReadyNodes<Direction>();
    heuristic->reset();

    // Schedule the dag; this process removes deps from nodes.
    // Store nodes in newly scheduled order.
    SchedDagNodeList rescheduledNodes;
    const bool printDetails = rescheduleId==1;
    for (int iter = 0; !dag.finished(); ++iter) {
      // Print ReadyNodes and HeuristicState
      const auto &readyNodes = dag.getReadyNodes();
      if (printDetails) {
        LDBG("Iter: " << iter);
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
      heuristic->notifySelected(selectedNode);

      // Place selected node in list, remove it from dag which updates
      // readyList.
      rescheduledNodes.push_back(selectedNode);
      dag.removeScheduledNode<Direction>(selectedNode);
    }

    // After scheduling, re-apply deps to prepare for adding additional deps.
    LDBG("dag.resetDeps() after rescheduling");
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

  SmallVector<Operation *> getOpList() {
    SmallVector<Operation *> opList;
    for (auto node : dag.nodeList) {
      Operation *op = node->getOp();
      opList.push_back(op);
    }
    return opList;
  }

  SchedDag dag;
  int32_t rescheduleId;
}; // SchedManager

/******************************************************************************
  TritonAMDGPURescheduleOps::applyReschedulingPasses()
  Top-level scheduling pass for a single block,
  whose purpose is to improve ttgir op order, and thereby improve llir op order
  to help improve scheduling and regalloc of backend compilers.
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
    if (mlirBlock->getOperations().size() < 3)
      return failure();
    return success();
  }

  void applyReschedulingPasses(Block *mlirBlock) {
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses()");

    SchedManager schedManager(mlirBlock);

    /*
      Scheduling Pass 0
      - Dependencies: Data, DotOrder, LocalStoreOrder, Barriers
      - Priorities: DotCriticalPath, LocalStoreCriticalPath
      - Heuristic: Priority
    */
    // Add Data deps based on def-use chains.
    schedManager.addDeps(std::make_unique<DataDependencyCalculator>());
    //LLVM_DEBUG(
    //  LDBG("Dag w/ only data dependencies");
    //  schedManager.dag.dumpDotFormat(llvm::dbgs());
    //);
    // Assume dots are in ideal order and propagate their critical path.
    schedManager.addDeps(
      std::make_unique<DotOrderDependencyCalculator>());
    schedManager.addPriorities(std::make_unique<DotCriticalPathPriorityCalculator>());
    // Assume local_stores are in ideal order and propagate their critical path.
    schedManager.addDeps(
      std::make_unique<LocalStoreOrderDependencyCalculator>());
    schedManager.addPriorities(std::make_unique<LocalStoreCriticalPathPriorityCalculator>());
    // Add dependencies for barriers (gpu.barrier, sched.barrier, setprio...).
    schedManager.addDeps(std::make_unique<BarrierDependencyCalculator>());
    SchedHeuristicPriority<SchedDirection::TopDown> shp;
    schedManager.reschedule<SchedDirection::TopDown>(&shp);

    /*
      Scheduling Pass 1
      - Dependencies: LocalLoadOrder, GlobalLoadOrder
      - Priorities: 0
      - Heuristic: LdsOps(MachineModel)
    */

    // Preserve memory op order determined by critical paths above.
    schedManager.addDeps(
        std::make_unique<LocalLoadOrderDependencyCalculator>());
    schedManager.addDeps(
        std::make_unique<GlobalLoadOrderDependencyCalculator>());
    // Now that we've ordered lds ops, spread them out with machine model.
    SchedHeuristicLdsOps<SchedDirection::BottomUp> shl;
    schedManager.reschedule<SchedDirection::BottomUp>(&shl);

    /*
      Scheduling Pass 2
      - Dependencies: DotLdsOrder; PriorityOrder
      - Priorities: 0
      - Heuristic: EarlyGlobalLoads
    */
    schedManager.addDeps(
      std::make_unique<DotLdsOrderDependencyCalculator>());
    // TODO(dtanner) - flash-attention needs the below deps pass implemented.
    //schedManager.addDeps(std::make_unique<LocalStoreGlobalLoadAntiDepsDependencyCalculator>());

    // keep this; it did seem to be working?
    schedManager.addDeps(
        std::make_unique<PriorityOrderDependencyCalculator>());
    schedManager.addDeps(std::make_unique<MemOrderDependencyCalculator>());
    SchedHeuristicGlobalLoadsEarly<SchedDirection::BottomUp> shg;
    schedManager.reschedule<SchedDirection::BottomUp>(&shg);

    /*
     (F) Final rescheduling restores original order except for dependencies.
    */
    //SchedHeuristicOriginalOrder sho;
    //schedManager.reschedule<SchedDirection::TopDown>(&sho);

    /*
      After Rescheduling, apply op order to block.
    */
    LDBG("Rescheduled Ops:");
    SmallVector<Operation *> rescheduledOps = schedManager.getOpList();
    // Print op (and not node) list.
    LLVM_DEBUG(for (auto op : rescheduledOps) {
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    // Apply schedule to basic block.
    for (auto it = rescheduledOps.rbegin(); it != rescheduledOps.rend(); ++it) {
      (*it)->moveBefore(mlirBlock, mlirBlock->begin());
    }
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses() - DONE");
  }

  void runOnOperation() override {
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
          LLVM_DEBUG((*it).print(llvm::dbgs());
          llvm::dbgs() << "\n";);
        }
        applyReschedulingPasses(block);
        LDBG("OpList after applyReschedulingPasses()");
        for (auto it = block->begin(); it != block->end(); ++it) {
          LLVM_DEBUG((*it).print(llvm::dbgs());
          llvm::dbgs() << "\n";);
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
