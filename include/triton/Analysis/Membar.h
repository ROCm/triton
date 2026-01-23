#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"

#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <tuple>

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

/// Represents a buffer index expression for multi-buffered memdesc_index.
///
/// Models expressions of the form: (baseValue + constantOffset)
/// - baseValue: Typically a loop-carried block argument, or nullptr for
/// constants
/// - constantOffset: Constant integer offset from the base
///
/// This enables proving that two accesses target different buffer slots within
/// the same loop iteration. For example, if we have:
///   - Read from slot[i]   -> BufferIndexExpr{base=i, offset=0}
///   - Write to slot[i+1]   -> BufferIndexExpr{base=i, offset=1}
/// We can prove they don't intersect (same base, different offsets).
///
/// Note: This only works for same-iteration comparisons. Cross-iteration
/// dependencies are handled separately via loop-carried slices.
struct BufferIndexExpr {
  /// The base value (typically a loop-carried block argument).
  /// nullptr indicates a pure constant index (offset only).
  Value baseValue;
  /// Constant offset added to the base value.
  int64_t constantOffset = 0;

  /// Returns true if this expression has a non-constant base.
  bool hasBase() const { return baseValue != nullptr; }

  /// Check if this expression is provably different from another.
  ///
  /// Returns true only if both expressions share the same base value and
  /// have different constant offsets. This proves they access different
  /// buffer slots within the same iteration.
  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    // Must be relative to the same base value (including both being nullptr)
    if (baseValue != other.baseValue)
      return false;
    // Different offsets from the same base = different slots
    return constantOffset != other.constantOffset;
  }

  bool operator==(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           constantOffset == other.constantOffset;
  }

  bool operator<(const BufferIndexExpr &other) const {
    if (baseValue.getAsOpaquePointer() != other.baseValue.getAsOpaquePointer())
      return baseValue.getAsOpaquePointer() <
             other.baseValue.getAsOpaquePointer();
    return constantOffset < other.constantOffset;
  }
};

// Represents the access to a slice of an allocation
// It contains information both on physical memory (the interval) and a
// logical view on it (layout, subslice offsets and shape for the access)
struct AllocationSlice {
public:
  // Create allocation slice from a value, collecting subslice offsets
  AllocationSlice(Value value, Interval<size_t> allocationInterval);

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, ..)
  AllocationSlice(Interval<size_t> interval)
      : allocationInterval(interval), accessTy(nullptr) {}

  bool operator<(const AllocationSlice &other) const {
    return asTuple() < other.asTuple();
  }

  bool operator==(const AllocationSlice &other) const {
    return asTuple() == other.asTuple();
  }

  // Check if a AllocationSlice intersects with another other.
  // This happens if their subslice regions intersect in all dimensions.
  // Returns true if it can't prove the AllocationSlices are disjoint.
  bool intersects(const AllocationSlice &other) const;

  void print(raw_ostream &os) const;

  // Create a copy of this slice marked as loop-carried.
  // Used when propagating slices across loop backedges.
  AllocationSlice createLoopCarriedCopy() const {
    AllocationSlice copy = *this;
    copy.isLoopCarried = true;
    return copy;
  }

  // Get the buffer index expression (for use in intersection checks)
  const std::optional<BufferIndexExpr> &getBufferIndexExpr() const {
    return bufferIndexExpr;
  }

  // Check if this slice is marked as loop-carried
  bool getIsLoopCarried() const { return isLoopCarried; }

private:
  std::tuple<Interval<size_t>, const void *, llvm::ArrayRef<int64_t>,
             std::optional<BufferIndexExpr>, bool>
  asTuple() const {
    return {allocationInterval, accessTy.getAsOpaquePointer(), subsliceOffsets,
            bufferIndexExpr, isLoopCarried};
  }
  // Offsets from subslice. Empty when offsets are unknown
  SmallVector<int64_t> subsliceOffsets;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the memory descriptor for this access
  triton::gpu::MemDescType accessTy;
  // Buffer index expression for multi-buffered accesses.
  // Used to prove non-intersection when comparing accesses within
  // the same loop iteration (e.g., read from slot i, write to slot i+1).
  std::optional<BufferIndexExpr> bufferIndexExpr;
  // True if this slice originates from a previous loop iteration (via
  // backedge). Expression matching is disabled for loop-carried slices.
  bool isLoopCarried = false;
};

/// Tracks memory access slices for barrier insertion analysis.
///
/// Slices are marked with an `isLoopCarried` flag to distinguish
/// intra-iteration from cross-iteration dependencies:
///
/// - Current-iteration slices: Can use expression matching (BufferIndexExpr) to
///   prove non-intersection (e.g., slot[i] vs slot[i+1] are provably
///   different).
///
/// - Loop-carried slices: Originate from previous iterations via loop
/// backedges.
///   Expression matching is disabled for these since cross-iteration
///   dependencies may alias even with different expressions (e.g., write to
///   slot[i+1] in iteration N conflicts with read from slot[i+1] in iteration
///   N+1).
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  /// Memory access slices, tagged with loop-carried status via AllocationSlice
  /// flag. Slices from current iteration and previous iterations are stored
  /// together, distinguished by the isLoopCarried flag.
  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  BlockInfo() = default;

  /// Unions two BlockInfo objects, merging slices.
  BlockInfo &join(const BlockInfo &other) {
    joinSlices(syncReadSlices, other.syncReadSlices);
    joinSlices(syncWriteSlices, other.syncWriteSlices);
    return *this;
  }

  /// Unions two BlockInfo objects, marking all incoming slices as loop-carried.
  /// Used when propagating state across loop backedges to distinguish
  /// cross-iteration dependencies from intra-iteration ones.
  BlockInfo &joinLoopCarried(const BlockInfo &other) {
    joinSlicesAsLoopCarried(syncReadSlices, other.syncReadSlices);
    joinSlicesAsLoopCarried(syncWriteSlices, other.syncWriteSlices);
    return *this;
  }

private:
  /// Helper to merge slice maps.
  static void joinSlices(SliceMapT &lhs, const SliceMapT &rhs) {
    for (const auto &[slice, ops] : rhs)
      lhs[slice].insert(ops.begin(), ops.end());
  }

  /// Helper to merge slice maps, marking incoming slices as loop-carried.
  static void joinSlicesAsLoopCarried(SliceMapT &lhs, const SliceMapT &rhs) {
    for (const auto &[slice, ops] : rhs) {
      AllocationSlice loopCarriedSlice = slice.createLoopCarriedCopy();
      lhs[loopCarriedSlice].insert(ops.begin(), ops.end());
    }
  }

public:
  void dump() {
    auto &err = llvm::errs();
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[slice, ops] : syncReadSlices) {
      err << "    ";
      slice.print(err);
      if (slice.getIsLoopCarried())
        err << " [loop-carried]";
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[slice, ops] : syncWriteSlices) {
      err << "    ";
      slice.print(err);
      if (slice.getIsLoopCarried())
        err << " [loop-carried]";
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
  }

  /// Returns true if Slices in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const {
    return /*RAW*/ isIntersected(syncWriteSlices, other.syncReadSlices,
                                 filter) ||
           /*WAR*/
           isIntersected(syncReadSlices, other.syncWriteSlices, filter) ||
           /*WAW*/
           isIntersected(syncWriteSlices, other.syncWriteSlices, filter);
  }

  /// Clears the slices because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadSlices == other.syncReadSlices &&
           syncWriteSlices == other.syncWriteSlices;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const SliceMapT &lhsSlices, const SliceMapT &rhsSlices,
                     MembarFilterFn filter) const {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          for (auto lhsOp : lhs.second)
            for (auto rhsOp : rhs.second)
              if (!filter || !filter(lhsOp, rhsOp))
                return true;
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;
  struct SuccessorInfo {
    VirtualBlock block;
    bool isBackedge = false;
  };

public:
  using FuncBlockInfoMapT = triton::CallGraph<BlockInfo>::FuncDataMapT;
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  MembarOrFenceAnalysis() = default;
  explicit MembarOrFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : allocation(allocation), filter(filter) {}

  virtual ~MembarOrFenceAnalysis() = default;

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

protected:
  /// Applies the barrier analysis based on the SCF dialect, in which each
  /// region has a single basic block only.
  /// Example:
  /// region1
  ///   op1
  ///   op2 (scf.if)
  ///      region2
  ///        op3
  ///        op4
  ///      region3
  ///        op5
  ///        op6
  ///   op7
  /// TODO: Explain why we don't use ForwardAnalysis:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation, DominanceInfo &domInfo,
                       SmallVector<SuccessorInfo> &successors);

  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) = 0;

  Allocation *allocation = nullptr;
  MembarFilterFn filter = nullptr;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
  MembarAnalysis() = default;
  explicit MembarAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

  ~MembarAnalysis() override = default;

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisType>
class ModuleMembarOrFenceAnalysis : public triton::CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation *moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : triton::CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation), filter(filter) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            AnalysisType analysis(allocation, filter);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
  MembarFilterFn filter;
};

typedef ModuleMembarOrFenceAnalysis<MembarAnalysis> ModuleMembarAnalysis;

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
