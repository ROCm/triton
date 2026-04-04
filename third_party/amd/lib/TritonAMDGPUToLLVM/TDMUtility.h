#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <optional>

using mlir::triton::AMD::TargetInfo;

namespace mlir::LLVM::AMD {

// Structure to hold TDM descriptor groups as LLVM vector Values.
// group0 is <4 x i32>, group1 is <8 x i32>, group2/group3 are <4 x i32>.
struct TDMDescriptor {
  Value group0; // <4 x i32>
  Value group1; // <8 x i32>
  std::optional<Value> group2; // <4 x i32>
  std::optional<Value> group3; // <4 x i32>

  // Get all groups as a vector of Values (for packLLElements)
  SmallVector<Value> getAllGroups() const;
};

// Helper: extract an i32 element from a vector Value.
inline Value vecGet(TritonLLVMOpBuilder &b, Value vec, int idx) {
  return b.extract_element(vec, b.i32_val(idx));
}

// Helper: insert an i32 element into a vector Value, returning the updated
// vector.
inline Value vecSet(TritonLLVMOpBuilder &b, Value vec, int idx, Value val) {
  return b.insert_element(vec, val, b.i32_val(idx));
}

// Create a TDM descriptor. This creates a partially filled descriptor, with
// shared memory address and pred set to zero. User of the descriptor is
// expected to fill these fields later.
// For 1D-2D tensors: returns TDMDescriptor with only group0 and group1
// For 3D-5D tensors: returns TDMDescriptor with all groups populated
TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType,
                                  SmallVector<int64_t> blockShape, int numWarps,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride, Value srcPtr,
                                  bool isRowMajor);

// Update the global memory address with offset, and fill the shared memory
// address and pred in a given TDM descriptor for regular load/store (1D-5D).
// For partitioned shared memory, dstPtrs contains multiple base pointers and
// the correct one is selected based on sharedLayout's partition dimension.
// group0 is <4 x i32>, group1 is <8 x i32>, group2/group3 are <4 x i32>.
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount, Value &group0, Value &group1,
    std::optional<std::reference_wrapper<Value>> group2,
    std::optional<std::reference_wrapper<Value>> group3,
    SmallVector<Value> offset, ArrayRef<Value> dstPtrs, Value pred,
    Value multicastMask, Value barrierPtr,
    const triton::LinearLayout &sharedLayout, Value ctaId, bool isStore,
    bool isRowMajor);

// Fill TDM descriptor for gather/scatter operations (2D only).
// Gather reads from non-contiguous rows in global memory to LDS.
// Scatter writes from LDS to non-contiguous rows in global memory.
// - rowIndices: which global rows to read from (gather) or write to (scatter)
// - ldsRowOffset: starting row within shared memory
// - globalColOffset: starting column in global memory
// - use32BitIndices: true for 32-bit indices (max 8 rows), false for 16-bit
// (max 16 rows)
void fillTDMDescriptorForGatherScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, unsigned padInterval, unsigned padAmount,
    Value &group0, Value &group1, Value &group2, Value &group3,
    Value ldsRowOffset, Value globalColOffset, Value ldsPtr, Value pred,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices);

// Emit a TDM load or store operation for regular (non-scatter) transfers.
// Supports 1D-5D tensors with contiguous access patterns.
// - offset: the starting position in global memory for each dimension
// - dstPtrs: base pointers to shared memory (multiple for partitioned encoding)
// - sharedLayout: the full shared memory LinearLayout (for partition selection)
// - isLoad: true for global->LDS, false for LDS->global
// desc contains [group0, group1] or [group0, group1, group2, group3] vectors.
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> shapePerCTA,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, ArrayRef<Value> dstPtrs,
                      Value pred, Value multicastMask, Type elementType,
                      Value barrierPtr, bool isLoad,
                      const triton::LinearLayout &sharedLayout, Value ctaId,
                      bool isRowMajor);

// Calculate the number of TDM gather/scatter instructions needed.
// - numIndices: number of row indices
// - use32BitIndices: true for 32-bit indices (max 8 rows/instr), false for
//   16-bit (max 16 rows/instr)
// Returns: the number of TDM instructions that will be emitted
size_t getTDMGatherScatterInstrinsicCount(size_t numIndices,
                                          bool use32BitIndices);

// Emit a TDM gather or scatter operation for non-contiguous row access.
// Gather: reads from non-contiguous global rows into LDS
// Scatter: writes from LDS to non-contiguous global rows
// - ldsPtr: pointer to shared memory (destination for gather, source for
// scatter)
// - rowIndices: which global rows to read from (gather) or write to (scatter)
// - colOffset: starting column offset in global memory
// - use32BitIndices: true for 32-bit indices (max 8 rows/instr), false for
//   16-bit (max 16 rows/instr)
// - isGather: true for gather (global->LDS), false for scatter (LDS->global)
// Multiple TDM instructions are issued automatically if more rows are needed.
// desc contains [group0, group1] vectors.
void emitTDMGatherScatter(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                          unsigned padInterval, unsigned padAmount,
                          Value ldsPtr, Value pred, Type elementType,
                          Value barrierPtr,
                          const triton::LinearLayout &cgaLayout, Value ctaId,
                          ArrayRef<Value> rowIndices, Value colOffset,
                          bool use32BitIndices, bool isGather);

// Emit prefetches for a TDM tile to make it available for an actual load in
// the future. Data is prefetched cooperatively across all CTAs, warps, and
// lanes to cover the entire TDM tile.
// Returns the prefetched memory offsets. This should only be used for testing
// purposes.
// desc contains [group0, group1] or [group0, group1, group2, group3] vectors.
SmallVector<Value> emitTDMPrefetch(RewriterBase &rewriter, Location loc,
                                   ArrayRef<Value> desc,
                                   ArrayRef<int64_t> blockShape, int numLanes,
                                   int numWarps, int numCTAs,
                                   ArrayRef<Value> offset, Value pred,
                                   Type elementType, Value laneId, Value warpId,
                                   Value ctaId, bool isSpeculative);

// Swap the trailing two dimensions of a vector for TDM operations.
template <typename T> inline void swapTrailingDims(SmallVector<T> &vec) {
  assert(vec.size() >= 2 && "need at least 2 dims to swap");
  std::swap(vec[vec.size() - 2], vec[vec.size() - 1]);
}

// Swap two output dimension names in a LinearLayout (for col-major support).
triton::LinearLayout swapOutDimSemantics(const triton::LinearLayout &layout,
                                         StringAttr dimA, StringAttr dimB);

// Incrementally advance a TDM descriptor's global_addr and tensor_dim.
// The descriptor must have per-warp offsets already applied.
// group0 (<4 x i32>) and group1 (<8 x i32>) are modified in-place.
void advanceTDMDescriptor(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          Value &group0, Value &group1,
                          ArrayRef<Value> offsets, size_t numDims,
                          Type elementType, bool isRowMajor = true,
                          bool updateBounds = true);

// Emit a lightweight TDM load from a pre-positioned descriptor.
// Only sets lds_addr, pred, barrier, and multicast — does NOT compute
// warp offsets or modify global_addr/tensor_dim.
// desc contains [group0, group1] or [group0, group1, group2, group3] vectors.
void emitTDMLoadFromAdvanced(RewriterBase &rewriter, Location loc,
                             const LLVMTypeConverter *typeConverter,
                             ArrayRef<Value> desc,
                             ArrayRef<int64_t> shapePerCTA, int numWarps,
                             unsigned padInterval, unsigned padAmount,
                             ArrayRef<Value> dstPtrs, Value pred,
                             Value multicastMask, Type elementType,
                             Value barrierPtr,
                             const triton::LinearLayout &sharedLayout,
                             Value ctaId, bool isRowMajor);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
