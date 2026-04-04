#include "TDMUtility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include <optional>

// Include shared C-compatible TDM utilities
#include "../../backend/include/TDMCommon.h"

namespace mlir::LLVM::AMD {
namespace {

// Helper to decode a value spanning two 32-bit words
static Value decode48BitValue(TritonLLVMOpBuilder &b, Value group, int startIdx) {
  Value low = b.lshr(vecGet(b, group, startIdx), b.i32_val(16));
  Value high = b.shl(vecGet(b, group, startIdx + 1), b.i32_val(16));
  return b.or_(low, high);
}

// C++ wrapper for the shared tdmGetWarpDistribution function.
SmallVector<unsigned> getWarpDistribution(ArrayRef<int64_t> blockShape,
                                          int numWarps) {
  int numDims = blockShape.size();
  SmallVector<int> warps(numDims);
  tdmGetWarpDistribution(blockShape.data(), numDims, numWarps, warps.data());
  return SmallVector<unsigned>(warps.begin(), warps.end());
}

} // namespace

SmallVector<Value> TDMDescriptor::getAllGroups() const {
  SmallVector<Value> result;
  result.push_back(group0);
  result.push_back(group1);
  if (group2.has_value())
    result.push_back(group2.value());
  if (group3.has_value())
    result.push_back(group3.value());
  return result;
}

// Decode a full TDM descriptor from group vectors for 1D-5D tensors.
// Returns (base, tensorShape[], tensorStride[], blockShape[])
std::tuple<Value, SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
decodeTDMDescriptorFull(RewriterBase &rewriter, Location loc,
                        Value group0, Value group1,
                        std::optional<Value> group2,
                        std::optional<Value> group3, size_t numDims) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type globalPtrTy = ptr_ty(ctx, 1);

  // Decode base address from group0
  Value globalAddrLow = vecGet(b, group0, 2);
  Value globalAddrHigh = b.and_(vecGet(b, group0, 3), b.i32_val(0x7FFFFFFF));
  globalAddrLow = b.zext(i64_ty, globalAddrLow);
  globalAddrHigh = b.shl(b.zext(i64_ty, globalAddrHigh), b.i64_val(32));
  Value globalAddr = b.or_(globalAddrLow, globalAddrHigh);
  Value srcPtr = b.inttoptr(globalPtrTy, globalAddr);

  SmallVector<Value> tensorShape(numDims);
  SmallVector<Value> tensorStride(numDims);
  SmallVector<Value> blockShape(numDims);

  // Decode dimensions from the end (inner dimensions first)
  tensorShape[numDims - 1] = decode48BitValue(b, group1, 1);

  if (numDims >= 2) {
    tensorShape[numDims - 2] = decode48BitValue(b, group1, 2);
    tensorStride[numDims - 2] = vecGet(b, group1, 5);

    if (numDims >= 3) {
      Value stride1Low =
          b.and_(b.lshr(vecGet(b, group1, 6), b.i32_val(16)), b.i32_val(0xFFFF));
      Value stride1High = b.and_(vecGet(b, group1, 7), b.i32_val(0xFFFF));
      tensorStride[numDims - 3] =
          b.or_(stride1Low, b.shl(stride1High, b.i32_val(16)));
    }
  }

  // tensor_dim2_stride from group2[2]
  if (numDims >= 4)
    tensorStride[numDims - 4] = vecGet(b, group2.value(), 2);

  // tensor_dim3_stride from group3[0]
  if (numDims == 5)
    tensorStride[numDims - 5] = vecGet(b, group3.value(), 0);

  // The innermost dimension always has stride 1
  tensorStride[numDims - 1] = b.i32_val(1);

  // Block shapes from group1
  blockShape[numDims - 1] =
      b.and_(b.lshr(vecGet(b, group1, 3), b.i32_val(16)), b.i32_val(0xFFFF));
  if (numDims >= 2) {
    Value g1_4 = vecGet(b, group1, 4);
    blockShape[numDims - 2] = b.and_(g1_4, b.i32_val(0xFFFF));

    // 3rd dimension from group2 if present
    if (numDims >= 3) {
      tensorShape[numDims - 3] = vecGet(b, group2.value(), 0);
      blockShape[numDims - 3] =
          b.and_(b.lshr(g1_4, b.i32_val(16)), b.i32_val(0xFFFF));
    }
  }

  // 4th dimension from group2/group3 if present
  if (numDims >= 4) {
    tensorShape[numDims - 4] = vecGet(b, group2.value(), 1);
    blockShape[numDims - 4] =
        b.and_(b.lshr(vecGet(b, group2.value(), 3), b.i32_val(16)),
               b.i32_val(0xFFFF));
  }

  // 5th dimension from group3 if present
  if (numDims == 5) {
    Value g3_2 = vecGet(b, group3.value(), 2);
    Value tensorDim4Low =
        b.and_(b.lshr(vecGet(b, group3.value(), 1), b.i32_val(16)),
               b.i32_val(0xFFFF));
    Value tensorDim4High = b.and_(g3_2, b.i32_val(0xFFFF));
    tensorShape[0] =
        b.or_(tensorDim4Low, b.shl(tensorDim4High, b.i32_val(16)));
    blockShape[0] =
        b.and_(b.lshr(g3_2, b.i32_val(16)), b.i32_val(0xFFFF));
  }

  return {srcPtr, tensorShape, tensorStride, blockShape};
}

TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType,
                                  SmallVector<int64_t> blockShape, int numWarps,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride, Value srcPtr,
                                  bool isRowMajor) {
  size_t numDims = tensorShape.size();
  assert(numDims >= 1 && numDims <= 5 && tensorStride.size() == numDims &&
         "TDM only supported for 1D-5D tensors.");
  assert(blockShape.size() == tensorStride.size() &&
         blockShape.size() == numDims &&
         "Block/tensor/stride dim count must all be equal.");
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (!isRowMajor) {
    swapTrailingDims(blockShape);
    swapTrailingDims(tensorStride);
    swapTrailingDims(tensorShape);
  }

  // Define common values for better readability
  Value v16 = b.i32_val(16);
  Value v32 = b.i64_val(32);
  Value mask16 = b.i32_val(0xFFFF);
  Value mask31 = b.i32_val(0x7FFFFFFF);

  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  auto elementSizeInBytes = elementBitWidth / 8;

  // Cast strides from i64 to i32
  for (size_t i = 0; i < numDims; ++i)
    tensorStride[i] = b.trunc(i32_ty, tensorStride[i]);

  // Distribute block among warps
  {
    int64_t blkShapePerWarp[5];
    tdmGetAdjustedBlockShape(blockShape.data(), numDims, numWarps,
                             &blkShapePerWarp[0]);
    blockShape.assign(blkShapePerWarp, blkShapePerWarp + blockShape.size());
  }

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred (to be filled later)
  // [30]:      Scatter/gather index size (0=16-bit, 1=32-bit)
  // [31]:      Scatter/gather enable (0=disabled, 1=enabled)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
  Value group0 = b.null(v4i32Ty);
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddr));
  Value addrHigh = b.trunc(i32_ty, b.lshr(globalAddr, v32));
  addrHigh = b.or_(addrHigh, b.i32_val(1 << 31));
  group0 = vecSet(b, group0, 3, addrHigh);

  /* group1 bit-field definition:

    NOTE that in this chart
    - {tensor|tile}-dim0 for means innermost dimension.
    - stride-dim0 refers to the stride of the 2nd innermost dimension.
      FIXME: Is the stride for innermost dimension always 1, and hence no
      need to set in the descriptor

    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ------------------------------------------------
      0      0          16         multicast mask
             16         2          data size - log2(element size in bytes)
             18         1          atomic barrier enable
             19         1          iterate enable
             20         1          pad enable
             22         3          pad interval
                                   (log2(pad interval in dwords) - 1)
             25         7          pad amount - pad amount in dwords - 1
                                   (pad amount in dwords - 1)
     ---------------------------------------------------------
     1       0          16         atomic barrier address
             16         16         tensor_dim0 (low-16-bit)
     --------------------------------------------------------
     2       0           16        tensor_dim0 (high-16-bit)
             16          16        tensor_dim1 (low-16-bit)
     ----------------------------------------------------------
     3       0           16        tensor_dim1 (high-16-bit)
             16          16        tile_dim0
     -------------------------------------------------------
     4       0           16        tile_dim1
             16          16        tile_dim2
     -------------------------------------------------------
     5       0           32        tensor_dim0_stride(low-32-bit)
     -------------------------------------------------------
     6       0           16        tensor_dim0_stride(high-16-bit)
            16           16        tensor_dim1_stride(low-16-bit)
     -------------------------------------------------------------
     7       0           32        tensor_dim1_stride(high-16-bit)
     ================================================================
  */
  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
  Value group1 = b.null(v8i32Ty);
  int32_t dataSize = log2(elementSizeInBytes);
  unsigned dwordSize = 32;
  auto padIntervalInDwords = padInterval * elementBitWidth / dwordSize;
  auto padAmountInDwords = padAmount * elementBitWidth / dwordSize;

  Value g1_0 = b.i32_val(dataSize << 16);
  if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
    assert(llvm::isPowerOf2_32(padIntervalInDwords));
    int32_t log2PadInterval = log2(padIntervalInDwords);
    g1_0 = b.or_(g1_0, b.i32_val(1 << 20));
    g1_0 = b.or_(g1_0, b.i32_val((log2PadInterval - 1) << 22));
    g1_0 = b.or_(g1_0, b.i32_val((padAmountInDwords - 1) << 25));
  }
  group1 = vecSet(b, group1, 0, g1_0);

  // Encode 32-bit tensor shapes
  group1 = vecSet(b, group1, 1, b.shl(tensorShape[numDims - 1], v16));
  group1 = vecSet(b, group1, 2, b.lshr(tensorShape[numDims - 1], v16));

  if (numDims >= 2) {
    group1 = vecSet(b, group1, 2,
                    b.or_(vecGet(b, group1, 2),
                          b.shl(tensorShape[numDims - 2], v16)));
    group1 = vecSet(b, group1, 3, b.lshr(tensorShape[numDims - 2], v16));
  }

  // Block shapes
  Value g1_3 = vecGet(b, group1, 3);
  g1_3 = b.or_(g1_3, b.i32_val(blockShape[numDims - 1] << 16));
  group1 = vecSet(b, group1, 3, g1_3);

  if (numDims >= 2) {
    Value g1_4 = b.i32_val(blockShape[numDims - 2] & 0xFFFF);
    if (numDims >= 3)
      g1_4 = b.or_(g1_4, b.i32_val(blockShape[numDims - 3] << 16));
    group1 = vecSet(b, group1, 4, g1_4);
  }

  // Handle strides
  if (numDims >= 2) {
    group1 = vecSet(b, group1, 5, tensorStride[numDims - 2]);
    if (numDims >= 3) {
      group1 = vecSet(b, group1, 6,
                      b.shl(tensorStride[numDims - 3], v16));
      group1 = vecSet(b, group1, 7,
                      b.lshr(tensorStride[numDims - 3], v16));
    }
  }

  if (numDims <= 2)
    return TDMDescriptor{group0, group1, std::nullopt, std::nullopt};

  /* For 3D-5D tensors, fill group2 and group3 */
  Value group2 = b.null(v4i32Ty);
  if (numDims >= 3)
    group2 = vecSet(b, group2, 0, tensorShape[numDims - 3]);

  if (numDims >= 4) {
    group2 = vecSet(b, group2, 1, tensorShape[numDims - 4]);
    group2 = vecSet(b, group2, 2, tensorStride[numDims - 4]);
    group2 = vecSet(b, group2, 3,
                    b.shl(b.i32_val(blockShape[numDims - 4]), v16));
  }

  Value group3 = b.null(v4i32Ty);
  if (numDims >= 4) {
    if (numDims == 5) {
      group3 = vecSet(b, group3, 1,
                      b.shl(tensorShape[numDims - 5], v16));
      group3 = vecSet(b, group3, 2,
                      b.lshr(tensorShape[numDims - 5], v16));
      group3 = vecSet(b, group3, 0, tensorStride[numDims - 5]);
      group3 = vecSet(b, group3, 2,
                      b.or_(vecGet(b, group3, 2),
                            b.shl(b.i32_val(blockShape[numDims - 5]), v16)));
    }
  }

  return TDMDescriptor{group0, group1, group2, group3};
}

// Returns a copy of `layout` where the semantics of dimA and dimB are
// exchanged.
triton::LinearLayout
swapOutDimSemantics(const triton::LinearLayout &layout, StringAttr dimA,
                    StringAttr dimB) {
  assert(layout.hasOutDim(dimA));
  assert(layout.hasOutDim(dimB));
  SmallVector<std::pair<StringAttr, int32_t>> renamedOutDims;
  for (auto [name, size] : layout.getOutDims()) {
    if (name == dimA)
      renamedOutDims.push_back({dimB, size});
    else if (name == dimB)
      renamedOutDims.push_back({dimA, size});
    else
      renamedOutDims.push_back({name, size});
  }
  return triton::LinearLayout(layout.getBases(), renamedOutDims,
                              /*requireSurjective=*/false)
      .transposeOuts(llvm::to_vector(layout.getOutDimNames()));
}

// Fill TDM descriptor for regular load/store operations (1D-5D tensors)
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> shapePerCTA, int numWarps, unsigned padInterval,
    unsigned padAmount, Value &group0, Value &group1,
    std::optional<std::reference_wrapper<Value>> group2,
    std::optional<std::reference_wrapper<Value>> group3,
    SmallVector<Value> offset, ArrayRef<Value> dstPtrs, Value pred,
    Value multicastMask, Value barrierPtr,
    const triton::LinearLayout &sharedLayout, Value ctaId, bool isStore,
    bool isRowMajor) {
  size_t numDims = offset.size();
  assert(numDims >= 1 && numDims <= 5 && "TDM supports 1D to 5D tensors.");
  assert(!dstPtrs.empty() && "dstPtrs cannot be empty");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  std::optional<triton::LinearLayout> adjustedSharedLayout;
  if (!isRowMajor) {
    swapTrailingDims(shapePerCTA);
    swapTrailingDims(offset);
    if (numDims >= 2) {
      auto dimN_2 = StringAttr::get(ctx, "dim" + std::to_string(numDims - 2));
      auto dimN_1 = StringAttr::get(ctx, "dim" + std::to_string(numDims - 1));
      adjustedSharedLayout = swapOutDimSemantics(sharedLayout, dimN_2, dimN_1);
    }
  }
  const auto &tdmViewSharedLayout =
      adjustedSharedLayout ? *adjustedSharedLayout : sharedLayout;

  // Decode the full TDM descriptor to get all values
  auto [srcPtr, tensorShape, tensorStride, decodedBlockShape] =
      decodeTDMDescriptorFull(
          rewriter, loc, group0, group1,
          group2.has_value()
              ? std::optional<Value>(group2.value().get())
              : std::nullopt,
          group3.has_value()
              ? std::optional<Value>(group3.value().get())
              : std::nullopt,
          numDims);

  auto kMessage = str_attr("message");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  auto kOffset = str_attr("offset");
  auto kPartition = str_attr("partition");

  auto cgaLayout = triton::gpu::SharedLinearEncodingAttr::get(
                       ctx, tdmViewSharedLayout, /*layoutAlignment=*/16)
                       .getCGALayout()
                       .getLinearLayout();

  auto warpsPerCTA = getWarpDistribution(shapePerCTA, numWarps);
  auto tdmLayout =
      triton::gpu::getTDMLinearLayout(shapePerCTA, warpsPerCTA, cgaLayout);

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  Value baseOffset = b.i32_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimOffset = b.mul(offset[i], tensorStride[i]);
    baseOffset = b.add(baseOffset, dimOffset);
  }
  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, baseOffset);

  auto tdmToShared = tdmLayout.invertAndCompose(tdmViewSharedLayout);
  auto sharedOffsets = applyLinearLayout(
      loc, rewriter, tdmToShared,
      {{kMessage, b.i32_val(0)}, {kWarp, warpId}, {kBlock, ctaId}});

  Value dstOffset = b.i32_val(0);
  Value partitionIdx = b.i32_val(0);
  bool isPartitioned = tdmToShared.hasOutDim(kPartition);
  for (auto &[name, val] : sharedOffsets) {
    if (name == kOffset)
      dstOffset = val;
    else if (name == kPartition)
      partitionIdx = val;
  }

  Value dstPtr = dstPtrs[0];
  if (isPartitioned) {
    assert(dstPtrs.size() > 1 &&
           "Partitioned tensors must have multiple bases");
    auto ptrTy = dstPtrs[0].getType();
    auto vecTy = VectorType::get({static_cast<int64_t>(dstPtrs.size())}, ptrTy);
    Value basesVec = b.undef(vecTy);
    for (size_t i = 0; i < dstPtrs.size(); ++i)
      basesVec = b.insert_element(basesVec, dstPtrs[i], b.i32_val(i));
    dstPtr = b.extract_element(basesVec, partitionIdx);
  }

  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(dstOffset, iVal), pVal);
    dstOffset = b.add(dstOffset, padOffset);
  }
  dstPtr = b.gep(sharedPtrTy, elementType, dstPtr, dstOffset);

  // Update tensor shapes based on offset
  for (size_t i = 0; i < numDims; ++i)
    tensorShape[i] = b.smax(b.i32_val(0), b.sub(tensorShape[i], offset[i]));

  bool adjustedBlockShape = false;
  if (isStore && padInterval > 0 && padAmount > 0) {
    adjustedBlockShape = true;
    Value originalTileDim0 = decodedBlockShape[numDims - 1];
    decodedBlockShape[numDims - 1] =
        b.add(originalTileDim0, b.i32_val(padAmount));
    Value cmp = b.icmp_ult(tensorShape[numDims - 1], originalTileDim0);
    tensorShape[numDims - 1] =
        b.select(cmp, tensorShape[numDims - 1], originalTileDim0);
  }

  // Update group0 with addresses
  Value globalAddrNew = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0 = vecSet(b, group0, 0, pred);
  group0 = vecSet(b, group0, 1, ldsAddr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddrNew));
  Value typeBit = b.and_(vecGet(b, group0, 3), b.i32_val(1 << 31));
  group0 = vecSet(b, group0, 3,
                  b.or_(typeBit,
                        b.trunc(i32_ty, b.lshr(globalAddrNew, b.i64_val(32)))));

  // Update group1 with tensor shapes
  Value g1_0 = vecGet(b, group1, 0);
  if (multicastMask)
    g1_0 = b.or_(g1_0, multicastMask);
  group1 = vecSet(b, group1, 0, g1_0);

  group1 = vecSet(b, group1, 1, b.shl(tensorShape[numDims - 1], b.i32_val(16)));
  group1 = vecSet(b, group1, 2, b.lshr(tensorShape[numDims - 1], b.i32_val(16)));

  if (numDims >= 2) {
    group1 = vecSet(b, group1, 2,
                    b.or_(vecGet(b, group1, 2),
                          b.shl(tensorShape[numDims - 2], b.i32_val(16))));
    Value g1_3_masked = b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF << 16));
    group1 = vecSet(b, group1, 3,
                    b.or_(g1_3_masked,
                          b.lshr(tensorShape[numDims - 2], b.i32_val(16))));
  }

  // Configure barrier
  if (barrierPtr) {
    group1 = vecSet(b, group1, 0,
                    b.or_(vecGet(b, group1, 0),
                          b.shl(b.i32_val(1), b.i32_val(18))));
    group1 = vecSet(b, group1, 1,
                    b.or_(vecGet(b, group1, 1),
                          b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr),
                                        b.i32_val(3)),
                                 b.i32_val(0x00FFFF))));
  } else {
    group1 = vecSet(b, group1, 0,
                    b.and_(vecGet(b, group1, 0), b.i32_val(0xFFFBFFFF)));
  }

  if (adjustedBlockShape) {
    Value g1_3 = b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF));
    g1_3 = b.or_(g1_3, b.shl(decodedBlockShape[numDims - 1], b.i32_val(16)));
    group1 = vecSet(b, group1, 3, g1_3);
  }

  // Update group2/group3 for higher dimensions
  if (numDims >= 3) {
    Value &g2 = group2.value().get();
    g2 = vecSet(b, g2, 0, tensorShape[numDims - 3]);
  }

  if (numDims >= 4) {
    Value &g2 = group2.value().get();
    g2 = vecSet(b, g2, 1, tensorShape[numDims - 4]);
  }

  if (numDims == 5) {
    Value &g3 = group3.value().get();
    Value g3_1 = b.and_(vecGet(b, g3, 1), b.i32_val(0xFFFF));
    g3_1 = b.or_(g3_1, b.shl(tensorShape[0], b.i32_val(16)));
    g3 = vecSet(b, g3, 1, g3_1);
    Value g3_2 = b.and_(vecGet(b, g3, 2), b.i32_val(0xFFFF << 16));
    g3_2 = b.or_(g3_2, b.lshr(tensorShape[0], b.i32_val(16)));
    g3 = vecSet(b, g3, 2, g3_2);
  }
}

// Fill TDM descriptor for gather/scatter operations (2D only).
void fillTDMDescriptorForGatherScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, unsigned padInterval, unsigned padAmount,
    Value &group0, Value &group1, Value &group2, Value &group3,
    Value ldsRowOffset, Value globalColOffset, Value ldsPtr, Value pred,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices) {
  assert(!rowIndices.empty() && "Gather/scatter requires row indices.");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode descriptor to get tensor info
  auto [globalPtr, tensorShape, tensorStride, decodedBlockShape] =
      decodeTDMDescriptorFull(rewriter, loc, group0, group1, group2, group3,
                              /*numDims=*/2);

  // Apply CTA column offset to the base pointer.
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
  Value cgaColOffset = b.mul(cgaOffsets[1].second, tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, cgaColOffset);

  Value colOffset = b.mul(globalColOffset, tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, colOffset);

  Value ldsOffset = b.mul(ldsRowOffset, b.i32_val(blockShape[1]));

  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(ldsOffset, iVal), pVal);
    ldsOffset = b.add(ldsOffset, padOffset);
  }
  ldsPtr = b.gep(sharedPtrTy, elementType, ldsPtr, ldsOffset);

  tensorShape[1] = b.smax(b.i32_val(0), b.sub(tensorShape[1], globalColOffset));

  // Update group0 with addresses and enable gather/scatter mode
  Value globalAddrNew = b.ptrtoint(i64_ty, globalPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, ldsPtr);

  Value predWithGatherScatter = b.or_(pred, b.i32_val(1 << 31));
  if (use32BitIndices)
    predWithGatherScatter = b.or_(predWithGatherScatter, b.i32_val(1 << 30));

  group0 = vecSet(b, group0, 0, predWithGatherScatter);
  group0 = vecSet(b, group0, 1, ldsAddr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddrNew));

  // group0[3]: preserve type bits, set global_addr upper 25 bits
  Value globalAddrHigh = b.trunc(i32_ty, b.lshr(globalAddrNew, b.i64_val(32)));
  globalAddrHigh = b.and_(globalAddrHigh, b.i32_val(0x01FFFFFF));
  Value typeBits = b.and_(vecGet(b, group0, 3), b.i32_val(0xC0000000));
  group0 = vecSet(b, group0, 3, b.or_(typeBits, globalAddrHigh));

  // Update group1 with adjusted tensor shapes
  group1 = vecSet(b, group1, 1, b.shl(tensorShape[1], b.i32_val(16)));
  Value g1_2 = b.lshr(tensorShape[1], b.i32_val(16));
  g1_2 = b.or_(g1_2, b.shl(tensorShape[0], b.i32_val(16)));
  group1 = vecSet(b, group1, 2, g1_2);
  Value g1_3 = b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF << 16));
  g1_3 = b.or_(g1_3, b.lshr(tensorShape[0], b.i32_val(16)));
  group1 = vecSet(b, group1, 3, g1_3);

  // Configure barrier
  if (barrierPtr) {
    group1 = vecSet(b, group1, 0,
                    b.or_(vecGet(b, group1, 0),
                          b.shl(b.i32_val(1), b.i32_val(18))));
    group1 = vecSet(b, group1, 1,
                    b.or_(vecGet(b, group1, 1),
                          b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr),
                                        b.i32_val(3)),
                                 b.i32_val(0x00FFFF))));
  } else {
    group1 = vecSet(b, group1, 0,
                    b.and_(vecGet(b, group1, 0), b.i32_val(0xFFFBFFFF)));
  }

  // Set tile_dim1 (number of valid indices)
  size_t numIndices = rowIndices.size();
  Value g1_4 = b.and_(vecGet(b, group1, 4), b.i32_val(0xFFFF0000));
  g1_4 = b.or_(g1_4, b.i32_val(numIndices & 0xFFFF));
  group1 = vecSet(b, group1, 4, g1_4);

  // Fill group2 and group3 with row indices
  if (use32BitIndices) {
    for (size_t i = 0; i < 4 && i < numIndices; ++i)
      group2 = vecSet(b, group2, i, rowIndices[i]);
    for (size_t i = 4; i < 8 && i < numIndices; ++i)
      group3 = vecSet(b, group3, i - 4, rowIndices[i]);
  } else {
    auto packIndices = [&](Value &group, size_t baseIdx) {
      for (size_t i = 0; i < 4; ++i) {
        Value dword = b.i32_val(0);
        size_t idx0 = baseIdx + i * 2;
        size_t idx1 = baseIdx + i * 2 + 1;
        if (idx0 < numIndices) {
          Value idx0_i32 = b.zext(i32_ty, rowIndices[idx0]);
          dword = b.and_(idx0_i32, b.i32_val(0xFFFF));
        }
        if (idx1 < numIndices) {
          Value idx1_i32 = b.zext(i32_ty, rowIndices[idx1]);
          dword = b.or_(
              dword, b.shl(b.and_(idx1_i32, b.i32_val(0xFFFF)), b.i32_val(16)));
        }
        group = vecSet(b, group, i, dword);
      }
    };
    packIndices(group2, 0);
    packIndices(group3, 8);
  }
}

// Incrementally advance a TDM descriptor's global_addr and tensor_dim.
void advanceTDMDescriptor(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          Value &group0, Value &group1,
                          ArrayRef<Value> offsets, size_t numDims,
                          Type elementType, bool isRowMajor,
                          bool updateBounds) {
  assert(numDims <= 2 && "advanceTDMDescriptor only supports 1D-2D tensors");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value v16 = b.i32_val(16);

  // For col-major tensors, swap offsets to match hardware dimension order
  SmallVector<Value> hwOffsets(offsets.begin(), offsets.end());
  if (!isRowMajor && numDims >= 2)
    swapTrailingDims(hwOffsets);

  // Decode strides: innermost stride is implicit (=1)
  SmallVector<Value> stride(numDims);
  stride[numDims - 1] = b.i32_val(1);
  if (numDims >= 2)
    stride[numDims - 2] = vecGet(b, group1, 5); // tensor_dim0_stride

  // Compute element delta = sum(hwOffsets[i] * stride[i])
  Value elementDelta = b.i32_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimDelta = b.mul(hwOffsets[i], stride[i]);
    elementDelta = b.add(elementDelta, dimDelta);
  }

  // Advance global_addr in group0[2:3] by adding the byte delta directly.
  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  Value byteDelta =
      b.mul(b.sext(i64_ty, elementDelta), b.i64_val(elementBitWidth / 8));
  // Pack group0[2:3] as i64 via vector bitcast — produces a single
  // s_add_nc_u64.
  auto v2i32Ty = VectorType::get(2, rewriter.getI32Type());
  Value addrVec = LLVM::UndefOp::create(rewriter, loc, v2i32Ty);
  addrVec = b.insert_element(addrVec, vecGet(b, group0, 2), b.i32_val(0));
  addrVec = b.insert_element(addrVec, vecGet(b, group0, 3), b.i32_val(1));
  Value addr = b.bitcast(addrVec, i64_ty);
  Value newAddr = b.add(addr, byteDelta);
  Value newAddrVec = b.bitcast(newAddr, v2i32Ty);
  group0 = vecSet(b, group0, 2, b.extract_element(newAddrVec, b.i32_val(0)));
  group0 = vecSet(b, group0, 3, b.extract_element(newAddrVec, b.i32_val(1)));

  if (!updateBounds)
    return;

  // Update tensor_dim for innermost dim (dim0 in hw = numDims-1 in code)
  {
    Value off = hwOffsets[numDims - 1];
    Value dimLo = b.lshr(vecGet(b, group1, 1), v16);
    Value dimHi = b.shl(b.and_(vecGet(b, group1, 2), b.i32_val(0xFFFF)), v16);
    Value dim = b.or_(dimLo, dimHi);
    Value newDim = b.smax(b.i32_val(0), b.sub(dim, off));
    // Preserve barrier_addr in group1[1] bits [15:0]
    group1 = vecSet(b, group1, 1,
                    b.or_(b.and_(vecGet(b, group1, 1), b.i32_val(0xFFFF)),
                          b.shl(newDim, v16)));
    // Preserve tensor_dim1_low in group1[2] bits [31:16]
    group1 = vecSet(b, group1, 2,
                    b.or_(b.and_(vecGet(b, group1, 2), b.i32_val(0xFFFF0000)),
                          b.and_(b.lshr(newDim, v16), b.i32_val(0xFFFF))));
  }

  // Update tensor_dim for second dim (dim1 in hw = numDims-2 in code)
  if (numDims >= 2) {
    Value off = hwOffsets[numDims - 2];
    Value dimLo = b.lshr(vecGet(b, group1, 2), v16);
    Value dimHi = b.shl(b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF)), v16);
    Value dim = b.or_(dimLo, dimHi);
    Value newDim = b.smax(b.i32_val(0), b.sub(dim, off));
    group1 = vecSet(b, group1, 2,
                    b.or_(b.and_(vecGet(b, group1, 2), b.i32_val(0xFFFF)),
                          b.shl(newDim, v16)));
    group1 = vecSet(b, group1, 3,
                    b.or_(b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF0000)),
                          b.and_(b.lshr(newDim, v16), b.i32_val(0xFFFF))));
  }
}

// Emit a TDM load or store operation for regular (non-scatter) transfers.
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> shapePerCTA,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, ArrayRef<Value> dstPtrs,
                      Value pred, Value multicastMask, Type elementType,
                      Value barrierPtr, bool isLoad,
                      const triton::LinearLayout &sharedLayout, Value ctaId,
                      bool isRowMajor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  assert(shapePerCTA.size() <= 5);

  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
  Value group4Zero = b.null(v8i32Ty);

  // desc[0] = group0 (<4 x i32>), desc[1] = group1 (<8 x i32>)
  // For 3D+: desc[2] = group2 (<4 x i32>), desc[3] = group3 (<4 x i32>)
  if (shapePerCTA.size() > 2) {
    Value group0 = desc[0];
    Value group1 = desc[1];
    Value group2 = desc[2];
    Value group3 = desc[3];

    fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                      to_vector(shapePerCTA), numWarps, padInterval, padAmount,
                      group0, group1, std::ref(group2), std::ref(group3),
                      to_vector(offset), dstPtrs, pred, multicastMask,
                      barrierPtr, sharedLayout, ctaId, !isLoad, isRowMajor);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                       : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2, group3, group4Zero, b.i32_val(0)});
  } else {
    Value group0 = desc[0];
    Value group1 = desc[1];

    fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                      to_vector(shapePerCTA), numWarps, padInterval, padAmount,
                      group0, group1, std::nullopt, std::nullopt,
                      to_vector(offset), dstPtrs, pred, multicastMask,
                      barrierPtr, sharedLayout, ctaId, !isLoad, isRowMajor);

    auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
    Value group2Zero = b.null(v4i32Ty);
    Value group3Zero = b.null(v4i32Ty);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                       : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2Zero, group3Zero, group4Zero, b.i32_val(0)});
  }
}

// Emit a lightweight TDM load from a pre-positioned descriptor.
void emitTDMLoadFromAdvanced(RewriterBase &rewriter, Location loc,
                             const LLVMTypeConverter *typeConverter,
                             ArrayRef<Value> desc,
                             ArrayRef<int64_t> shapePerCTA, int numWarps,
                             unsigned padInterval, unsigned padAmount,
                             ArrayRef<Value> dstPtrs, Value pred,
                             Value multicastMask, Type elementType,
                             Value barrierPtr,
                             const triton::LinearLayout &sharedLayout,
                             Value ctaId, bool isRowMajor) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  size_t numDims = shapePerCTA.size();
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // desc[0] = group0 (<4 x i32>), desc[1] = group1 (<8 x i32>)
  // group0 is already a <4 x i32> — we only patch pred and lds_addr,
  // leaving global_addr (elements 2,3) untouched from the advance.
  Value group0 = desc[0];
  Value group1 = desc[1];

  // Handle col-major
  auto hwShapePerCTA = to_vector(shapePerCTA);
  std::optional<triton::LinearLayout> adjustedSharedLayout;
  if (!isRowMajor) {
    swapTrailingDims(hwShapePerCTA);
    if (numDims >= 2) {
      auto dimN_2 = StringAttr::get(ctx, "dim" + std::to_string(numDims - 2));
      auto dimN_1 = StringAttr::get(ctx, "dim" + std::to_string(numDims - 1));
      adjustedSharedLayout = swapOutDimSemantics(sharedLayout, dimN_2, dimN_1);
    }
  }
  const auto &tdmViewSharedLayout =
      adjustedSharedLayout ? *adjustedSharedLayout : sharedLayout;

  auto kMessage = StringAttr::get(ctx, "message");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kPartition = StringAttr::get(ctx, "partition");

  auto cgaLayout = triton::gpu::SharedLinearEncodingAttr::get(
                       ctx, tdmViewSharedLayout, /*layoutAlignment=*/16)
                       .getCGALayout()
                       .getLinearLayout();

  auto warpsPerCTA = getWarpDistribution(hwShapePerCTA, numWarps);
  auto tdmLayout =
      triton::gpu::getTDMLinearLayout(hwShapePerCTA, warpsPerCTA, cgaLayout);

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  auto tdmToShared = tdmLayout.invertAndCompose(tdmViewSharedLayout);
  auto sharedOffsets = applyLinearLayout(
      loc, rewriter, tdmToShared,
      {{kMessage, b.i32_val(0)}, {kWarp, warpId}, {kBlock, ctaId}});

  Value dstOffset = b.i32_val(0);
  Value partitionIdx = b.i32_val(0);
  bool isPartitioned = tdmToShared.hasOutDim(kPartition);
  for (auto &[name, val] : sharedOffsets) {
    if (name == kOffset)
      dstOffset = val;
    else if (name == kPartition)
      partitionIdx = val;
  }

  Value dstPtr = dstPtrs[0];
  if (isPartitioned) {
    assert(dstPtrs.size() > 1);
    auto ptrTy = dstPtrs[0].getType();
    auto vecTy =
        VectorType::get({static_cast<int64_t>(dstPtrs.size())}, ptrTy);
    Value basesVec = b.undef(vecTy);
    for (size_t i = 0; i < dstPtrs.size(); ++i)
      basesVec = b.insert_element(basesVec, dstPtrs[i], b.i32_val(i));
    dstPtr = b.extract_element(basesVec, partitionIdx);
  }

  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(dstOffset, iVal), pVal);
    dstOffset = b.add(dstOffset, padOffset);
  }
  dstPtr = b.gep(sharedPtrTy, elementType, dstPtr, dstOffset);

  // Patch only pred (element 0) and lds_addr (element 1) in group0.
  // Elements 2,3 (global_addr) are already correct from the advance.
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0 = vecSet(b, group0, 0, pred);
  group0 = vecSet(b, group0, 1, ldsAddr);

  // Set multicast mask
  if (multicastMask)
    group1 = vecSet(b, group1, 0,
                    b.or_(vecGet(b, group1, 0), multicastMask));

  // Set barrier
  if (barrierPtr) {
    group1 = vecSet(b, group1, 0,
                    b.or_(vecGet(b, group1, 0), b.i32_val(1 << 18)));
    Value barrierBits =
        b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
               b.i32_val(0x00FFFF));
    group1 = vecSet(b, group1, 1,
                    b.or_(b.and_(vecGet(b, group1, 1), b.i32_val(0xFFFF0000)),
                          barrierBits));
  } else {
    group1 = vecSet(b, group1, 0,
                    b.and_(vecGet(b, group1, 0), b.i32_val(0xFFFBFFFF)));
  }

  // Issue intrinsic
  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
  Value group4Zero = b.null(v8i32Ty);

  if (numDims > 2) {
    Value group2 = desc[2];
    Value group3 = desc[3];
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.tensor.load.to.lds", {},
        {group0, group1, group2, group3, group4Zero, b.i32_val(0)});
  } else {
    auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
    Value group2Zero = b.null(v4i32Ty);
    Value group3Zero = b.null(v4i32Ty);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.tensor.load.to.lds", {},
        {group0, group1, group2Zero, group3Zero, group4Zero, b.i32_val(0)});
  }
}

// Emit a TDM gather or scatter operation for non-contiguous row access.
size_t getTDMGatherScatterInstrinsicCount(size_t numIndices,
                                          bool use32BitIndices) {
  if (numIndices == 0)
    return 0;
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;
  return llvm::divideCeil(numIndices, maxIndicesPerInstr);
}

void emitTDMGatherScatter(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                          unsigned padInterval, unsigned padAmount,
                          Value ldsPtr, Value pred, Type elementType,
                          Value barrierPtr,
                          const triton::LinearLayout &cgaLayout, Value ctaId,
                          ArrayRef<Value> rowIndices, Value colOffset,
                          bool use32BitIndices, bool isGather) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  assert(!rowIndices.empty() && "Gather/scatter requires row indices");
  assert(colOffset && "Gather/scatter requires column offset");

  size_t numIndices = rowIndices.size();
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;
  size_t numInstructions =
      getTDMGatherScatterInstrinsicCount(numIndices, use32BitIndices);

  // desc[0] = group0 (<4 x i32>), desc[1] = group1 (<8 x i32>)
  Value group0Base = desc[0];
  Value group1Base = desc[1];

  auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
  Value group2Init = b.null(v4i32Ty);
  Value group3Init = b.null(v4i32Ty);

  for (size_t instrIdx = 0; instrIdx < numInstructions; ++instrIdx) {
    size_t startIdx = instrIdx * maxIndicesPerInstr;
    size_t endIdx = std::min(startIdx + maxIndicesPerInstr, numIndices);

    SmallVector<Value> batchIndices(rowIndices.begin() + startIdx,
                                    rowIndices.begin() + endIdx);

    Value g0 = group0Base;
    Value g1 = group1Base;
    Value g2 = group2Init;
    Value g3 = group3Init;

    fillTDMDescriptorForGatherScatter(
        rewriter, loc, typeConverter, elementType, to_vector(blockShape),
        padInterval, padAmount, g0, g1, g2, g3, b.i32_val(startIdx), colOffset,
        ldsPtr, pred, barrierPtr, cgaLayout, ctaId, batchIndices,
        use32BitIndices);

    auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
    Value group4Zero = b.null(v8i32Ty);

    const char *intrinsicName = isGather ? "llvm.amdgcn.tensor.load.to.lds"
                                         : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, {},
                                    {g0, g1, g2, g3, group4Zero, b.i32_val(0)});
  }
}

SmallVector<Value> emitTDMPrefetch(RewriterBase &rewriter, Location loc,
                                   ArrayRef<Value> desc,
                                   ArrayRef<int64_t> blockShape, int numLanes,
                                   int numWarps, int numCTAs,
                                   ArrayRef<Value> offset, Value pred,
                                   Type elementType, Value laneId, Value warpId,
                                   Value ctaId, bool isSpeculative) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  int numDims = blockShape.size();
  Type globalPtrTy = ptr_ty(loc.getContext(), 1);

  // desc[0] = group0 (<4 x i32>), desc[1] = group1 (<8 x i32>)
  Value group0 = desc[0];
  Value group1 = desc[1];
  std::optional<Value> group2, group3;
  if (numDims > 2) {
    group2 = desc[2];
    group3 = desc[3];
  }

  auto [basePtr, tensorShape, tensorStride, decodedBlockShape] =
      mlir::LLVM::AMD::decodeTDMDescriptorFull(rewriter, loc, group0, group1,
                                                group2, group3, numDims);

  auto dot64 = [&](ArrayRef<Value> indices, ArrayRef<Value> strides) {
    Value ret = b.i64_val(0);
    for (auto [index, stride] : llvm::zip(indices, strides))
      ret = b.add(ret, b.mul(b.zext(i64_ty, index), b.zext(i64_ty, stride)));
    return ret;
  };

  Value tileOffset = dot64(offset, tensorStride);
  auto tilePtr = b.gep(globalPtrTy, elementType, basePtr, tileOffset);

  Value linearTensorSize =
      b.mul(b.zext(i64_ty, tensorShape[0]), b.zext(i64_ty, tensorStride[0]));
  Value maxOffsetFromTile = b.sub(linearTensorSize, tileOffset);

  const int bytesPerPrefetch = 256;
  int elemPerPrefetch =
      (bytesPerPrefetch * 8) / elementType.getIntOrFloatBitWidth();

  SmallVector<int64_t> scaledBlockShape(blockShape.begin(), blockShape.end());
  scaledBlockShape.back() =
      ceil<int64_t>(scaledBlockShape.back(), elemPerPrefetch);

  auto blockedEnc = triton::gpu::getDefaultBlockedEncoding(
      loc.getContext(), scaledBlockShape, numWarps, numLanes, numCTAs);
  auto ll = triton::gpu::toLinearLayout(scaledBlockShape, blockedEnc);

  auto kRegister = rewriter.getStringAttr("register");
  auto kLane = rewriter.getStringAttr("lane");
  auto kWarp = rewriter.getStringAttr("warp");
  auto kBlock = rewriter.getStringAttr("block");

  auto scaledStride = tensorStride;
  scaledStride.back() = b.i32_val(elemPerPrefetch);

  auto baseIndices = applyLinearLayout(loc, rewriter, ll,
                                       {{kRegister, b.i32_val(0)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, ctaId}});

  constexpr int cacheScope = 8;
  const int hintValue = cacheScope | static_cast<int>(isSpeculative);
  IntegerAttr hint = rewriter.getI32IntegerAttr(hintValue);

  SmallVector<Value> offsets(ll.getInDimSize(kRegister));
  for (int reg = 0; reg < ll.getInDimSize(kRegister); reg++) {
    auto regIndices =
        ll.apply({{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

    SmallVector<Value> indices;
    for (auto [base, regIdx] : llvm::zip(baseIndices, regIndices)) {
      assert(base.first == regIdx.first);
      Value combined = b.xor_(base.second, b.i32_val(regIdx.second));
      indices.emplace_back(combined);
    }

    Value localOffset = dot64(indices, scaledStride);
    Value prefetchPtr = b.gep(globalPtrTy, elementType, tilePtr, localOffset);

    Value inBounds = b.icmp_slt(localOffset, maxOffsetFromTile);
    Value combinedPred = isSpeculative ? pred : b.and_(pred, inBounds);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterPrefetch =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *prefetchBlock = rewriter.createBlock(afterPrefetch);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, combinedPred, prefetchBlock,
                           afterPrefetch);

    rewriter.setInsertionPointToStart(prefetchBlock);

    ROCDL::GlobalPrefetchOp::create(rewriter, loc, prefetchPtr, hint, {}, {},
                                    {});

    rewriter.setInsertionPointToEnd(prefetchBlock);
    LLVM::BrOp::create(rewriter, loc, afterPrefetch);
    rewriter.setInsertionPointToStart(afterPrefetch);

    offsets[reg] =
        b.select(combinedPred, b.add(localOffset, tileOffset), b.i64_val(0));
  }
  return offsets;
}
} // namespace mlir::LLVM::AMD
