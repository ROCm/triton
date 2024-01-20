#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::MfmaEncodingAttr;
using ttg::SliceEncodingAttr;

enum class MatrixCoreVersion {
  CDNA_MFMA1,
  CDNA_MFMA2,
  CDNA_MFMA3,
  RDNA_WMMA,
  UNKNOWN
};

MatrixCoreVersion getMatrixCoreVersion(StringRef archGen) {
  if (archGen.contains("gfx11"))
    return MatrixCoreVersion::RDNA_WMMA;
  if (archGen.contains("gfx908"))
    return MatrixCoreVersion::CDNA_MFMA1;
  if (archGen.contains("gfx90a"))
    return MatrixCoreVersion::CDNA_MFMA2;
  if (archGen.contains("gfx940") ||
      archGen.contains("gfx941") ||
      archGen.contains("gfx942"))
    return MatrixCoreVersion::CDNA_MFMA3;
  return MatrixCoreVersion::UNKNOWN;
}

int getMfmaVersion(MatrixCoreVersion matrixCoreVer) {
  if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer)
    return 1;
  if (MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer)
    return 2;
  if (MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer)
    return 3;
  return 0;
}

SmallVector<unsigned, 2>
warpsPerTile(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
             SmallVector<int64_t, 2> shapePerWarp) {
  // TODO: needs to be updated with appropriate shapePerWarp etc.
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  mlir::ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  mlir::BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp)) {
      if (shape[0] >= shape[1]) {
        return {(unsigned)numWarps, 1};
      } else {
        return {1, (unsigned)numWarps};
      }
    }

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 2> ret = {1, 1};

  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0])
        ret[0] *= 2;
      else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp[1] > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileMFMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
                 SmallVector<int64_t, 2> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}

SmallVector<unsigned, 2>
warpsPerTileWMMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  return warpsPerTile(dotOp, shape, numWarps, {16, 16});
}

class BlockedToMFMA : public mlir::RewritePattern {
  int mfmaVersion;
  int enforcedNonKDim;

public:
  BlockedToMFMA(mlir::MLIRContext *context, int mfmaVersion, int nonKDim)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        mfmaVersion(mfmaVersion), enforcedNonKDim(nonKDim) {}

  bool isChainDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    mlir::ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    mlir::SetVector<mlir::Operation*> fwdSlices;
    mlir::getForwardSlice(static_cast<mlir::Operation*>(dotOp), &fwdSlices, fwdOpt);
    for (Operation *op : fwdSlices) {
      // ensure output of the first dot is the operand 0 of the second dot
      if (isa<tt::DotOp>(op) && (op != dotOp)) {
        auto dOp = dyn_cast<tt::DotOp>(op);
        auto oper0 = dOp.getOperand(0).getDefiningOp();
        if(std::find(fwdSlices.begin(), fwdSlices.end(), oper0) != fwdSlices.end()) {
          return true;
        }
      }
    }

    mlir::BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    mlir::SetVector<mlir::Operation*> bwdSlices;
    // search backward of the operand 0 of the dot 
    auto oper0 = dotOp.getOperand(0).getDefiningOp();
    mlir::getBackwardSlice(dyn_cast<mlir::Operation*>(oper0), &bwdSlices, bwdOpt);
    int i = 0;
    for (Operation *op : bwdSlices) {
      llvm::outs() << "<<<<<bwd_op-" << i++ << " = " << *op << "\n";
      if (isa<tt::DotOp>(op) && (op != dotOp)) {
        llvm::outs() << "bwd_found, return\n";
        return true;
      }
    }

    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @return pair {nonKDim, kDim} sizes of one MFMA instruction arguments
  std::pair<unsigned, unsigned> chooseMfmaDimensions(tt::DotOp dot) const {
    // number of matrix elements along k dim per one MFMA intruction
    unsigned kDim = 0;
    auto opType = dot.getA().getType().cast<RankedTensorType>();

    auto dataTypeA = opType.getElementType();
    auto dataTypeB =
        dot.getB().getType().cast<RankedTensorType>().getElementType();

    auto resType = dot.getD().getType().cast<RankedTensorType>();
    auto resShape = resType.getShape();

    unsigned nonKDim = 0;
    if (enforcedNonKDim != 0) {
      nonKDim = enforcedNonKDim;
    } else {
      nonKDim = 0;
      int minSize = std::min(resShape[0], resShape[1]);
      if (minSize >= 32)
        nonKDim = 32;
      if (minSize >= 16 && minSize < 32)
        nonKDim = 16;
      if (minSize < 16)
        nonKDim = 4;
      assert(nonKDim != 0);
    }

    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(nonKDim, dataTypeA, dataTypeB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");
    else
      kDim = (*maybeMfmaInsn).getKDim();
    assert(kDim != 0);
    assert(nonKDim != 0);
    assert(resShape[0] % nonKDim == 0 && resShape[1] % nonKDim == 0);
    assert(opType.getShape()[1] % kDim == 0);
    return {nonKDim, kDim};
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    if (!supportMFMA(dotOp))
      return failure();

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());
    assert(CTALayout.getCTAsPerCGA().size() == 2);
    assert(CTALayout.getCTAsPerCGA()[0] == 1);
    assert(CTALayout.getCTAsPerCGA()[1] == 1);

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    ttg::MfmaEncodingAttr mfmaEnc;

    auto [nonKDim, kDim] = chooseMfmaDimensions(dotOp);

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, retShape, numWarps, {nonKDim, nonKDim});

    bool isTransposed = isChainDot(dotOp);
    mfmaEnc = ttg::MfmaEncodingAttr::get(
        oldRetType.getContext(),
        /*versionMajor*/ mfmaVersion, /*versionMinor*/ 0, warpsPerTile,
        /*instrShape*/ nonKDim, nonKDim, isTransposed);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mfmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);
    auto oldAOrder = oldAType.getEncoding()
                         .cast<ttg::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<ttg::BlockedEncodingAttr>()
                         .getOrder();
    auto oldBOrder = oldBType.getEncoding()
                         .cast<ttg::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<ttg::BlockedEncodingAttr>()
                         .getOrder();

    // kWidth is a number of consecutive elements per one instruction per one
    // thread
    auto kWidth = kDim;
    // in mfma 32x32 case argument matrix groups elements in 2 groups
    // in mfma 16x16 case argument matrix groups elements in 4 groups
    // in mfma 4x4 case arguemnt matrix groups in 16 groups
    switch (nonKDim) {
    case 32:
      kWidth /= 2;
      break;
    case 16:
      kWidth /= 4;
      break;
    case 4:
      kWidth /= 16;
      break;
    default:
      llvm::report_fatal_error("unsupported kDim in mfma dot");
    }
    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getAllowTF32(),
					     dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};

class BlockedToWMMA : public mlir::RewritePattern {
public:
  BlockedToWMMA(mlir::MLIRContext *context)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
  auto dotOp = cast<tt::DotOp>(op);

  auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    if (!supportWMMA(dotOp))
      return failure();

    // get WMMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    ttg::WmmaEncodingAttr wmmaEnc;

    int nonKDim = 16;
    int kDim = 16;

    auto warpsPerTile = warpsPerTileWMMA(dotOp, retShape, numWarps);
    wmmaEnc = ttg::WmmaEncodingAttr::get(oldRetType.getContext(), warpsPerTile);
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);

    // kWidth is a number of consecutive elements per one instruction per one
    // thread
    auto kWidth = kDim / 2;

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, wmmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, wmmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getAllowTF32(),
					     dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonAMDGPUAccelerateMatmulPass
    : public TritonAMDGPUAccelerateMatmulBase<
          TritonAMDGPUAccelerateMatmulPass> {
public:
  TritonAMDGPUAccelerateMatmulPass() = default;
  TritonAMDGPUAccelerateMatmulPass(StringRef archGen,
                                   int matrixInstructionSize,
                                   bool enableWmmaTransform) {
    this->archGenerationName = archGen.data();
    this->matrixInstructionSize = matrixInstructionSize;
    this->enableWmmaTransform = enableWmmaTransform;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    auto matrixCoreVer = getMatrixCoreVersion(archGenerationName);
    if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer) {
      patterns.add<::BlockedToMFMA>(context, getMfmaVersion(matrixCoreVer),
                                    matrixInstructionSize);
    } else if (MatrixCoreVersion::RDNA_WMMA == matrixCoreVer &&
               enableWmmaTransform) {
      patterns.add<::BlockedToWMMA>(context);
    }
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUAccelerateMatmulPass(std::string archGen,
                                             int matrixInstructionSize,
                                             bool enableWmmaTransform) {
  return std::make_unique<TritonAMDGPUAccelerateMatmulPass>(
      archGen, matrixInstructionSize, enableWmmaTransform);
}
