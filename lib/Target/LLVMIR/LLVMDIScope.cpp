#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

//===----------------------------------------------------------------------===//
// This file implements a pass to add debug info scope to LLVM operations, and
// is inspired by the DIScopeForLLVMFuncOpPass in LLVM/MLIR. Different from the
// DIScopeForLLVMFuncOpPass, this pass also handles inlined functions.
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_LLVMDISCOPE
#include "triton/Target/LLVMIR/Passes.h.inc"

namespace {

/// Attempt to extract a filename for the given loc.
FileLineColLoc extractFileLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    return extractFileLoc(fusedLoc.getLocations().front());
  // Prefer the innermost callee for callsite locations.
  if (auto csLoc = dyn_cast<CallSiteLoc>(loc))
    return extractFileLoc(csLoc.getCallee());
  StringAttr unknownFile = mlir::StringAttr::get(loc.getContext(), "<unknown>");
  return mlir::FileLineColLoc::get(unknownFile, 0, 0);
}

auto calcBitWidth(mlir::Type type) -> std::optional<unsigned> {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  } else if (mlir::isa<mlir::VectorType>(type)) {
    auto vectorType = dyn_cast<mlir::VectorType>(type);
    llvm::ArrayRef<int64_t> shape = vectorType.getShape();
    mlir::Type elementType = vectorType.getElementType();
    llvm::ArrayRef<bool> scalableDims = vectorType.getScalableDims();
    unsigned size = 1;
    for (auto i : shape) {
      size *= i;
    }

    if (auto elementTypeSize = calcBitWidth(elementType);
        elementTypeSize.has_value()) {
      return size * elementTypeSize.value();
    }
  }

  return std::nullopt;
}

// Note: mlir does not provided any built-in conversion from mlir::Type to
// mlir::LLVM::DITypeAttr
LLVM::DITypeAttr convertType(MLIRContext *context, mlir::Type type) {
  if (type.isInteger(1)) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "bool"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_boolean);
  }
  if (type.isInteger()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "int"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_signed);
  } else if (type.isF16()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "half"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (type.isF32()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "float"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (type.isF64()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "double"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (mlir::isa<mlir::VectorType>(type)) {
    if (auto vectorTypeSize = calcBitWidth(type); vectorTypeSize.has_value()) {
      return LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type,
          mlir::StringAttr::get(context, "vector"), vectorTypeSize.value(),
          llvm::dwarf::DW_ATE_float);
    } else {
      // TODO: falling back to unknown_type, perhaps theres a better way to
      // handle when element type size is not determined
    }
  }
  return LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type,
      mlir::StringAttr::get(context, "unknown_type"), 0,
      llvm::dwarf::DW_ATE_signed);
}

LLVM::DITypeAttr convertPtrType(MLIRContext *context, mlir::Type pointerType,
                                mlir::Type pointeeType, unsigned sizeInBits) {
  // LLVMPointerType does not include pointee info, need to pass from external
  // source
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(pointerType)) {
    unsigned addrSpace = ptrType.getAddressSpace();

    LLVM::DITypeAttr diElTypeAttr = convertType(context, pointeeType);
    LLVM::DITypeAttr diTypeAttr = mlir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_pointer_type,
        mlir::StringAttr::get(context, ""), diElTypeAttr, sizeInBits,
        /*alignInBits=*/0, /*offset=*/0,
        /*optional<address space>=*/addrSpace, /*extra data=*/nullptr);
    return diTypeAttr;
  }
  // Return unknown_type if fail to construct DIDerivedTypeAttr with
  // WD_TAG_pointer_type.
  return LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type,
      mlir::StringAttr::get(context, "unknown_type"), 0,
      llvm::dwarf::DW_ATE_signed);
}

} // anonymous namespace

/// Add a debug info scope to LLVMFuncOp that are missing it.
struct LLVMDIScopePass : public impl::LLVMDIScopeBase<LLVMDIScopePass> {
  void setSubprogramAttr(LLVM::LLVMFuncOp funcOp) {
    Location loc = funcOp.getLoc();
    if (loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    MLIRContext *context = &getContext();

    // To find a DICompileUnitAttr attached to a parent (the module for
    // example), otherwise create a default one.
    LLVM::DICompileUnitAttr compileUnitAttr;
    if (ModuleOp module = funcOp->getParentOfType<ModuleOp>()) {
      auto fusedCompileUnitAttr =
          module->getLoc()
              ->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
      if (fusedCompileUnitAttr)
        compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    }

    // Filename, line and colmun to associate to the function.
    LLVM::DIFileAttr fileAttr;
    int64_t line = 1, col = 1;
    FileLineColLoc fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
      col = fileLoc.getColumn();
      StringRef inputFilePath = fileLoc.getFilename().getValue();
      fileAttr = LLVM::DIFileAttr::get(
          context, llvm::sys::path::filename(inputFilePath),
          llvm::sys::path::parent_path(inputFilePath));
    }

    // Figure out debug information (`subprogramFlags` and `compileUnitAttr`) to
    // attach to the function definition / declaration. External functions are
    // declarations only, and are defined in a different compile unit, so mark
    // them appropriately in `subprogramFlags`, and set an empty
    // `compileUnitAttr`.
    DistinctAttr recId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            recId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true,
            triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES")
                ? LLVM::DIEmissionKind::Full
                : LLVM::DIEmissionKind::
                      LineTablesOnly); // DIEmissionKind::Full is required by
                                       // emiting ptx with dbg-metadata
                                       // (otherwise assertion fail)
      }
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Definition;
    } else {
      compileUnitAttr = {};
    }

    // TODO: support nested types
    llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
    for (auto resTy : funcOp.getResultTypes()) {
      LLVM::DITypeAttr tyAttr = convertType(context, resTy);
      types.push_back(tyAttr);
    }
    // If no return type then add a null type as a place holder for that.
    if (types.empty())
      types.push_back(mlir::LLVM::DINullTypeAttr::get(context));
    for (auto [idx, inTy] : llvm::enumerate(funcOp.getArgumentTypes())) {
      if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(inTy)) {
        auto pointeeTy =
            funcOp.getArgAttrOfType<TypeAttr>(idx, "tt.pointee_type");
        auto ptrRange =
            funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
        if (pointeeTy && ptrRange) {
          LLVM::DITypeAttr tyAttr = convertPtrType(
              context, ptrTy, pointeeTy.getValue(), ptrRange.getInt());
          types.push_back(tyAttr);
        }

      } else {
        // Here assume remained inTy are only scalar types
        LLVM::DITypeAttr tyAttr = convertType(context, inTy);
        types.push_back(tyAttr);
      }
    }

    auto subroutineTypeAttr = LLVM::DISubroutineTypeAttr::get(
        context, llvm::dwarf::DW_CC_normal, types);

    StringAttr funcNameAttr = funcOp.getNameAttr();
    // Note that scopeline is set differently from LLVM's
    // DIScopeForLLVMFuncOpPass. I don't find reasons why scopeline should be
    // the column offset
    auto id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/true, id, compileUnitAttr, fileAttr,
        funcNameAttr, funcNameAttr, fileAttr,
        /*line=*/line, /*scopeline=*/line, subprogramFlags, subroutineTypeAttr,
        /*retainNodes=*/{}, /*annotations=*/{});

    llvm::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
    // Handle function arguments and add them to retainedNodes:
    // 1. Create DebugValueOp for each arg
    // 2. Add each arg as DILocalVariableAttr to retainedNodes
    for (auto [idx, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
      LLVM::DITypeAttr argTypeAttr;
      BlockArgument arg = funcOp.getArgument(idx);
      if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(argType)) {
        auto pointeeTy =
            funcOp.getArgAttrOfType<TypeAttr>(idx, "tt.pointee_type");
        auto ptrRange =
            funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
        if (!pointeeTy || !ptrRange) {
          continue;
        }
        argTypeAttr = convertPtrType(context, ptrTy, pointeeTy.getValue(),
                                     ptrRange.getInt());
      } else {
        argTypeAttr = convertType(context, argType);
      }

      Location argLoc = arg.getLoc();
      auto nameLoc = dyn_cast<NameLoc>(argLoc);
      if (!nameLoc)
        continue;
      Location childLoc = nameLoc.getChildLoc();
      StringAttr nameAttr = nameLoc.getName();

      auto localScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(subprogramAttr);
      auto diFlag = LLVM::DIFlags::Zero;
      auto argVarAttr = LLVM::DILocalVariableAttr::get(
          context, localScopeAttr, nameAttr, fileAttr, line, idx + 1, 0,
          argTypeAttr, diFlag);

      auto exprAttr = LLVM::DIExpressionAttr::get(context);
      OpBuilder b(context);
      b.setInsertionPointToStart(&funcOp.getBody().front());
      Operation *dbgOp =
          LLVM::DbgValueOp::create(b, childLoc, arg, argVarAttr, exprAttr);

      retainedNodes.push_back(argVarAttr);
    }

    id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    subprogramAttr = LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/false, id, compileUnitAttr, fileAttr,
        funcNameAttr, funcNameAttr, fileAttr,
        line, line, subprogramFlags, subroutineTypeAttr,
        retainedNodes, /*annotations=*/{});

    funcOp->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }

  void setLexicalBlockFileAttr(Operation *op) {
    Location opLoc = op->getLoc();
    if (!isa<CallSiteLoc>(opLoc))
      return;

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto funcOpLoc = mlir::cast<FusedLoc>(funcOp.getLoc());
    auto scopeAttr =
        mlir::cast<LLVM::DISubprogramAttr>(funcOpLoc.getMetadata());

    MLIRContext *ctx = op->getContext();
    std::function<Location(Location)> makeScoped =
        [&](Location loc) -> Location {
      if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
        Location newCallee = makeScoped(cs.getCallee());
        Location newCaller = makeScoped(cs.getCaller());
        return CallSiteLoc::get(newCallee, newCaller);
      }

      // Build a DIFile for this leaf location
      FileLineColLoc fileLine = extractFileLoc(loc);
      StringRef inputFilePath = fileLine.getFilename().getValue();
      LLVM::DIFileAttr fileAttr =
          LLVM::DIFileAttr::get(ctx, llvm::sys::path::filename(inputFilePath),
                                llvm::sys::path::parent_path(inputFilePath));

      auto lexicalBlock =
          LLVM::DILexicalBlockFileAttr::get(ctx, scopeAttr, fileAttr,
                                            /*discriminator=*/0);
      return FusedLoc::get(ctx, {loc}, lexicalBlock);
    };

    op->setLoc(makeScoped(opLoc));
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op))
        setSubprogramAttr(cast<LLVM::LLVMFuncOp>(op));
      else
        setLexicalBlockFileAttr(op);
    });
  }
};

} // namespace mlir
