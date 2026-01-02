#include "triton/Dialect/TritonGPU/Transforms/LdsPrefetchUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

FailureOr<SmallVector<Value>> findLocalLoadForDotOperand(Value v) {
  Operation *op = v.getDefiningOp();
  bool foundLocalLoad = false;
  SmallVector<Value> rets;
  rets.push_back(op->getResult(0));
  LDBG("Looking for local_load starting at: " << *op);
  while (op) {
    if (!op->getResult(0).hasOneUse())
      return failure();
    if (auto ll = dyn_cast<LocalLoadOp>(op)) {
      if (isa<DotOperandEncodingAttr>(ll.getType().getEncoding())) {
        rets.push_back(op->getOperand(0));
        foundLocalLoad = true;
        break;
      }
      return failure();
    } else {
      if (op->getNumOperands() != 1)
        return failure();
      rets.push_back(op->getOperand(0));
    }
    op = op->getOperand(0).getDefiningOp();
    if (op)
      LDBG("op between dot and local_load: " << *op);
  }
  std::reverse(rets.begin(), rets.end());

  if (foundLocalLoad)
    return rets;
  return failure();
}

void clonePrefetchElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                 OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

Value getIncomingLoopArg(scf::ForOp forOp, Value v) {
  if (auto arg = mlir::dyn_cast<BlockArgument>(v))
    if (arg.getOwner()->getParentOp() == forOp.getOperation())
      return forOp.getTiedLoopInit(arg)->get();
  return Value();
}

Value getYieldOperand(scf::ForOp forOp, scf::YieldOp yieldOp, Value v) {
  auto arg = mlir::cast<BlockArgument>(v);
  unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
  return yieldOp.getOperand(yieldIdx);
}

} // namespace mlir::triton::gpu
