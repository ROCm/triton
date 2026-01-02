#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_LDSPREFETCHUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_LDSPREFETCHUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

namespace mlir::triton::gpu {

// Walk backwards from a dot operand through single-operand, single-use
// elementwise ops to find a LocalLoadOp with DotOperandEncodingAttr.
// Returns the chain of values from the LocalLoadOp's memdesc source
// through to the dot operand (in def-use order, i.e. memdesc first).
FailureOr<SmallVector<Value>> findLocalLoadForDotOperand(Value v);

// Clone elementwise ops between the LocalLoadOp and the dot operand,
// remapping to a new prefetched slice value. `vals` is the chain returned
// by findLocalLoadForDotOperand; `ret` is updated in-place to point to the
// final cloned value.
void clonePrefetchElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                 OpBuilder &builder);

// Given a value that is a block argument of forOp's body,
// return the corresponding init arg passed into the loop.
// Returns a null Value if v is not a block argument of forOp.
Value getIncomingLoopArg(scf::ForOp forOp, Value v);

// Given a value that is a block argument of forOp's body,
// return the corresponding yield operand.
Value getYieldOperand(scf::ForOp forOp, scf::YieldOp yieldOp, Value v);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_LDSPREFETCHUTILS_H_
