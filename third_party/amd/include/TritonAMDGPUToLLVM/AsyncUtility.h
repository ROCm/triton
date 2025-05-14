#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

#include "mlir/IR/Value.h"

namespace mlir::triton::AMD {
// Traverses the def-chain including control flow of the token and returns true
// if all defining operations are an AsyncWait
bool comesFromAsyncWait(mlir::Value value);
} // namespace mlir::triton::AMD

#endif
