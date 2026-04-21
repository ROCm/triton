#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_TDMSTORESPIPELINE_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_TDMSTORESPIPELINE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {

// Convert tt.descriptor_store / tt.descriptor_scatter ops inside `forOp`
// into the asynchronous AMD TDM form, lifting the LDS allocation outside
// the loop and hoisting `amdg.async_tdm_wait num=0` to BEFORE the
// `ttg.local_store`. This single-buffers the LDS allocation across
// iterations and overlaps the outgoing TDM write with the next
// iteration's compute.
//
// This mirrors NVIDIA's `pipelineTMAStores` and runs after the AMD load
// pipeliner has finished expanding loads, but before
// `tritonamdgpu-convert-tensor-ops` lowers any remaining
// (non-pipelinable) descriptor stores/scatters.
//
// Returns true if any store/scatter in the loop was rewritten.
bool pipelineTDMStores(scf::ForOp forOp);

} // namespace mlir

#endif
