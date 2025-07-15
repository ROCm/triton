#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#undef LLVM_DEBUG
#define LLVM_DEBUG(X) X

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-scheduler-machine-model"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

/*******************************************************************************
 * Simple machine model which enabled the scheduler passes to create an
 * operation schedule optimized for both performance and resource allocation.
 *
 * Operation properties are expressed in terms of single hardware instruction,
 * e.g. v_add_fp32 taking 4 cycles, whereas Triton ops may be refined to various
 * degrees, such that single ttgir op maps to one or many hardware instruction.
 *
 * For complex ops (ones which don't directly map to single hardware ops),
 * Triton may attempt to break down the op into it's constituent ops,
 * get the properties for all of them combined, then return the properties
 * for the combined complex op.
 * Therefore OpProperties can be added together.
 *
 * Change these to an attribute, and add attribute to each operation
 * Create walkers for each optype like arith::Mulf to handle op types.
 *
 ******************************************************************************/

namespace {

  // Keep track of which operations map to same / different hardware resources.
  // using MachineModelResourcePipeMask = uint32_t;
  // Note: this can be changed to a mask of pipes for complex instructions.
  enum MachineModelResourcePipe : uint32_t {
    None = 0,
    Mfma = 1,
    Lds = 2,
    Global = 3,
    Valu = 4,
    Other = 5,
  };
  constexpr uint32_t numResourcePipes = 6;

  StringRef toString(MachineModelResourcePipe pipe) {
    switch(pipe) {
      case None: return "0";
      case Mfma: return "M";
      case Lds: return "L";
      case Global: return "G";
      case Valu: return "V";
      case Other: return "O";
    }
  };

  // Properties of a simple operation (maps to one resource pipe),
  // which can be queried for op,
  // and guides the Scheduler to create a more performant operation order.
  struct MachineModelOpProperties {

    MachineModelOpProperties(MachineModelResourcePipe pipe,
        int32_t seqBusy, int32_t pipeBusy, StringRef n) :
        resourcePipe(pipe), seqBusyCycles(seqBusy), pipeBusyCycles(pipeBusy), name(n) {}

    // To which resource pipe does this op map.
    MachineModelResourcePipe resourcePipe;
    
    // Op blocks all other ops from issuing.
    // Call this sequencer Busy
    int32_t seqBusyCycles;

    // How long to wait before issuing another op to same
    // resource pipe.
    // Call this pipeBusy
    int32_t pipeBusyCycles;

    // To be used for debugging, e.g. saying that op was
    // assumed to map to 16 v_pk_add_fp32 instructions.
    StringRef name;
  };

  // This needs to be converted into walkers.

  // Abstract base class for querying op properties.
  // Each new generation of hardware inherits from the previous,
  // and only needs to override ops with new properties.
  struct MachineModel {
    // Returns the properties for the op.
    virtual MachineModelOpProperties getOpProperties(Operation *op) = 0;
    int32_t getDataLatency(MachineModelResourcePipe pipe) {
      switch(pipe) {
        case Lds: return 64;
        case Global: return 4096; // TODO(dtanner) -1?
        default: return 0;
      }
    };
    virtual ~MachineModel() = default;
  };

  // MI250
  struct MachineModelGFX90A : MachineModel {
    MachineModelOpProperties getOpProperties(Operation *op) {
      // TODO(dtanner) subviews are nops
      if (llvm::isa<triton::gpu::MemDescSubviewOp,
                    triton::gpu::MemDescTransOp,
                    tt::amdgpu::ExtractSliceOp,
                    ROCDL::SchedBarrier,
                    tt::amdgpu::ConcatOp>(op)) {
        return MachineModelOpProperties(
          MachineModelResourcePipe::Other, 0, 0, "nop");
      }
      // Fallback is simple valu
      return MachineModelOpProperties(
          MachineModelResourcePipe::Valu, 4, 4, "valu");
    }
  };

  // MI300
  struct MachineModelGFX942 : MachineModelGFX90A {

    MachineModelOpProperties getOpProperties(Operation *op) {
      // Mfma
      if (isa<triton::DotOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Mfma, 4, 16, "mfma_16x16x16");

      // LDS Ops
      } else if (isa<triton::gpu::LocalLoadOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 4, 4+16, "ds_read_b128");
      } else if (isa<triton::gpu::LocalStoreOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 40, 40+32, "ds_write_b128");
      } else if (isa<mlir::gpu::BarrierOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 8, 8+0, "s_barrier");

      // Global Memory Ops
      } else if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Global, 8, 64, "global_load");
      }

      return MachineModelGFX90A::getOpProperties(op);
    }
  };

  /*
    MachineModel State
    track how many cycles since last op of type was issued.
    this is already too sophisticated
      scheduling a barrier will put a cooldown on the ldsread pipe and lds write pipe for 64 cycles

    TODO(dtanner) we're a little backwards here.
    The pipeBusyCycles are for after the instruction.
    So, when we're scheduling bottom up, we need to track when the last op was to each pipe
    and then query if we can fit in the queried op.
    E.g. s_barrier followed by ds_write should be totally fine.
    Instead of resourcePipeCoolDown going from N -> ) to say ready (wich would be fine forwards)
    We'll track time stamp which last used.

    TODO(dtanner) I conflated pipeBusyCycles with dataLatency for lds ops.
    When only dealing with ops, that gives def/use ability, but doesn't let us say that
    loads and stores have a data dependency on barriers and end of kernel.
    Need a way to say that when a barrier gets scheduled, all parent nodes
  */

  struct MachineState {
    MachineState(MachineModel *model, bool topDown) : currentCycle(0), machineModel(model), topDown(topDown) {
      for (int32_t i = 0; i < numResourcePipes; ++i) {
        cyclePipeReady.push_back(0);
      }
      reset();
    }

    // Assume we need to wait for ds_writes, ds_reads and buffer_loads at the top of the loop.
    void reset() {
      currentCycle = 0;
      for (int32_t i = 0; i < numResourcePipes; ++i) {
        cyclePipeReady[i] = 0;
      }
      opDataReadyCycle.clear();
    }

    // Elapsed time will be cycles until pipe is ready + seqBusyCycles
    void scheduleOp(Operation *op) {
      LDBG("scheduleOp()");
      // record time before stepping forward
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      int32_t elapsedCycles = scheduleOpCalcElapsedCycles(op);
      scheduleOpUpdateCurrentCycle(elapsedCycles);
      scheduleOpUpdatePipesReady(properties);
      scheduleOpUpdateDepsReady(op);
      LDBG("scheduleOp() - DONE");
    }

    // Issue op and step time forward.
    int32_t scheduleOpCalcElapsedCycles(Operation *op) {
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      MachineModelResourcePipe pipe = properties.resourcePipe;
      // If resource pipe wasn't ready, need to first wait for it to empty before issuing next op.
      int32_t elapsedCycles = getCyclesUntilOpReady(op) + properties.seqBusyCycles;
      LDBG("scheduleOpCalcElapsedCycles=" << elapsedCycles);
      return elapsedCycles;
    }

    // Queries opDataReadyCycle
    int32_t getCyclesUntilOpReady(Operation *op) {
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      int32_t cycles = std::max(getCyclesUntilDataReady(op), getCyclesUntilPipeReadyForOp(properties));
      LDBG("getCyclesUntilOpReady=" << cycles);
      return cycles;
    }

    // Op's pipe is ready at
    // pipeLastUsed + pipeBusyCyclesForOp - currentCycle
    // 32 + 8 - 36 = 
    int32_t getCyclesUntilPipeReadyForOp(MachineModelOpProperties properties) {
      int32_t readyCycle;
      if (topDown) {
        // TODO(dtanner) top down needs to keep parent pipeBusyCycles
        readyCycle = (cyclePipeReady[properties.resourcePipe] - getCurrentCycle()); //  + properties.pipeBusyCycles;
        LDBG("getCyclesUntilPipeReady=" << readyCycle);
      } else {
        // BottomUp
        // How long ago was pipe last used compared to when it is now.
        readyCycle = (cyclePipeReady[properties.resourcePipe] - getCurrentCycle())
            // Only care about pipe cycles > seq cycles.
            + (properties.pipeBusyCycles - properties.seqBusyCycles);
        LDBG("getCyclesUntilPipeReady=" << readyCycle);
      }
      LDBG("getCyclesUntilPipeReady(): pr=" << cyclePipeReady[properties.resourcePipe]
          << ", cc=" << getCurrentCycle()
          << ", pb=" << properties.pipeBusyCycles
          << ", sq=" << properties.seqBusyCycles
          << ", ready=" << readyCycle);
      readyCycle = std::max(0, readyCycle);
      return readyCycle;
    }

    int32_t getCyclesUntilDataReady(Operation *op) {
      auto find = opDataReadyCycle.find(op);
      int32_t cycle = 0;
      if (find != opDataReadyCycle.end()) {
        cycle = std::max(0, find->getSecond() - getCurrentCycle());
      }
      LDBG("getCyclesUntilDataReady=" << cycle);
      return cycle;
    }

    // Update when pipes will be ready.
    void scheduleOpUpdatePipesReady(MachineModelOpProperties properties) {
      LDBG("scheduleOpUpdatePipesReady");
      MachineModelResourcePipe pipe = properties.resourcePipe;
      if (topDown) {
        cyclePipeReady[pipe] = getCurrentCycle() + properties.pipeBusyCycles;
      } else {
        cyclePipeReady[pipe] = getCurrentCycle();
      }
    }

    void scheduleOpUpdateCurrentCycle(int32_t elapsedCycles) {
      currentCycle += elapsedCycles;
    }

    // Updates ready for parents/children.
    // This sets the opDataReadyCycle.
    void scheduleOpUpdateDepsReady(Operation *op) {
      LDBG("scheduleOpUpdateDepsReady()");
      if (topDown) {
        // TopDown - scheduled op, therefore children will all be ready
        // when op is done.
        for (auto result : op->getResults()) {
          for (auto child : result.getUsers()) {
            updateOpDataReady(child, op);
          }
        }
      } else {
        // BottomUp - scheduled op, therefore determine when parents
        // will be ready, they're who define the operands.
        for (auto operand : op->getOperands()) {
          auto parent = operand.getDefiningOp();
          if (parent) {
            updateOpDataReady(parent, op);
          }
        }
      }
    }

    // Updates ready for op (called externally)
    // This sets the opDataReadyCycle.
    // other is parent if topDown
    // other is child if BottomUp
    void updateOpDataReady(Operation *target, Operation *other) {
      assert(target && other);
      target->print(llvm::dbgs());
      other->print(llvm::dbgs());
      int32_t readyCycle = getCurrentCycle();
      if (topDown) {
        readyCycle += calcCyclesUntilDataReady(other);
      } else {
        readyCycle += calcCyclesUntilDataReady(target);
      }
      setDataReadyCycle(target, readyCycle);
    }

    int32_t getCurrentCycle() const {
      return currentCycle;
    }

    // This depends on seeing only 1 op at a time for each pipe.
    // Therefore all lds reads/writes need to be serialized.
    int32_t calcCyclesUntilDataReady(Operation *op) {
      LDBG("calcCyclesUntilDataReady()");
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      MachineModelResourcePipe pipe = properties.resourcePipe;
      LDBG("calcCyclesUntilDataReady() pipe=" << pipe);
      int32_t cyclesUntilDataReady = machineModel->getDataLatency(pipe);
      LDBG("cyclesUntilDataReady=" << cyclesUntilDataReady);
      return cyclesUntilDataReady;
    }

    // Specify when op will be ready based on data latency and pipe.
    // If already exists, updates to max cycles.
    void setDataReadyCycle(Operation *op, int32_t c) {
      LDBG("setDataReadyCycle()");
      auto find = opDataReadyCycle.find(op);
      if (find != opDataReadyCycle.end()) {
        int32_t updated = std::max(c, find->getSecond());
        LDBG("setDataReadyCycle() t=" << c << " (updated), op=" << op->getName());
        opDataReadyCycle[op] = updated;
      } else {
        LDBG("setDataReadyCycle() t=" << c << ", op=" << op->getName());
        opDataReadyCycle[op] = c;
      }
    }

    int32_t currentCycle;
    MachineModel *machineModel;
    /* Tracks pipe readiness differently for TopDown vs BottomUp.
      BottomUp: tracks the cycle during which pipe was last used.
      TopDown: tracks the cycle last used + prev op's pipe busy cycles.
    */
    SmallVector<int32_t, numResourcePipes> cyclePipeReady;
    DenseMap<Operation *, int32_t> opDataReadyCycle;
    bool topDown;
  };

  // Format: [@nodeId opName p={parent nodes} c={child nodes}]
  llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const MachineState &machine) {
    out << "[t=" << machine.getCurrentCycle();
    for (int32_t i = 1; i < numResourcePipes; ++i) {
      out << ", " << toString(static_cast<MachineModelResourcePipe>(i));
      out << "=" << machine.cyclePipeReady[i];
    }
    out << "]";
    if (true) {
      for (auto entry : machine.opDataReadyCycle) {
        out << "\t t=" << entry.getSecond() << " ready << " << entry.getFirst()->getName() << "\n";
      }
    }
    return out;
  }

} // namespace
