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

//#undef LLVM_DEBUG
//#define LLVM_DEBUG(X) X

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-scheduler-machine-model"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

/*******************************************************************************
  Relatively simple machine model which enabled the scheduler passes to create a
  "not bad" op schedule optimized for both performance and resource allocation.
  To achieve this, the scheduler needs to know things like how many independent DotOps
  should there be between a LocalLoadOp and it's dependent DotOp.
  It is assumed that DotOps are the most important,
  that memory ops whcih feed them are in a close second,
  and all other ops are tertiary in importance.
  Therefore it essentially only model ops as being dot, load/stores, nop, and other.

  To achieve this there is
  (1) MachineModel which stores basic properties of the most important ops,
  such as how many cycles does it take to execute, and which can co-execute.
  (2) MachineState tracker which stores, during scheduling, the hypothetical
  state of the machine to know which ops are best scheduled next.
  For example, when a dot is scheduled BottomUp, store at what time can the
  dependent LocalLoadOp be scheduled.


  However, to give even an approximate interleaving of memory op with 
  Information needed to schedule ops:
  - How far


  


 ******************************************************************************/

namespace {

  // Ops which use different resource pipes can be co-executed.
  //
  enum MachineModelResourcePipe : uint32_t {
    None = 0,
    Mfma = 1,
    Lds = 2,
    Global = 3,
    Other = 4,
  };
  constexpr uint32_t numResourcePipes = 5;

  StringRef toString(MachineModelResourcePipe pipe) {
    switch(pipe) {
      case None: return "0";
      case Mfma: return "M";
      case Lds: return "L";
      case Global: return "G";
      case Other: return "O";
    }
  };

  /*
    MachineModel Op Properties
    Basic properties of the hardware instruction which the ttg ops
    best map to.
    For example, mfma_16x16x16 on MI300X
    - Takes 8 cycles to issue.
    - Also keeps the mfma pipe busy for an additional 8 cycles
    Therefore 4 back-to-back mfmas will take 4*16=64 cycles since
    they're each taking up 16 cycles of the mfma pipe.
    And 1 mfma, 1 lds op, 1 mfma, 1 other op only takes 4*16=32 cycles
    since the mfmas took 16 cycles but the other ops mapped to different
    hardware pipes and could be co-executed for free.

    This complexity of modeling co-execution of ops is necessary
    for determining, e.g., at the top of a loop, how many LoadOps
    can be grouped with how many DotOps, since the LoadOps need
    to increment their addresses and then can co-execute with
    the DotOps.
  */
  struct MachineModelOpProperties {

    MachineModelOpProperties(MachineModelResourcePipe pipe,
        int32_t seqBusy, StringRef n, int32_t pipeBusyAfterSeq = 0) :
        resourcePipe(pipe), cyclesSeqBusy(seqBusy), cyclesPipeBusyAfterSeq(pipeBusyAfterSeq), name(n) {}

    // To which resource pipe does this op map.
    MachineModelResourcePipe resourcePipe;
    
    // Op blocks all other ops from issuing.
    int32_t cyclesSeqBusy;

    /*
      How long to wait before issuing another op to same
      resource pipe even after sequencer issues.
      E.g. mfma "takes" 16 cycles, but this is broken into 4 cycles that the
      sequencer is busy issuing the mfma, and 12 more cycles that the
      mfma pipe is still busy but other pipes can be used.
    */
    int32_t cyclesPipeBusyAfterSeq;

    // To be used for debugging, e.g. saying that op was
    // assumed to map to 16 v_pk_add_fp32 instructions.
    StringRef name;
  };

  /*
    Abstract base class for querying op properties based on GPU generation.
    Each new generation of hardware inherits from the previous,
    therefore each new geneation only needs to override ops with new properties.
    Most ops we don't yet care about, so most ops will fall back to whatever
    the most common instruction is, e.g., 4 cycles of valu.
  */
  struct MachineModel {

    virtual MachineModelOpProperties getOpProperties(Operation *op) = 0;

    /*
      Models how many cycles does it take between issuing a memory
      op and when the data is ready.
      This is currently modeled based on pipe and not on op since
      it is assumed, e.g., that ds_read_b32 has same latency
      as ds_read_b64.
      It is also assumed that read and writes to a pipe are the same.
    */
    virtual int32_t getDataLatency(MachineModelResourcePipe pipe) {
      switch(pipe) {
        case Lds: return 64;
        case Global: return 1000000;
        default: return 0;
      }
    };

    virtual ~MachineModel() = default;
  };

  // MI250
  struct MachineModelGFX90A : MachineModel {
    MachineModelOpProperties getOpProperties(Operation *op) {
      if (llvm::isa<triton::gpu::MemDescSubviewOp,
                    triton::gpu::MemDescTransOp,
                    tt::amdgpu::ExtractSliceOp,
                    ROCDL::SchedBarrier,
                    tt::amdgpu::ConcatOp>(op)) {
        return MachineModelOpProperties(
          MachineModelResourcePipe::None, 0, "nop");
      }
      // Fallback is 4 cycles.
      return MachineModelOpProperties(
          MachineModelResourcePipe::Other, 4, "valu");
    }
  };

  // MI300
  // TODO(dtanner) these all need to reflect how many asm instructions
  // are in the op and how large the tensors are (_b64 vs _b128)
  struct MachineModelGFX942 : MachineModelGFX90A {

    MachineModelOpProperties getOpProperties(Operation *op) {
      // When specifying that memory ops should be spaced "2 mfmas apart"
      // Since the second is co-scheduled with a mfma, don't include the pipe busy time for the 2nd.
      int32_t mfmaCycles1 = 1*16-12;
      int32_t mfmaCycles2 = 2*16-12;
      int32_t mfmaCycles3 = 3*16-12;
      int32_t mfmaCycles4 = 4*16-12;

      // Mfma
      if (isa<triton::DotOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Mfma, 4, "mfma_16x16x16", 12);

      // LDS Ops
      } else if (isa<triton::gpu::LocalLoadOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 4, "ds_read_b128", mfmaCycles1);
      } else if (isa<triton::gpu::LocalStoreOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 40, "ds_write_b128", mfmaCycles2);
      } else if (isa<mlir::gpu::BarrierOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Lds, 8, "s_barrier");

      // Global Memory Ops
      } else if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(op)) {
        return MachineModelOpProperties(
            MachineModelResourcePipe::Global, 4, "buffer_load", mfmaCycles2);
      }

      // Fallback to MI250.
      return MachineModelGFX90A::getOpProperties(op);
    }
  };

  /*
    
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

    // Elapsed time will be cycles until pipe is ready + cyclesSeqBusy
    void scheduleOp(Operation *op) {
      // record time before stepping forward
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      int32_t elapsedCycles = scheduleOpCalcElapsedCycles(op);
      scheduleOpUpdateCurrentCycle(elapsedCycles);
      scheduleOpUpdatePipesReady(properties);
      scheduleOpUpdateDepsReady(op);
    }

    // Issue op and step time forward.
    int32_t scheduleOpCalcElapsedCycles(Operation *op) {
      MachineModelOpProperties properties = machineModel->getOpProperties(op);
      MachineModelResourcePipe pipe = properties.resourcePipe;
      // If resource pipe wasn't ready, need to first wait for it to empty before issuing next op.
      int32_t elapsedCycles = getCyclesUntilOpReady(op) + properties.cyclesSeqBusy;
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
        // TODO(dtanner) top down needs to keep parent cyclesPipeBusyAfterSeq
        readyCycle = (cyclePipeReady[properties.resourcePipe] - getCurrentCycle()); //  + properties.cyclesPipeBusyAfterSeq;
        LDBG("getCyclesUntilPipeReady=" << readyCycle);
      } else {
        // BottomUp
        // How long ago was pipe last used compared to when it is now.
        readyCycle = (cyclePipeReady[properties.resourcePipe] - getCurrentCycle())
            // Only care about pipe cycles > seq cycles.
            + properties.cyclesPipeBusyAfterSeq;
        LDBG("getCyclesUntilPipeReady=" << readyCycle);
      }
      LDBG("getCyclesUntilPipeReady(): pr=" << cyclePipeReady[properties.resourcePipe]
          << ", cc=" << getCurrentCycle()
          << ", pb=" << properties.cyclesPipeBusyAfterSeq
          << ", sq=" << properties.cyclesSeqBusy
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
        cyclePipeReady[pipe] = getCurrentCycle() + properties.cyclesPipeBusyAfterSeq;
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
    if (false) {
      for (auto entry : machine.opDataReadyCycle) {
        out << "\t t=" << entry.getSecond() << " ready << " << entry.getFirst()->getName() << "\n";
      }
    }
    return out;
  }

} // namespace
