#include <memory>
#include "mlir/Support/LogicalResult.h"

// 4D coordinate within loop.
struct DotCoord {
  public:
  DotCoord(int b = -1, int m = -1, int n = -1, int k = -1) :
      b(b), m(m), n(n), k(k) {
  }

  bool operator==(const DotCoord& other) const {
    return getB() == other.getB() &&
           getM() == other.getM() &&
           getN() == other.getN() &&
           getK() == other.getK();
  };

  int getB() const { return b; }
  int getM() const { return m; }
  int getN() const { return n; }
  int getK() const { return k; }

  int b;
  int m;
  int n;
  int k;
};

/*
  Abstract Parent Class
  DotOrdering children must contain
   - Strategy for iterating over b,m,n,k iterations.
   - State for iterator.
  DotOrdering children must specify
   - getFirst() state given to iterator::begin().
   - getLast() state given to iterator::end().
   - next() called by iterator++ to advance state of child.
   - getDotCoord() returns DotCoord from child.
*/
class DotOrdering {
public:

  // Returns whether fma while loop is done.
  bool isDone() {
    return done;
  }

  // Returns current DotCoord.
  virtual DotCoord getDotCoord() const = 0;

  // Advance to next coord; eventually calls setDone().
  virtual void next() = 0;
  
  protected:
  // first dotCoord is always valid, i.e. at least 1 fma.  
  DotOrdering() : done(false) {
    llvm::outs() << "DotOrdering()\n";
  }
    
  // Declare that no more valid fmas/iterations.
  void setDone() {
    llvm::outs() << "setDone()\n";
    done = true;
  }

  private:
  bool done;

}; // DotOrdering


/*
  Default ordering.
*/
class DotOrderingBMNK : public DotOrdering {
public:
  // When constructed here, the state doesn't matter (only within iterator).
  DotOrderingBMNK(int numRepB, int numRepM, int numRepN, int numRepK) :
                  DotOrdering(),
                  numReps(numRepB, numRepM, numRepN, numRepK),
                  iter(0, 0, 0, 0) {
    llvm::outs() << numReps.b << numReps.m << numReps.n << numReps.k << iter.b << iter.m << iter.n << iter.k << "\n";
  }

  /*
    Specify from abstract parent class.
    Copy tiling parameters, and override coord state.
  */
  //std::shared_ptr<DotOrdering> getFirst() const {
  //  std::shared_ptr<DotOrdering> ptr = std::make_shared<DotOrderingBMNK>(
  //      numReps.b, numReps.m, numReps.n, numReps.k,
  //      0, 0, 0, 0);
  //  return ptr;
  //}

  //std::shared_ptr<DotOrdering> getLast() const {
  //  std::shared_ptr<DotOrdering> ptr = std::make_shared<DotOrderingBMNK>(
  //      numReps.b, numReps.m, numReps.n, numReps.k,
  //      numReps.b, 0, 0, 0);
  //  return ptr;
  //}
  // increment the state of the ordering to the next DotCoord.
  // returns true if after incrementing to the next DotCoord produces a valid DotCoord.
  // returns false if DotCoord cannot be called after next();
  void next() {
    // Loop order in B, M, N, K; start with inner-most.
    iter.k++;
    llvm::outs() << "k=" << iter.k << "\n";
    if (iter.k >= numReps.k) {
      iter.k = 0;
      iter.n++;
      llvm::outs() << "n=" << iter.n << "\n";
    }
    if (iter.n >= numReps.n) {
      iter.n = 0;
      iter.m++;
      llvm::outs() << "m=" << iter.m << "\n";
    }
    if (iter.m >= numReps.m) {
      iter.m = 0;
      iter.b++;
      llvm::outs() << "b=" << iter.b << "\n";
    }
    if (iter.b >= numReps.b) {
      // Done iterating.
      setDone();
    }
  }

  DotCoord getDotCoord() const {
    return iter;
  }

  //bool operator==(const DotOrdering& other) const {
  //  const DotOrderingBMNK* derivedOther = dynamic_cast<const DotOrderingBMNK*>(&other);
  //  if (!derivedOther) {
  //      return false;
  //  }

  //  bool equals = getDotCoord() == derivedOther->getDotCoord();
  //  llvm::outs() << "DotOrderingBMNK==" << (equals ? "True" : "False") << "\n";
  //  return equals;
  //}

  //bool DotOrdering::operator==(const iterator& other) const {
    
  //}


  private:
  DotCoord numReps;
  DotCoord iter;
}; // DotOrderingBMNK


/*
  DotOrderingTiled is used for controlling the order in which FMAs are
  emitted to minimize lifetimes of A, B operands in registers.
  Doing so helps backend compilers minimize register pressure
  and hide latency.
  Well-order of FMAs have two benefits.

  (1) Whether M or N is the outer vs inner loop.
  When the MxN shape of the dot is not square, it is prefferable to register
  pressure to have the larger size be the outer loop and the shorter side be
  the inner loop. Doing so reduces the peak register pressure for A,B operands
  during the lifetime of the dot. The larger/smaller comparison is in terms of bits.

  (2) Tiling.
  Rather than simply ordering the FMAs as row-major or col-major according
  to M, N loops, tiling gives a benefit to prefetching and register allocation
  at the beginning and end of the dot. Starting the dot with a squarish
  tile (along M, N dims) Means fewer local_loads will supply more mfmas.
  The corollary of this is that, at the end of the dot, more
  operands/registers are being freed while there are still more FMAs;
  this allows a smoother transition from one dot to another in terms
  of register pressure.

  Args:
   - numRepM,N,K - total numReps for M, N or K.
   - tileSizeM,N,K - how many reps belong to a tile.
   - outerTileN - Since M and N form an outer product, it is numerically correct
       to have either M or N be the outer or inner loop.
       Whether the outer tiling should be N is calculated outside of
       this class and based on wether it requires fewer registers to
       keep all of A[M] or B[N] alive (and not re-fetch data from LDS);
       based on whether having M or N be the outer loop would require
       fewer registers (based on aType, bType, M and N).
       Some dots will have all of A or B already loaded into registers
       outside of the loops, e.g. FA, and therefore that operand
       should be the inner tile.
  
  The below examples how a dot MxNxK needs a different number of live
  operands/registers based on the order while still wanting to
  prefetch the data from LDS to hide it's latency

  == Example 0 ==
  No tiling (m, n, k ordering)
  numRep = {4, 8, 2}
  tileSize = {4, 8, 2} // effectively

                        N
    +----+----+----+----+----+----+----+----+
    |  0 |  2 |  4 |  6 |  8 | 10 | 12 | 14 |
    +----+----+----+----+----+----+----+----+
    | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
  M +----+----+----+----+----+----+----+----+
    | 32 | 34 | 36 | 38 | 40 | 42 | 44 | 46 |
    +----+----+----+----+----+----+----+----+
    | 48 | 50 | 52 | 54 | 56 | 58 | 60 | 62 |
    +----+----+----+----+----+----+----+----+  K[0]

                        N
    +----+----+----+----+----+----+----+----+
    |  1 |  3 |  5 |  7 |  9 | 11 | 13 | 15 |
    +----+----+----+----+----+----+----+----+
    | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 |
  M +----+----+----+----+----+----+----+----+
    | 33 | 35 | 37 | 39 | 41 | 43 | 45 | 47 |
    +----+----+----+----+----+----+----+----+
    | 49 | 51 | 53 | 55 | 57 | 59 | 61 | 63 |
    +----+----+----+----+----+----+----+----+  K[1]

    Loads needed for first 8 FMAs: 10
    Peak live opds (to prefetch by 8 FMAs): 19

  
  == Example 1 == 2x2x1 tiling
  numRep = {4, 8, 2}
  tileSize = {2, 2, 1}
  outerTileN = False
                        N
    +----+----+----+----+----+----+----+----+
    |  0 |  1 |  8 |  9 | 16 | 17 | 24 | 25 |
    +----+----+----+----+----+----+----+----+
    |  2 |  3 | 10 | 11 | 18 | 19 | 26 | 27 |
  M +----+----+----+----+----+----+----+----+
    |  4 |  5 | 12 | 13 | 20 | 21 | 28 | 29 |
    +----+----+----+----+----+----+----+----+
    |  6 |  7 | 14 | 15 | 22 | 23 | 30 | 31 |
    +----+----+----+----+----+----+----+----+  K[0]

                        N
    +----+----+----+----+----+----+----+----+
    | 32 | 33 | 40 | 41 | 48 | 49 | 56 | 57 |
    +----+----+----+----+----+----+----+----+
    | 34 | 35 | 42 | 43 | 50 | 51 | 58 | 59 |
  M +----+----+----+----+----+----+----+----+
    | 36 | 37 | 44 | 45 | 52 | 53 | 60 | 61 |
    +----+----+----+----+----+----+----+----+
    | 38 | 39 | 46 | 47 | 54 | 55 | 62 | 63 |
    +----+----+----+----+----+----+----+----+  K[1]

    Loads needed for first 8 FMAs: 6
    Peak live opds (to prefetch by 8 FMAs): 9
*/





/*
  DotOrderingTiled is both the strategy
  and state for the iterator.
*/
class DotOrderingTiled : public DotOrdering {
  public:
  // numRep* must be evenly divisible by tileSize*

  explicit DotOrderingTiled(
    int64_t numRepB,
    int64_t numRepM,
    int64_t numRepN,
    int64_t numRepK,
    int64_t tileSizeB,
    int64_t tileSizeM,
    int64_t tileSizeN,
    int64_t tileSizeK,
    bool outerTileN)
      : DotOrdering(),
        numRepB(numRepB),
        numRepM(numRepM),
        numRepN(numRepN),
        numRepK(numRepK),
        tileSizeB(std::min(tileSizeB, numRepB)),
        tileSizeM(std::min(tileSizeM, numRepM)),
        tileSizeN(std::min(tileSizeN, numRepN)),
        tileSizeK(std::min(tileSizeK, numRepK)),
        numTilesB(numRepB / tileSizeB),
        numTilesM(numRepM / tileSizeM),
        numTilesN(numRepN / tileSizeN),
        numTilesK(numRepK / tileSizeK),
        outerTileN(outerTileN),
        tiledCoord(getTiledLoopIndices(), getLoopNumIter())
    {
    // Num mfmas must evenly divide into tiles.
    if (numTilesB * tileSizeB != numRepB) {
      llvm::errs() << "ERROR: DotOrderingTiled not valid with numRepB=" << numRepB << ", and tileSizeB=" << tileSizeB << "\n";
    }
    if (numTilesM * tileSizeM != numRepM) {
      llvm::errs() << "ERROR: DotOrderingTiled not valid with numRepM=" << numRepM << ", and tileSizeM=" << tileSizeM << "\n";
    }
    if (numTilesN * tileSizeN != numRepN) {
      llvm::errs() << "ERROR: DotOrderingTiled not valid with numRepN=" << numRepN << ", and tileSizeN=" << tileSizeN << "\n";
    }
    if (numTilesK * tileSizeK != numRepK) {
      llvm::errs() << "ERROR: DotOrderingTiled not valid with numRepK=" << numRepK << ", and tileSizeK=" << tileSizeK << "\n";
    }
    llvm::outs() << "DotOrderingTiled: " << numRepB << numRepM << numRepN << numRepK << ", "
        << tileSizeB << tileSizeM << tileSizeN << tileSizeK << ", "
        << numTilesB << numTilesM << numTilesN << numTilesK << "\n";
  }

  // Stores the order over which the 8D tiling space will be iterated.
  struct TiledLoopIndices {
    // Tile indices.
    int bT;
    int kT;
    int mT;
    int nT;
    // Element indices.
    int b;
    int m;
    int n;
    int k;

    bool operator==(const TiledLoopIndices& other) const {
      return bT == other.bT
          && kT == other.kT
          && mT == other.mT
          && nT == other.nT
          && b == other.b
          && m == other.m
          && n == other.n
          && k == other.k;
    };
  };

  // Custom ordering of 8D tiled loops.
  TiledLoopIndices getTiledLoopIndices() const {
    TiledLoopIndices loopIdx;
    // Outer-most loop.
    loopIdx.bT = 7;
    loopIdx.kT = 6;
    loopIdx.mT = outerTileN ? 4 : 5;
    loopIdx.nT = outerTileN ? 5 : 4;
    loopIdx.b = 3;
    // Have inner loop do opposite as outer loops to eventually "snake" and preserve operand.
    loopIdx.m = outerTileN ? 2 : 1;
    loopIdx.n = outerTileN ? 1 : 2;
    loopIdx.k = 0;
    // Inner-most loop.
    return loopIdx;
  }

  // returns num iterations for each loop.
  std::array<int, 8> getLoopNumIter() const {
    TiledLoopIndices loopIdx = getTiledLoopIndices();
    std::array<int, 8> max_indices;
    max_indices[loopIdx.b] = tileSizeB;
    max_indices[loopIdx.m] = tileSizeM;
    max_indices[loopIdx.n] = tileSizeN;
    max_indices[loopIdx.k] = tileSizeK;
    // NumTiles.
    max_indices[loopIdx.bT] = numTilesB;
    max_indices[loopIdx.mT] = numTilesM;
    max_indices[loopIdx.nT] = numTilesN;
    max_indices[loopIdx.kT] = numTilesK;
    return max_indices;
  }
  
  /*
    8D coordinate within tile space.
    Also implements next() and getDotCoord() for iterating.
  */
  struct TiledCoord {

    const TiledLoopIndices loopIdx;
    // Max iterations of each loop, i.e. tile size, and num tiles.
    const std::array<int, 8> max_indices;
    // Index of each loop.
    std::array<int, 8> indices;
    // done==true when iterated past last index.
    bool currentIsValid;

    TiledCoord(const TiledLoopIndices& loopIndices, const std::array<int, 8>& maxIndices) :
        loopIdx(loopIndices),
        max_indices(maxIndices),
        indices{{0,0,0,0,0,0,0,0}}, currentIsValid(true) {

          llvm::outs() << "max_indices: ";
          for (int i = 0; i < 8; i++) llvm::outs() << max_indices[i] << ", ";
          llvm::outs() << "\n";
        }

    // Once each loop level reaches max, reset it and move to next loop level.
    void next(int idx) {
      if (idx >= indices.size()) {
        // Done iterating.
        currentIsValid = false;
        return;
      }
      indices[idx]++;
      if (indices[idx] >= max_indices[idx]) {
        indices[idx] = 0;
        next(idx+1);
      }
    }

    // Start by incrementing the 0th index, i.e. inner-most loop.
    bool next() {
      if (!currentIsValid) return currentIsValid;
      next(0);
      llvm::outs() << "next: " << getB() << ", " << getM() << ", " << getN() << ", " << getK() << "\n";
      llvm::outs() << "next: ";
      for (int i = 0; i < 8; i++) {
        llvm::outs() << indices[i] << ", ";
      }
      llvm::outs() << "\n";
      return currentIsValid;
    }

    // Each value combines (element idx within tile) + (tile idx)*(tile size).
    int getB() const { return indices[loopIdx.b] + indices[loopIdx.bT] * max_indices[loopIdx.b]; }
    int getM() const { return indices[loopIdx.m] + indices[loopIdx.mT] * max_indices[loopIdx.m]; }
    int getN() const { return indices[loopIdx.n] + indices[loopIdx.nT] * max_indices[loopIdx.n]; }
    int getK() const { return indices[loopIdx.k] + indices[loopIdx.kT] * max_indices[loopIdx.k]; }

    // Convert TileCoord(8D) to DotCoord(4D)
    DotCoord getDotCoord() const {
      llvm::outs() << "getDotCoord: " << getB() << ", " << getM() << ", " << getN() << ", " << getK() << "\n";
      return DotCoord(getB(), getM(), getN(), getK());
    }

  }; // TiledCoord

  
  void next() {
    if (!tiledCoord.next()) {
      setDone();
    }
  }

  DotCoord getDotCoord() const {
    return tiledCoord.getDotCoord();
  }

  private:
  // Tiling strategy.
  const int64_t numRepB;
  const int64_t numRepM;
  const int64_t numRepN;
  const int64_t numRepK;
  const int64_t tileSizeB;
  const int64_t tileSizeM;
  const int64_t tileSizeN;
  const int64_t tileSizeK;
  const int64_t numTilesB;
  const int64_t numTilesM;
  const int64_t numTilesN;
  const int64_t numTilesK;
  const bool outerTileN;

  // State for iterator.
  TiledCoord tiledCoord;

}; // DotOrderingTiled
