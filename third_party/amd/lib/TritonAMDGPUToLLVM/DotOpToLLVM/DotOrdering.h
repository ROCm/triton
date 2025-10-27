
/*
  4D coordinate within loop.
*/
class DotCoord {
  public:
  DotCoord(int b, int m, int n, int k) :
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

  private:
  int b;
  int m;
  int n;
  int k;
};

/*
  Abstract Parent Class
  DotOrdering children will carry state for iterator
  but first/last are determined by DotCoord only.
*/
class DotOrdering {

  // Children will define these.
  // Child class with state reflecting beginning and ending.
  virtual std::unique_ptr<DotOrdering> getFirst() const = 0;
  virtual std::unique_ptr<DotOrdering> getLast() const = 0;
  virtual void next() const = 0;
  virtual DotCoord get() const = 0;

  class iterator {
  private:
    DotOrdering *dotOrdering; // stores state
    // DotCoord dc;

  public:
    iterator(DotOrdering *dO) : dotOrdering(dO) {}

    DotCoord operator*() const {
      return dotOrdering->get();
    }

    iterator& operator++() {
      dotOrdering->next();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // Iterators are same if 4D coord is same.
    bool operator==(const iterator& other) const {
      return dotOrdering->get() == other.dotOrdering->get();
    }

    bool operator!=(const iterator& other) const {
      return !(*this == other);
    }

  }; // iterator

  iterator begin() const {
    return iterator(this->getFirst());
  }

  iterator end() const {
    return iterator(this->getLast());
  }

}; // DotOrdering

/*
  DotTiling is used for controlling the order in which FMAs are
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

struct LoopIndices {
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
};

/*
  DotTiling is both a strategy/ordering
  and state for the iterator.
*/
class DotTiling : DotOrdering {
  public:
  // numRep* must be evenly divisible by tileSize*

  DotTiling (const DotTiling& other) = default;

  explicit DotTiling(
    int64_t numRepB,
    int64_t numRepM,
    int64_t numRepN,
    int64_t numRepK,
    int64_t tileSizeB,
    int64_t tileSizeM,
    int64_t tileSizeN,
    int64_t tileSizeK,
    bool outerTileN)
      : numRepB(numRepB),
        numRepM(numRepM),
        numRepN(numRepN),
        numRepK(numRepK),
        tileSizeB(std::min(tileSizeB, numRepB)),
        tileSizeM(std::min(tileSizeM, numRepM)),
        tileSizeN(std::min(tileSizeN, numRepN)),
        tileSizeK(std::min(tileSizeK, numRepK)),
        outerTileN(outerTileN),
        numTilesB(numRepB / tileSizeB),
        numTilesM(numRepM / tileSizeM),
        numTilesN(numRepN / tileSizeN),
        numTilesK(numRepK / tileSizeK),
        tileSizeOuter(outerTileN ? tileSizeN : tileSizeM),
        tileSizeInner(outerTileN ? tileSizeM : tileSizeN),
        numTilesOuter(outerTileN ? numTilesN : numTilesM),
        numTilesInner(outerTileN ? numTilesM : numTilesN),
        dotTilingCoord(this, 0) {
    // Num mfmas must evenly divide into tiles.
    if (numTilesB * tileSizeB != numRepB) {
      llvm::errs() << "ERROR: DotTiling not valid with numRepB=" << numRepB << ", and tileSizeB=" << tileSizeB << "\n";
    }
    if (numTilesM * tileSizeM != numRepM) {
      llvm::errs() << "ERROR: DotTiling not valid with numRepM=" << numRepM << ", and tileSizeM=" << tileSizeM << "\n";
    }
    if (numTilesN * tileSizeN != numRepN) {
      llvm::errs() << "ERROR: DotTiling not valid with numRepN=" << numRepN << ", and tileSizeN=" << tileSizeN << "\n";
    }
    if (numTilesK * tileSizeK != numRepK) {
      llvm::errs() << "ERROR: DotTiling not valid with numRepK=" << numRepK << ", and tileSizeK=" << tileSizeK << "\n";
    }
  }
  
  int64_t getTileSizeB() const { return tileSizeB; }
  int64_t getTileSizeM() const { return tileSizeM; }
  int64_t getTileSizeN() const { return tileSizeN; }
  int64_t getTileSizeK() const { return tileSizeK; }
  int64_t getTileSizeO() const { return tileSizeOuter; }
  int64_t getTileSizeI() const { return tileSizeInner; }

  int64_t getNumTilesB() const { return numTilesB; }
  int64_t getNumTilesM() const { return numTilesM; }
  int64_t getNumTilesN() const { return numTilesN; }
  int64_t getNumTilesK() const { return numTilesK; }
  int64_t getNumTilesO() const { return numTilesOuter; }
  int64_t getNumTilesI() const { return numTilesInner; }

  int64_t getTileStartM(int tileIdxOuter, int tileIdxInner) const {
    if (outerTileN) {
      return tileIdxInner * tileSizeInner; // M is inner tile loop.
    } else {
      return tileIdxOuter * tileSizeOuter; // M is outer tile loop.
    }
  }
  int64_t getTileStartN(int tileIdxOuter, int tileIdxInner) const {
    if (outerTileN) {
      return tileIdxOuter * tileSizeOuter;
    } else {
      return tileIdxInner * tileSizeInner;
    }
  }
  int64_t getTileStartK(int tileIdxK) const {
    return tileIdxK * tileSizeK;
  }

  LoopIndices getLoopIndices() const {
    LoopIndices loopIdx;
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

  std::array<int, 8> getLoopNumIter() const {
    LoopIndices loopIdx = getLoopIndices();
    std::array<int, 8> max_indices;
    max_indices[loopIdx.b] = getTileSizeB();
    max_indices[loopIdx.m] = getTileSizeM();
    max_indices[loopIdx.n] = getTileSizeN();
    max_indices[loopIdx.k] = getTileSizeK();
    // NumTiles.
    max_indices[loopIdx.bT] = getNumTilesB();
    max_indices[loopIdx.mT] = getNumTilesM();
    max_indices[loopIdx.nT] = getNumTilesN();
    max_indices[loopIdx.kT] = getNumTilesK();
    return max_indices;
  }

  /*
    8D coordinate within tile space.
  */
  struct DotTilingCoord {

    const LoopIndices loopIdx;
    // Max iterations of each loop, i.e. tile size, and num tiles.
    const std::array<int, 8> max_indices;

    std::array<int, 8> indices;

    DotTilingCoord(const DotTiling *dotTiling, size_t i = 0) :
        loopIdx(dotTiling->getLoopIndices()),
        max_indices(dotTiling->getLoopNumIter()),
        indices{{0,0,0,0,0,0,0,0}} {

    }

    // Once each loop level reaches max, reset it and move to next loop level.
    void next(int idx) {
      if (idx >= indices.size())
        return;
      indices[idx]++;
      if (indices[idx] >= max_indices[idx]) {
        indices[idx] = 0;
        next(idx+1);
      }
    }

    // Start by incrementing the 0th index, i.e. inner-most loop.
    void next() {
      next(0);
    }

    // Each value combines (element idx within tile) + (tile idx)*(tile size).
    int getB() const { return indices[loopIdx.b] + indices[loopIdx.bT] * max_indices[loopIdx.b]; }
    int getM() const { return indices[loopIdx.m] + indices[loopIdx.mT] * max_indices[loopIdx.m]; }
    int getN() const { return indices[loopIdx.n] + indices[loopIdx.nT] * max_indices[loopIdx.n]; }
    int getK() const { return indices[loopIdx.k] + indices[loopIdx.kT] * max_indices[loopIdx.k]; }

    // Convert TileCoord(8D) to DotCoord(4D)
    DotCoord get() const {
      return DotCoord(getB(), getM(), getN(), getK());
    }
  }; // DotTilingCoord

  /*
    Specify from abstract parent class.
    Copy tiling parameters, and override coord state.
  */
  DotOrdering *getFirst() const {
    std::unique_ptr<DotOrdering> first = std::make_unique<DotTiling>(this);
    first.dotTilingCoord = DotTilingCoord(this, 0);
    return first;
  }

  DotOrdering *getLast() const {
    std::unique_ptr<DotOrdering> last = std::make_unique<DotTiling>(this);
    last.dotTilingCoord = DotTilingCoord(this, numRepB*numRepM*numRepN*numRepK);
    return first;
  }
  
  void next() {
    tileCoord.next();
  }

  DotCoord get() const {
    return dotTilingCoord.get();
  }

  /*
    DotTiling state.
  */
  private:
  // Constant tiling strategy.
  const int64_t numRepB;
  const int64_t numRepM;
  const int64_t numRepN;
  const int64_t numRepK;
  const int64_t tileSizeB;
  const int64_t tileSizeM;
  const int64_t tileSizeN;
  const int64_t tileSizeK;
  const bool outerTileN;

  const int64_t numTilesB;
  const int64_t numTilesM;
  const int64_t numTilesN;
  const int64_t numTilesK;
  const int64_t tileSizeOuter;
  const int64_t tileSizeInner;
  const int64_t numTilesOuter;
  const int64_t numTilesInner;

  // State for iterator.
  DotTilingCoord dotTilingCoord;

}; // DotTiling

/*


Create DotOrdering object
iterate over DotOrdering::iterator
each iterator return DotCoord()



*/