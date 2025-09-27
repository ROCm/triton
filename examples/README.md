# Re-design buffer ops


Fundamentally the kernel is accessing one or more sub-tiles from a parent tensor.
Let's say the base pointer of the parent tensor and the child are `bPtrParent` and `bPtrChild`. 
The parent tensor has stride `strideM` and `strideN`.
Note that the child tile share the same strides as the parent.

We are always given the `bPtrParent` and what we need is
1. Get `bPtrChild`
2. Spawn the pointers, i.e. make_range, to cover the 2D child tile

Let's assume the tensor is always 2D.
For each dimentsion, there are three parameters:
1. `Distance`: between bPtrParent and bPtrChild along this dim. E.g.`pid * BLOCK_M`.
   This is threads uniform, meaning it's shared by all threads.
2. `Range`: make_range to cover the elements along this dim for the child tile.
   E.g. `make_range(0, BLOCK_M)`. This is threads variant, meaning
   different threads can have different values.
3. `Stride`: along this dim. This is given by unit of elements.

Therefore, the addresses along this dim is
```python
bPtrParent + (Distance + Range) * Stride
```

The addresses for the child tile is
```python
bPtrParent + (Distance0 + Range0) * Stride0 [:, None] + (Distance1 + Range1) * Stride1 [None, :]
```

This is the most intuitive way to calculate the addresses for the child tile
and is used in the tutorial.




