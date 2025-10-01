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

## Pointer Canonicalization

In the IR, the load of the child tile and pointer advance is usually like
```mlir
%in_ptrs_init = ...
scf.for %i = %c0 to %max step %c1 iter_args(%in_ptrs = %in_ptrs_init) ->
(tensor<128x64x!tt.ptr<f16>, #blocked> ) : i32 {
    %data = tt.load %in_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked>
    %in_ptrs_new = tt.addptr %in_ptrs, %cst : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    scf.yield %in_ptrs_new : tensor<128x64x!tt.ptr<f16>, #blocked>
}
```

The pointer canonicalizer will process `%in_ptrs_init` into uniform and thread-load parts.
The above IR will be transformed into
```mlir
%offset_init = ...
scf.for %i = %c0 to %max step %c1 iter_args(%offset = %offset_init) ->
(tensor<128x64xi32, #blocked> ) : i32 {
    %bPtrParent = tt.splat %in_ptr : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %in_ptrs = tt.addptr %bPtrParent, %offset
    %data = tt.load %in_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked>
    %offset_new = arith.addi %offset, %cst : tensor<128x64xi32, #blocked>
    scf.yield %offset_new : tensor<128x64xi32, #blocked>
}
```

Now the thread-local `offset` is made loop carried and it's updated by a constant
`%cst` inside the loop.


## Conversion to Buffer ops

The above IR format can be naturally converted to use `buffer_load` as
```mlir
%offset_init = ...
scf.for %i = %c0 to %max step %c1 iter_args(%offset = %offset_init) ->
(tensor<128x64xi32, #blocked> ) : i32 {
    %data = amdgpu.buffer_load %in_ptr[%offset] : tensor<128x64xf16, #blocked> loc(#loc32)
    %offset_new = arith.addi %offset, %cst : tensor<128x64xi32, #blocked>
    scf.yield %offset_new : tensor<128x64xi32, #blocked>
}
```

The above IR
