# Enable paddedShared layout with in-thread-transpose

## Original Case

Triton compiler: 81d266942d

```
 python fa/flash-attention.py 
```

LWV and LRV instructions are saved in `orig_itt_swizzled/ds_inst_V.s`,
in which we can see
- 8 x `ds_write_b32` that need 8 vgprs to hold addresses (v99-106)
- 28 x `ds_read_b64` + 2 x `ds_read2st64_b64` that need 29 vgprs to hold addresses

We need 37 vgprs just for addresses of ds instructions for V tensor.
This is due to the design of rotatingShared layout.

## First attempt of paddedShared layout, 512:8

triton compiler: 7e6a3da26490

The above commit enables a naive paddedShared layout for in-thread-transpose pass.
It's naive because elements in the tensor are distributed linearly in the LDS, thus
LDS bank conflicts are expected.
This commit is to demonstrate the effectiveness of paddedShared layout in terms
of vgpr usage.

LWV and LRV instructions are saved in `padding_512-8/ds_inst_V.s`,
in which we can see
- 4 x `ds_write2_b32` that need 2 vgprs to hold addresses (v166 and v172)
- 16 x `ds_read2_b64` that need 4 vgprs to hold addresses (v167, v176-178)

With linear distribution of elements in LDS, we should expect even fewer
vgpr usages for addresses. This is already a huge reduction compared to 37 vgprs,
and we still need to figure out a better distribution to avoid bank conflicts.
Therefore, we won't try to further reduce vgprs based on this version.
