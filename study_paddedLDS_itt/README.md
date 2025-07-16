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

pmc counters: 0,0
UNALIGNED: 0

tflops: 475

## First attempt of paddedShared layout: <512:+8>

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

## 3-layer paddedShared layout: <64:+2,2048:+4,4096:+16>

triton compiler: bb19837d979ba8

This commit uses a 3 layer of padding information to avoid bank conflicts
for both `ds_read` and `ds_write` when in-thread-transpose is enabled.
The previous commit leads to explosive vgpr usage for `ds_read` addresses
for V tensor. This is because the `emitPadding` function is applied to
the final LDS offset, which contains both `Value` (from lane id and warp id)
and constants (from reg id).
Since the `emitPadding` function involves `shl` and `shr`, and the more
layers of the padding, the more `shl`-`shr` chains are involved.
Therefore, the backend compiler is not able to optimize the addresses.

This commit fixes this issue by applying the `emitPadding` function
separately to the `Value` part ([code](https://github.com/ROCm/triton/blob/bb19837d979ba844feb9163063421d4a619ce53d/lib/Conversion/TritonGPUToLLVM/Utility.cpp#L791-L792))
and constant part ([code](https://github.com/ROCm/triton/blob/bb19837d979ba844feb9163063421d4a619ce53d/lib/Conversion/TritonGPUToLLVM/Utility.cpp#L801)).

LWV and LRV instructions are saved in `padding_64-2_2048-4_4096_16/ds_inst_V.s`,
in which we can see
- 4 x `ds_write2_b32` that need 1 vgpr to hold addresses (v164)
- 16 x `ds_read2_b64` that need 4 vgprs to hold addresses (v165, v167, v172-173)

Now the `ds_write_b32` can be combined to `ds_write2_b32` and shared the same
base vgpr. `ds_read` need 4 vgprs because of the limit of max `offset` value
of the instruction. `offset` has 8 bits and it's max value is 255, which means
an offset of 2040 bytes.
However, in this kernel, we need offsets as large as 4096 and 8192 bytes.
Therefore, 4 vgprs for the addresses is already the best we can do.

tflops: 242

pmc counters: 0,0
UNALIGNED_STALL: 12079595520

## 3-layer paddedShared layout: <64:+2,2048:+4,4096:+16> with row rotating (RR0)


triton compiler: 325c1eaf3f0

The current paddedShared layout does not provide information about
the underlying linear layout for LDS.
The default layout is an identity linear layout, which leads to bank conflicts
even with padding.
This commit applies row rotating in the following order:
```bash
0, 16, 32, 48, 64, 80, 96, 112,
1, 17, 33, 49, 65, 81, 97, 113,
...
15, 31, 47, 63, 79, 95, 111, 127,
128, ...
```

LWV and LRV instructions are saved in `padding_64-2_2048-4_4096_16_RR0/ds_inst_V.s`,
in which we can see
- 8 x `ds_write_b32` that need 1 vgpr to hold address (v164)
  - The backend does not generate `ds_write2_b64`.
    The distance between offsets (1056) should be representable as `offset0/1`.
- 16 x `ds_read2_b64` that need 1 vgpr to hold address (v165)
  - With RR0, the elements read by each thread is closer to each other
    so that relatively smaller offsets can be used.


tflops: 334

However, thread trace shows that `ds_read2_b64` still take very long to issue.
I did not check the counters for bank conflicts, but this is very likely
to be bank conflicts.

Not sure why performance improves.

pmc counters: 100663296,0.113366
UNALIGNED: 6039797760


## RR1 + pad<128:+2>

triton compiler: bee8cd64da3f03bdad8

Another design of the padding and RR pattern. Row rotating order is as follows
```bash
0, 4, 8, 12, 16, 20, ... 60,
1, 5, 9, 13, 17, 21, ... 61,
2, 6, 10, 14, 18, 22, ..., 62,
3, 7, ..................., 63,
64, ...
```

This version has a lot of bank conflicts. And the perf goes back to 200 tflops.

pmc counters: 301989888,0.213392


## pad<64:+4> without RR

triton compiler: f6b3bd22aa

This version removes the RR and uses a very simple padding pattern <64:+4>.
This avoids bank conflicts for `ds_read2_b64`, but has a few conflicts for `ds_write2_b32`.

thread trace shows that `ds_read2_b64` do not have any conflicts.
And `ds_write2_b32` do take longer to issue.

However, pmc counters (`SQ_LDS_BANK_CONFLICT` and `LDSBankConflict`) show
larger numbers for bank conflicts.

pmc counters: 704643072,1.339672
UNALIGNED_STALL: 0

tflops: 460


## pad<64:+4, 1024:+4> without RR

triton compiler: 38d89639dc

Based on the previous commit, the 2nd layer of padding, 1024:+4, helps to avoid
bank conflicts for `ds_write2_b32`.


bank conflicts: 0,0
UNALIGNED: 0

tflops: 488


## Summary

| LDS layout                             | `SQ_LDS_BANK_CONFLICT` | `LDSBankConflict` | `SQ_LDS_UNALIGNED_STALL` | vgprs | TFLOPS |
|----------------------------------------|------------------------|-------------------|--------------------------|-------|--------|
| Baseline: RotatingShared               | 0                      | 0                 | 0                        | 215   | 475    |
| Padding<64:+2,2048:+4,4096:+16> w/o RR | 0                      | 0                 | 12079595520              | 176   | 242    |
| Padding<64:+2,2048:+4,4096:+16> w/ RR0 | 100663296              | 0.1133            | 6039797760               | 176   | 334    |
| Padding<64:+4> w/o RR                  | 704643072              | 1.3396            | 0                        | 176   | 460    |
| Padding<64:+4,1024:+4> w/o RR          | 0                      | 0                 | 0                        | 176   | 488    |


### rocprofv2

To collect these counters with rocprofv2, put the following into pmc.txt
```
pmc: SQ_LDS_BANK_CONFLICT, LDSBankConflict, SQ_LDS_UNALIGNED_STALL
```
and run rocprofv2 as
```bash
rocprofv2 -i pmc.txt -d pmc_result python fa/flash-attention.py
```

The meaning of these counters can be found in `rocprofv2 --list-counters`:
- `SQ_LDS_UNALIGNED_STALL` : Number of cycles LDS is stalled processing flat unaligned load/store ops. (emulated)
- `SQ_LDS_BANK_CONFLICT` : Number of cycles LDS is stalled by bank conflicts. (emulated)
- `LDSBankConflict` : The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0% (optimal) to 100% (bad).

To collect thread trace, put the following into att.txt
```
att: TARGET_CU=0
SE_MASK=0x1
SIMD_SELECT=0xF
ISA_CAPTURE_MODE=2
DISPATCH=30
```
and run rocprofv2 as
```bash
rocprofv2 -i att.txt --plugin att auto --mode file -d ./att_output python fa/flash-attention.py
```

### Observations and insights

- Padding can reduce the number of vgprs when in-thread-transpose is used
  - For multi layer padding pattern, we need to apply padded offset carefully,
    as done in bb19837d979
- When LDS accesses are not aligned, the perf drops a lot. The observation from thread trace
  is that ds_ instructions takes very long to issue.
  This can be monitored by `SQ_LDS_UNALIGNED_STALL`.
- For padding <64:+2,2048:+4,4096:+16>, the effect of RR0 is
  - Introduced bank conflicts. <-- This is counter-intuitive !!!
  - Reduced unaligned LDS accesses. And since alignment is more important than
    bank conflicts, RR0 improved perf a little.
- For padding <64:+4,1024:+4> without RR, this is perfect in terms of
  bank conflicts and aligned accesses.
  However, this padding pattern uses a lot extra LDS.
  This should be fine in current Triton for most cases.
