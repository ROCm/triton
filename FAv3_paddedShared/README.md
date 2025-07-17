# FAv3 on MI300 with Padded LDS Layout

Continute the development based on triton branch: `enable_paddedSharedLayout`.

## Merge the new 4-stage pipeline and pingpong

- Merged in https://github.com/ROCm/triton/commits/aweinrau/fourStage/ @a7efc093fbec768da5480138
- Adjust pipeline scheduler and pingpong <-- this is done by d0e810eec6e
  - Need to adjust `scheduleOpsBetweenDots` based on https://github.com/ROCm/triton/pull/842.
    Otherwise, all valu ops go to VEC2. 
- Add `remove_layout_conversions` pass at the end of `make_ttgir`.
  Otherwise, there is an extra `convert_layout` in the epilogue that causes out of LDS error.
  <-- this is done by 017559de92d2

### More notes about the `convert_layout` issue

The IR after `in_thread_transpose` is saved in `FAv3_paddedShared/rm_cvt_issue/after_itt_before_rmcvt.mlir`.
The IR after `remove_layout_conversions` is saved in `FAv3_paddedShared/rm_cvt_issue/after_itt_rmcvt.mlir`.

The IR looks like
```mlir
%157:13 = scf.for %arg27 = %c0_i32 to %c8000_i32 step %c64_i32 iter_args(
  %arg28 = %cst_1, %arg29 = %cst_3, %arg30 = %142, %arg31 = %139, 
  %arg32 = %155, %arg33 = %c0_i32, %arg34 = %151, %arg35 = %148, 
  %arg36 = %150, %arg37 = %152, %arg38 = %140, %arg39 = %156, %arg40 = %154) {
  // other stuff
  %260 = tt.addptr %arg40, %97 : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
  // This is the only user of %arg32
  %261 = tt.addptr %arg32, %98 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
  %262 = tt.load %260 : tensor<64x128x!tt.ptr<f16>, #blocked2>
  scf.yield %245, %235, %247, %243, %261, %241, %257, %253, %256, %258, %244, %262, %260
}
// stuff in the epilogue
%187 = tt.addptr %157#4, %98 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
%188 = ttg.convert_layout %187 : tensor<64x128x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked2>
%189 = tt.load %188 : tensor<64x128x!tt.ptr<f16>, #blocked2>
```

It seems the `remove_layout_conversions` pass makes a copy of of the pointer tensor
for V and keep updating two copies (`%260` and `%261`).
Then in the epilogue, `%261` is used, but converted to `#blocked2` layout, which is
the layout of `%260`.

So why does `remove_layout_conversions` pass do so?

## Baseline Performance

compiler commit: 017559de92d2364dced

command
```bash
AMDGCN_SCALARIZE_PACKED_FOPS=1 python fa/flash-attention.py
```

tflops: 470
vgprs: 256 with 46 spills

Todo: need to rebase TritonInterleaveAndRematRebase0 on top of 570885128351868c1308bb22e8ca35

