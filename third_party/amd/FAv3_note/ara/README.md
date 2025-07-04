# Advanced Register Analyzer `ara.py`

Usage:
```bash
# cd into current dir
python3 ara.py <FAv3 assembly file>
```

This tool analyzes the register usage of the computer clusters in the main loop for FAv3 kernels.
The register usage is saved in `ara/reg_usage.txt`. 
The splitted clusters are saved in `ara/cluster_x`.
The contents of the reg usage for
`impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue/attn_fwd.s` is shown below
```bash
QK: 32;66:97
PP: 33;1,195:222,246:249
P: 16;114:129
K: 64;80:95,114:129,162:193
V: 64;162:193,196:223,246:249
Q: 32;130:161
ACC: 64;2:65
```
The format is `SYMBOL: N;x:y`, which means
- SYMBOL: the symbol for the tensor.
  - QK: result tensor of 1st dot.
    - These values are produced by mfma instructions in cluster 0.
    - These values are used (or updated online) by `v_max`, `v_fma`, and `v_exp` instructions in cluster 2.
  - PP: result of exp(QK) in 32-bit.
    - These values are produced by `v_exp` instructions in cluster 2.
    - These values are used by `v_add` and `v_cvt_pkrtz` instructions in cluster 0.
  - P:  result of exp(QK) in 16-bit. This is the result of truncation from PP.
    - These values are produced by `v_cvt_pkrtz` instructions in cluster 0
    - These values are used as the 2-th operands of mfma instructions in cluser 2.
  - K:  K tensor.
    - These values are read by `ds_read` instructions in cluster 3.
    - These values are used as the 1-th operands of mfma instructions in cluster 0.
  - V:  V tensor.
    - These values are read by `ds_read` instructions in cluster 1.
    - These values are used as the 1-th operands of mfma instructions in cluster 2.
  - Q:  Q tensor. Loop invariant.
    - These values are read into registers in the prologue.
    - These values are used as the 2-th operands of mfma instructions in cluster 0.
  - ACC: result tensor of the 2nd dot. Loop invariant
    - These values are initialized in registers in th prologue.
    - These values are used and updated as the accumulator (0-th and 3-th operands) of
      mfma instructions in cluster 2.
- N: number of registers used to hold values for this tensor
- x:y: register indices

The `reg_usage.txt` file tells us the register usage of these tensors.
What is more interesting is the overlap of these registers. For example, K tensor and V tensor
have disjoint liveness in registers, which means we can use the exactly same set of registers
for K and V. The LLVM backend compiler may or may not allocate registers as we expect.
The above command will output the overlap of each pair of tensors.
It also outputs the total number of registers, together with the register indices 
for all of these tensors.
