# Study the effect of cache swizzling


A simple kernel that does self-add of a tensor is provided in `issues/cache_swizzling.py`.
The tensor is loaded twice, one with `buffer_load` and the other with `global_load`.
The `buffer_load` has the cache swizzling bit set as follows
```asm
s_or_b32 s1, s1, 2.0
buffer_load_dwordx2 v[4:5], v3, s[0:3], 0 offen
```
The `s_or s1, s1, 2.0` is effectly setting the 62-th bit of s0:s1, which is
the cache swizzle bit according to table 47 of the [mi300 official doc](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)

The kernel returns incorrect results compared to torch kernel.
But if we comment out the `s_or` instruction, the results are correct.
Here are the commands to run the assembly code. 

- correct results: `TRITON_ALWAYS_COMPILE=1 AMD_INSERT_AMDGCN=issues/add_kernel_noCacheSwizzle.s python issues/cache_swizzling.py `
- wrong results: `TRITON_ALWAYS_COMPILE=1 AMD_INSERT_AMDGCN=issues/add_kernel_orig.s python issues/cache_swizzling.py `
