IS_JIT_COMPILING = True
if IS_JIT_COMPILING:
    from triton.language import constexpr as constexpr_or_i32
    from triton.language import constexpr as constexpr_or_f32
    # from triton.language import constexpr as constexpr_or_bool  # Unused for now
else:
    from triton.language import int32 as constexpr_or_i32
    from triton.language import float32 as constexpr_or_f32
    # from triton.language import int1 as constexpr_or_bool
from triton.language import uint64 as u64  # type annotation of strides
