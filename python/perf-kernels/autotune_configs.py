import triton

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ("gfx1030", "gfx1100", "gfx1101",
                                                                                   "gfx1102", "gfx1200", "gfx1201")


def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
    ], [
        'CAUSAL_TYPE', 'dropout_p', 'Max_seqlen_q', 'Max_seqlen_k', 'Head_dim', 'Num_seqlens', 'Num_head_q',
        'Num_head_k'
    ]


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
    ], [
        'CAUSAL_TYPE', 'dropout_p', 'Max_seqlen_q', 'Max_seqlen_k', 'Head_dim', 'Num_seqlens', 'Num_head_q',
        'Num_head_k'
    ]


def get_autotune_configs():
    if is_rdna():
        return get_rdna_autotune_configs()
    elif is_cdna():
        return get_cdna_autotune_configs()
    else:
        raise ValueError("Unknown Device Type")
