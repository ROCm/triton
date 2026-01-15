#!/usr/bin/env python3
"""
Triton Config Generator for GEMM Kernel
针对 AMD MI308X (80 CU, gfx942) 优化
"""

from itertools import product
from typing import List, Dict, Any
import random


def generate_triton_configs(
    max_configs: int = 50,
    verbose: bool = True
) -> List[str]:
    """
    生成 Triton Config 组合
    
    Args:
        max_configs: 最大配置数量
        verbose: 是否打印详细信息
    
    Returns:
        配置代码字符串列表
    """
    
    # ==================== 参数空间定义 ====================
    param_space = {
        'BLOCK_SIZE_M': [16, 32, 64, 128],
        'BLOCK_SIZE_N': [64, 128, 256, 512],
        'BLOCK_SIZE_K': [32, 64, 128, 256],
        'GROUP_SIZE_M': [1, 2, 4, 8],
        'waves_per_eu': [0, 2],
        'kpack': [1, 2],
        'matrix_instr_nonkdim': [16],  # 固定值
    }
    
    num_warps_options = [2, 4, 8, 16]
    # num_warps_options = [8]
    num_stages_options = [2]
    
    # ==================== 生成配置 ====================
    configs = []
    config_id = 0
    
    for m in param_space['BLOCK_SIZE_M']:
        for n in param_space['BLOCK_SIZE_N']:
            for k in param_space['BLOCK_SIZE_K']:
                for gm in param_space['GROUP_SIZE_M']:
                    for wpe in param_space['waves_per_eu']:
                        for kp in param_space['kpack']:
                            for warps in num_warps_options:
                                for stages in num_stages_options:
                                    
                                    # ========== 过滤规则（已注释） ==========
                                    
                                    # # 规则1: 块大小限制（放宽上限）
                                    # block_size = m * n
                                    # if block_size > 81920:  # 放宽到80K，允许更大的block
                                    #     continue
                                    # if block_size < 2048:  # 降低下限，允许更小的block
                                    #     continue
                                    
                                    # # 规则2: 小M场景优化（仅限制极小M）
                                    # if m <= 32 and gm > 4:  # 只限制M<=32且gm>4的情况
                                    #     continue
                                    
                                    # # 规则3: 大M场景优化（只排除gm=1）
                                    # if m >= 256 and gm == 1:  # 只排除最不合理的组合
                                    #     continue
                                    
                                    # # 规则4: K维度与kpack的匹配
                                    # if k < 64 and kp == 2:
                                    #     # 小K不需要kpack=2
                                    #     continue
                                    
                                    # # 规则5: num_warps与block size的匹配（大幅放宽）
                                    # work_per_warp = block_size / warps
                                    # if work_per_warp < 128:  # 降低下限
                                    #     continue
                                    # # 移除对大block的warps限制，让autotuner自己选择
                                    
                                    # # 规则6: num_stages与BLOCK_SIZE_K的匹配（放宽）
                                    # if k >= 256 and stages >= 4:  # 只限制最极端的情况
                                    #     # 超大K + 大stages会消耗太多LDS
                                    #     continue
                                    
                                    # # 规则7: waves_per_eu=0的限制
                                    # if wpe == 0 and stages > 2:
                                    #     # 不限制waves时，大stages可能导致资源冲突
                                    #     continue
                                    
                                    # ========== 生成配置 ==========
                                    block_size = m * n
                                    config = {
                                        'BLOCK_SIZE_M': m,
                                        'BLOCK_SIZE_N': n,
                                        'BLOCK_SIZE_K': k,
                                        'GROUP_SIZE_M': gm,
                                        'waves_per_eu': wpe,
                                        'kpack': kp,
                                        'matrix_instr_nonkdim': 16,
                                    }
                                    
                                    config_str = (
                                        f"        triton.Config(\n"
                                        f"            {{\n"
                                        f"                'BLOCK_SIZE_M': {m}, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': {k}, "
                                        f"'GROUP_SIZE_M': {gm}, 'waves_per_eu': {wpe},\n"
                                        f"                'kpack': {kp}, 'matrix_instr_nonkdim': 16\n"
                                        f"            }}, num_warps={warps}, num_stages={stages}),"
                                    )
                                    
                                    configs.append({
                                        'id': config_id,
                                        'config': config,
                                        'num_warps': warps,
                                        'num_stages': stages,
                                        'code': config_str,
                                        'block_size': block_size,
                                    })
                                    config_id += 1
                                    
                                    # 限制数量
                                    if len(configs) >= max_configs:
                                        break
                                if len(configs) >= max_configs:
                                    break
                            if len(configs) >= max_configs:
                                break
                        if len(configs) >= max_configs:
                            break
                    if len(configs) >= max_configs:
                        break
                if len(configs) >= max_configs:
                    break
            if len(configs) >= max_configs:
                break
        if len(configs) >= max_configs:
            break
    
    # ==================== 打印统计信息 ====================
    if verbose:
        print(f"生成了 {len(configs)} 个配置")
        print(f"\n参数分布统计:")
        
        # 统计各参数的分布
        for param in ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M']:
            values = [c['config'][param] for c in configs]
            unique_values = sorted(set(values))
            counts = {v: values.count(v) for v in unique_values}
            print(f"  {param}: {counts}")
        
        # num_warps和num_stages分布
        warps_counts = {}
        stages_counts = {}
        for c in configs:
            w = c['num_warps']
            s = c['num_stages']
            warps_counts[w] = warps_counts.get(w, 0) + 1
            stages_counts[s] = stages_counts.get(s, 0) + 1
        print(f"  num_warps: {warps_counts}")
        print(f"  num_stages: {stages_counts}")
    
    return [c['code'] for c in configs]


def select_representative_configs(
    all_configs: List[Dict[str, Any]], 
    target_count: int = 100,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    从所有有效配置中选择有代表性的子集
    
    策略：
    1. 分层采样：确保每个重要参数的各个值都有代表
    2. 多样性优先：覆盖参数空间的不同区域
    3. 性能倾向：优先选择可能性能更好的配置
    
    Args:
        all_configs: 所有有效配置列表
        target_count: 目标配置数量
        seed: 随机种子
    
    Returns:
        选中的配置列表
    """
    random.seed(seed)
    
    if len(all_configs) <= target_count:
        return all_configs
    
    print(f"\n从 {len(all_configs)} 个有效配置中选择 {target_count} 个代表性配置...")
    
    # 定义参数重要性权重和优先值
    param_importance = {
        'BLOCK_SIZE_M': 1.0,
        'BLOCK_SIZE_N': 1.0,
        'BLOCK_SIZE_K': 1.0,
        'GROUP_SIZE_M': 0.8,
        'num_warps': 0.9,
        'waves_per_eu': 0.7,
        'kpack': 0.6,
    }
    
    # 性能倾向配置 (这些值通常性能较好)
    preferred_values = {
        'BLOCK_SIZE_N': [128, 256],  # 中等N值通常较好
        'BLOCK_SIZE_K': [64, 128],   # 中等K值平衡内存和计算
        'kpack': [2],                # kpack=2通常更好
        'waves_per_eu': [2],         # 限制waves通常更稳定
    }
    
    selected = []
    remaining = all_configs.copy()
    
    # 阶段1: 强制覆盖 - 确保每个重要参数的每个值至少有一个代表
    print("阶段1: 确保参数空间覆盖...")
    coverage_targets = {
        'BLOCK_SIZE_M': set(),
        'BLOCK_SIZE_N': set(),
        'BLOCK_SIZE_K': set(),
        'GROUP_SIZE_M': set(),
        'num_warps': set(),
    }
    
    # 收集所有可能的值
    for config in all_configs:
        for param in coverage_targets:
            if param in config['config']:
                coverage_targets[param].add(config['config'][param])
            elif param == 'num_warps':
                coverage_targets[param].add(config['num_warps'])
    
    # 为每个参数值选择至少一个配置
    for param, values in coverage_targets.items():
        for value in sorted(values):
            # 找到包含该参数值的配置
            candidates = [c for c in remaining 
                         if (c['config'].get(param) == value or 
                             (param == 'num_warps' and c['num_warps'] == value))]
            
            if candidates and len(selected) < target_count:
                # 优先选择性能倾向配置
                scored_candidates = []
                for c in candidates:
                    score = 0
                    for pref_param, pref_vals in preferred_values.items():
                        if pref_param == 'num_warps':
                            if c.get('num_warps') in pref_vals:
                                score += 1
                        elif c['config'].get(pref_param) in pref_vals:
                            score += 1
                    scored_candidates.append((score, c))
                
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                chosen = scored_candidates[0][1]
                selected.append(chosen)
                remaining.remove(chosen)
    
    print(f"  覆盖采样选择了 {len(selected)} 个配置")
    
    # 阶段2: 分层采样 - 按BLOCK_SIZE组合分层
    print("阶段2: 分层采样...")
    if len(selected) < target_count and remaining:
        # 按 (M, N, K) 分组
        strata = {}
        for config in remaining:
            key = (config['config']['BLOCK_SIZE_M'], 
                   config['config']['BLOCK_SIZE_N'],
                   config['config']['BLOCK_SIZE_K'])
            if key not in strata:
                strata[key] = []
            strata[key].append(config)
        
        # 从每个层中采样
        per_stratum = max(1, (target_count - len(selected)) // len(strata))
        
        for stratum_configs in strata.values():
            if len(selected) >= target_count:
                break
            
            # 按性能倾向评分
            scored = []
            for c in stratum_configs:
                score = 0
                for pref_param, pref_vals in preferred_values.items():
                    if pref_param == 'num_warps':
                        if c.get('num_warps') in pref_vals:
                            score += 2
                    elif c['config'].get(pref_param) in pref_vals:
                        score += 2
                scored.append((score, c))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # 从这个层中选择
            n_to_select = min(per_stratum, len(scored), target_count - len(selected))
            for i in range(n_to_select):
                selected.append(scored[i][1])
                remaining.remove(scored[i][1])
    
    print(f"  分层采样后共 {len(selected)} 个配置")
    
    # 阶段3: 随机补充 - 如果还没达到目标数量
    print("阶段3: 多样性补充...")
    if len(selected) < target_count and remaining:
        needed = target_count - len(selected)
        
        # 优先选择性能倾向配置
        scored_remaining = []
        for c in remaining:
            score = 0
            for pref_param, pref_vals in preferred_values.items():
                if pref_param == 'num_warps':
                    if c.get('num_warps') in pref_vals:
                        score += 1
                elif c['config'].get(pref_param) in pref_vals:
                    score += 1
            scored_remaining.append((score, c))
        
        scored_remaining.sort(key=lambda x: x[0], reverse=True)
        
        # 前70%选最优，后30%随机
        top_portion = int(needed * 0.7)
        random_portion = needed - top_portion
        
        for i in range(min(top_portion, len(scored_remaining))):
            selected.append(scored_remaining[i][1])
        
        if random_portion > 0 and len(scored_remaining) > top_portion:
            random_choices = random.sample(scored_remaining[top_portion:], 
                                          min(random_portion, len(scored_remaining) - top_portion))
            selected.extend([c[1] for c in random_choices])
    
    print(f"  最终选择了 {len(selected)} 个配置")
    
    # 打印选择的配置分布
    print("\n选中配置的参数分布:")
    for param in ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M']:
        values = [c['config'][param] for c in selected]
        unique_values = sorted(set(values))
        counts = {v: values.count(v) for v in unique_values}
        print(f"  {param}: {counts}")
    
    warps_counts = {}
    for c in selected:
        w = c['num_warps']
        warps_counts[w] = warps_counts.get(w, 0) + 1
    print(f"  num_warps: {warps_counts}")
    
    return selected


def generate_representative_configs(
    target_count: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> List[str]:
    """
    生成有代表性的配置子集
    
    Args:
        target_count: 目标配置数量
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        配置代码字符串列表
    """
    # 首先生成所有有效配置（不限制数量）
    print("="*70)
    print(f"生成所有有效配置...")
    print("="*70)
    
    all_configs = []
    
    param_space = {
        'BLOCK_SIZE_M': [16, 32, 64, 128],
        'BLOCK_SIZE_N': [64, 128, 256, 512],
        'BLOCK_SIZE_K': [32, 64, 128, 256],
        'GROUP_SIZE_M': [1, 2, 4, 8],
        'waves_per_eu': [0, 2],
        'kpack': [1, 2],
        'matrix_instr_nonkdim': [16],
    }
    
    num_warps_options = [2, 4, 8, 16]
    num_stages_options = [2]
    
    config_id = 0
    for m in param_space['BLOCK_SIZE_M']:
        for n in param_space['BLOCK_SIZE_N']:
            for k in param_space['BLOCK_SIZE_K']:
                for gm in param_space['GROUP_SIZE_M']:
                    for wpe in param_space['waves_per_eu']:
                        for kp in param_space['kpack']:
                            for warps in num_warps_options:
                                for stages in num_stages_options:
                                    
                                    # 应用过滤规则（已注释）
                                    block_size = m * n
                                    # if block_size > 81920 or block_size < 2048:
                                    #     continue
                                    # if m <= 32 and gm > 4:
                                    #     continue
                                    # if m >= 256 and gm == 1:
                                    #     continue
                                    # if k < 64 and kp == 2:
                                    #     continue
                                    # work_per_warp = block_size / warps
                                    # if work_per_warp < 128:
                                    #     continue
                                    # if k >= 256 and stages >= 4:
                                    #     continue
                                    # if wpe == 0 and stages > 2:
                                    #     continue
                                    
                                    # 生成配置
                                    config = {
                                        'BLOCK_SIZE_M': m,
                                        'BLOCK_SIZE_N': n,
                                        'BLOCK_SIZE_K': k,
                                        'GROUP_SIZE_M': gm,
                                        'waves_per_eu': wpe,
                                        'kpack': kp,
                                        'matrix_instr_nonkdim': 16,
                                    }
                                    
                                    config_str = (
                                        f"        triton.Config(\n"
                                        f"            {{\n"
                                        f"                'BLOCK_SIZE_M': {m}, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': {k}, "
                                        f"'GROUP_SIZE_M': {gm}, 'waves_per_eu': {wpe},\n"
                                        f"                'kpack': {kp}, 'matrix_instr_nonkdim': 16\n"
                                        f"            }}, num_warps={warps}, num_stages={stages}),"
                                    )
                                    
                                    all_configs.append({
                                        'id': config_id,
                                        'config': config,
                                        'num_warps': warps,
                                        'num_stages': stages,
                                        'code': config_str,
                                        'block_size': block_size,
                                    })
                                    config_id += 1
    
    print(f"共生成 {len(all_configs)} 个有效配置")
    
    # 选择代表性配置
    selected_configs = select_representative_configs(all_configs, target_count, seed)
    
    return [c['code'] for c in selected_configs]


def generate_specialized_configs(scenario: str = 'small_m') -> List[str]:
    """
    生成针对特定场景的配置
    
    Args:
        scenario: 'small_m' (M<128), 'medium_m' (128<=M<512), 'large_m' (M>=512)
    
    Returns:
        配置代码字符串列表
    """
    configs = []
    
    if scenario == 'small_m':
        # 针对 M=64, N=3072, K=2048 这种情况
        print("生成小M场景配置 (M < 128)...")
        
        for n in [128, 256]:
            for k in [64, 128]:
                for stages in [2, 3, 4]:
                    # 配置1: 保守配置
                    configs.append(
                        f"        triton.Config(\n"
                        f"            {{\n"
                        f"                'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': {k}, "
                        f"'GROUP_SIZE_M': 1, 'waves_per_eu': 2,\n"
                        f"                'kpack': 2, 'matrix_instr_nonkdim': 16\n"
                        f"            }}, num_warps=8, num_stages={stages}),"
                    )
                    
                    # 配置2: 中等配置
                    configs.append(
                        f"        triton.Config(\n"
                        f"            {{\n"
                        f"                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': {k}, "
                        f"'GROUP_SIZE_M': 2, 'waves_per_eu': 2,\n"
                        f"                'kpack': 2, 'matrix_instr_nonkdim': 16\n"
                        f"            }}, num_warps=8, num_stages={stages}),"
                    )
    
    elif scenario == 'medium_m':
        print("生成中等M场景配置 (128 <= M < 512)...")
        
        for m in [128]:
            for n in [128, 256]:
                for k in [32, 64, 128]:
                    for stages in [2, 3]:
                        configs.append(
                            f"        triton.Config(\n"
                            f"            {{\n"
                            f"                'BLOCK_SIZE_M': {m}, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': {k}, "
                            f"'GROUP_SIZE_M': 4, 'waves_per_eu': 2,\n"
                            f"                'kpack': 2, 'matrix_instr_nonkdim': 16\n"
                            f"            }}, num_warps=8, num_stages={stages}),"
                        )
    
    elif scenario == 'large_m':
        print("生成大M场景配置 (M >= 512)...")
        
        for m in [256]:
            for n in [128, 256]:
                for stages in [2, 3]:
                    for wpe in [0, 2]:
                        configs.append(
                            f"        triton.Config(\n"
                            f"            {{\n"
                            f"                'BLOCK_SIZE_M': {m}, 'BLOCK_SIZE_N': {n}, 'BLOCK_SIZE_K': 64, "
                            f"'GROUP_SIZE_M': 4, 'waves_per_eu': {wpe},\n"
                            f"                'kpack': 2, 'matrix_instr_nonkdim': 16\n"
                            f"            }}, num_warps=8, num_stages={stages}),"
                        )
    
    print(f"生成了 {len(configs)} 个专用配置")
    return configs


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 Triton GEMM 配置')
    parser.add_argument('--max-configs', type=int, default=30, help='最大配置数量')
    parser.add_argument('--scenario', type=str, choices=['all', 'small_m', 'medium_m', 'large_m'],
                       default='all', help='配置场景')
    parser.add_argument('--representative', type=int, metavar='N', 
                       help='生成N个有代表性的配置（从所有有效配置中智能选择）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于代表性采样）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Triton Config Generator for AMD MI308X")
    print("="*70)
    print()
    
    # 生成配置
    if args.representative:
        configs = generate_representative_configs(
            target_count=args.representative, 
            seed=args.seed,
            verbose=True
        )
    elif args.scenario == 'all':
        configs = generate_triton_configs(max_configs=args.max_configs, verbose=True)
    else:
        configs = generate_specialized_configs(scenario=args.scenario)
    
    print()
    print("="*70)
    print("生成的配置代码:")
    print("="*70)
    print()
    print("    configs=[")
    for config in configs:
        print(config)
    print("    ],")
    print()
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Auto-generated Triton Configs\n")
            f.write("# Generated by generate_configs.py\n\n")
            f.write("configs = [\n")
            for config in configs:
                f.write(config + "\n")
            f.write("]\n")
        print(f"✓ 配置已保存到: {args.output}")
    else:
        print("提示: 使用 --output 参数保存到文件")
        print("或者直接复制上面的代码到 gemm.py 的 @triton.autotune(configs=[...]) 中")


if __name__ == '__main__':
    main()

