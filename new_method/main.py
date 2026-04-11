#!/usr/bin/env python3
"""
L-BFGS-GPR 混合优化项目主程序
分子几何构型优化 - L-BFGS 与 GPR 混合策略
"""
import os
import sys
import argparse
import yaml
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.molecule import Molecule
from core.calculator import QuantumCalculator
from optimizers.lbfgs import LBFGSOptimizer, run_lbfgs_optimization
from optimizers.hybrid import HybridOptimizer, run_hybrid_optimization
from visualization.structure3d import MoleculeVisualizer3D
from visualization.plots import OptimizationPlotter
from utils.io_utils import OutputManager, create_output_manager


def _get_ai_method_suffix(ai_method: str) -> str:
    """获取 AI 方法的简短后缀"""
    suffix_map = {
        'simple': 'gpr',
        'gradient': 'ggpr',
        'random_forest': 'rf',
        'neural_network': 'nn'
    }
    return suffix_map.get(ai_method, ai_method) if ai_method else ''


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置"""
    result = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def run_optimization(method: str, molecule: Molecule, config: Dict[str, Any],
                     output_manager: OutputManager,
                     ai_method: str = None) -> Dict[str, Any]:
    """
    运行优化

    Args:
        method: 优化方法 ('lbfgs' 或 'hybrid')
        molecule: 初始分子
        config: 配置字典
        output_manager: 输出管理器
        ai_method: AI 方法类型（'simple'/'gradient'/'random_forest'等）

    Returns:
        results: 优化结果
    """
    print(f"\n{'='*70}")
    print(f"运行优化：{method.upper()}")
    print(f"{'='*70}")
    
    # 创建计算器
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )
    
    # 选择优化器
    if method == 'lbfgs':
        optimizer = LBFGSOptimizer(config)
    elif method == 'hybrid':
        optimizer = HybridOptimizer(config)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 执行优化
    history = optimizer.optimize(molecule, calculator)
    
    # 保存结果
    metadata = {
        'method': method,
        'molecule': molecule.name,
        'smiles': molecule.smiles,
        'n_atoms': molecule.n_atoms,
        'config': config
    }
    
    # 保存历史
    output_manager.save_history(history, method, metadata)
    
    # 保存轨迹
    output_manager.save_trajectory(history, method, molecule.atom_symbols)
    
    # 保存详细迭代信息
    if config.get('output', {}).get('save_details', True):
        output_manager.save_iteration_details(history, method, molecule.atom_symbols)
    
    # 获取最优结构
    best_coords = history.get_best_coords()
    if best_coords is not None:
        best_mol = Molecule(
            molecule.atom_symbols,
            best_coords.reshape(-1, 3),
            molecule.smiles,
            f"{molecule.name}_best_{method}"
        )
        output_manager.save_final_structure(best_mol, method)
    
    # 生成图表
    vis_config = config.get('visualization', {})
    plotter = OptimizationPlotter(
        font_size=vis_config.get('font_size', 14),
        figure_size=tuple(vis_config.get('figure_size', [12, 8])),
        dpi=vis_config.get('dpi', 300),
        ai_method=ai_method  # 传递 AI 方法类型
    )

    plots_dir = os.path.join(output_manager.save_dir, 'plots')
    
    # 构建图表标题前缀：包含 AI 方法信息
    if ai_method:
        ai_suffix = _get_ai_method_suffix(ai_method)
        title_prefix = f"{method}_{ai_suffix} - "
    else:
        title_prefix = f"{method} - "
    
    plotter.plot_all(history, f"{plots_dir}/{method}", title_prefix, ai_method=ai_method)

    # 生成 3D 结构图
    if vis_config.get('show_3d_structure', True):
        vis = MoleculeVisualizer3D(
            font_size=vis_config.get('font_size', 14),
            figure_size=tuple(vis_config.get('figure_size', [10, 8])),
            dpi=vis_config.get('dpi', 300),
            ai_method=ai_method  # 传递 AI 方法类型
        )

        # 初始结构
        output_manager.save_initial_structure(molecule, method)

        # 最优结构（使用 best_coords 而不是 final）
        best_mol = Molecule(
            molecule.atom_symbols,
            best_coords.reshape(-1, 3),
            molecule.smiles,
            f"{molecule.name}_best"
        )
        
        # 保存最优结构为 XYZ 文件
        output_manager.save_final_structure(best_mol, method)

        structures_dir = os.path.join(output_manager.save_dir, 'structures')

        # 构建结构文件名：如果有 AI 方法后缀，则添加到文件名中
        if ai_method:
            ai_suffix = _get_ai_method_suffix(ai_method)
            struct_filename = f"{method}_{ai_suffix}_comparison.png"
        else:
            struct_filename = f"{method}_comparison.png"

        # 图题中标注 AI 方法
        if ai_method:
            titles = [
                'Initial', 
                f'Best ({ai_method})\nE={best_mol.coords[0,0]:.4f}...'
            ]
        else:
            titles = ['Initial', f'Best (E={best_mol.coords[0,0]:.4f}...)']

        vis.visualize_comparison(
            [molecule, best_mol],
            titles=titles,
            save_path=f"{structures_dir}/{struct_filename}",
            elevation=30,  # 更有立体感的仰角
            azimuth=45     # 方位角
        )
    
    # 返回结果
    best = history.get_best_iteration('energy')
    results = {
        'method': method,
        'converged': history.converged,
        'best_energy': best.energy if best else None,
        'best_gradient_norm': best.gradient_norm if best else None,
        'iterations': len(history),
        'history': history
    }
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='L-BFGS-GPR 混合优化 - 分子几何构型优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --method lbfgs --molecule ethanol
  python main.py --method hybrid --molecule ethanol --perturb 0.5
  python main.py --config my_config.yaml
        """
    )
    
    parser.add_argument('--method', type=str, default='lbfgs',
                       choices=['lbfgs', 'hybrid'],
                       help='优化方法 (default: lbfgs)')
    parser.add_argument('--molecule', type=str, default='ethanol',
                       help='分子名称或 SMILES (default: ethanol)')
    parser.add_argument('--smiles', type=str, default=None,
                       help='SMILES 字符串 (覆盖 --molecule)')
    parser.add_argument('--perturb', type=float, default=0.0,
                       help='初始扰动强度 (Å) (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--max-iter', type=int, default=None,
                       help='最大迭代次数')
    parser.add_argument('--threshold', type=float, default=None,
                       help='收敛阈值')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.output:
        if 'output' not in config:
            config['output'] = {}
        config['output']['save_dir'] = args.output
    
    if args.max_iter:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['max_iterations'] = args.max_iter
    
    if args.threshold:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['convergence_threshold'] = args.threshold
    
    # 分子设置
    if 'molecule' not in config:
        config['molecule'] = {}
    
    if args.smiles:
        config['molecule']['smiles'] = args.smiles
    elif args.molecule == 'ethanol':
        config['molecule']['smiles'] = 'CCO'
    elif args.molecule == 'water':
        config['molecule']['smiles'] = 'O'
    elif args.molecule == 'methane':
        config['molecule']['smiles'] = 'C'
    else:
        config['molecule']['smiles'] = 'CCO'  # 默认乙醇
    
    config['molecule']['seed'] = args.seed
    config['molecule']['perturb_strength'] = args.perturb

    # 获取 AI 方法类型（用于输出文件命名）
    ai_method = None
    if args.method == 'hybrid':
        ai_method = config.get('gpr', {}).get('type', 'simple')

    # 创建输出管理器
    output_manager = create_output_manager(config, ai_method=ai_method)

    # 创建分子
    smiles = config['molecule']['smiles']
    perturb = config['molecule']['perturb_strength']
    seed = config['molecule']['seed']

    print(f"\n{'='*70}")
    print("L-BFGS-GPR 混合优化项目")
    print(f"{'='*70}")
    print(f"分子：{smiles}")
    print(f"扰动：{perturb} Å")
    print(f"种子：{seed}")
    print(f"方法：{args.method}")
    if ai_method:
        print(f"AI 方法：{ai_method}")
    print(f"输出目录：{output_manager.save_dir}")
    print(f"{'='*70}")
    
    # 生成初始分子
    molecule = Molecule.from_smiles(smiles, seed=seed, perturb_strength=perturb)
    
    print(f"\n初始结构:")
    print(f"  原子数：{molecule.n_atoms}")
    print(f"  自由度：{molecule.n_atoms * 3}")
    
    # 运行优化
    results = run_optimization(args.method, molecule, config, output_manager, ai_method)
    
    # 打印结果
    print(f"\n{'='*70}")
    print("优化结果")
    print(f"{'='*70}")
    print(f"方法：{results['method']}")
    print(f"收敛：{'是' if results['converged'] else '否'}")
    print(f"最优能量：{results['best_energy']:.10f} Hartree")
    print(f"最优梯度范数：{results['best_gradient_norm']:.6f}")
    print(f"迭代次数：{results['iterations']}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main()
