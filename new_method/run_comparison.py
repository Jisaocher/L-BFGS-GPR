#!/usr/bin/env python3
"""
对比运行脚本
运行 L-BFGS 和 L-BFGS+GPR 混合优化，进行横向对比
"""
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.molecule import Molecule
from optimizers.lbfgs import LBFGSOptimizer
from optimizers.hybrid import HybridOptimizer
from core.calculator import QuantumCalculator
from visualization.plots import OptimizationPlotter
from visualization.structure3d import MoleculeVisualizer3D
from utils.io_utils import OutputManager


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def run_comparison(smiles: str = 'CCO', perturb: float = 0.0, seed: int = 42,
                   config: Dict[str, Any] = None,
                   output_dir: str = None) -> Dict[str, Any]:
    """
    运行对比实验
    
    Args:
        smiles: 分子 SMILES
        perturb: 扰动强度
        seed: 随机种子
        config: 配置字典
        output_dir: 输出目录
    
    Returns:
        results: 对比结果
    """
    if config is None:
        config = load_config()
    
    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./output/comparison_{timestamp}"
    
    output_manager = OutputManager(output_dir, format='json')
    
    print(f"\n{'='*70}")
    print("L-BFGS vs L-BFGS+GPR 对比实验")
    print(f"{'='*70}")
    print(f"分子：{smiles}")
    print(f"扰动：{perturb} Å")
    print(f"种子：{seed}")
    print(f"输出目录：{output_dir}")
    print(f"{'='*70}")
    
    # 创建分子（使用相同的初始结构）
    molecule = Molecule.from_smiles(smiles, seed=seed, perturb_strength=perturb)
    
    print(f"\n初始结构:")
    print(f"  原子数：{molecule.n_atoms}")
    print(f"  自由度：{molecule.n_atoms * 3}")
    
    # 创建计算器
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )
    
    # 存储结果
    histories = {}
    results = {}
    
    # ========== 运行 L-BFGS ==========
    print(f"\n{'='*70}")
    print("运行 L-BFGS (基准方法)")
    print(f"{'='*70}")
    
    lbfgs_optimizer = LBFGSOptimizer(config)
    lbfgs_history = lbfgs_optimizer.optimize(molecule, calculator)
    
    histories['L-BFGS'] = lbfgs_history
    results['L-BFGS'] = {
        'converged': lbfgs_history.converged,
        'best_energy': lbfgs_history.get_best_iteration('energy').energy,
        'best_gradient_norm': lbfgs_history.get_best_iteration('energy').gradient_norm,
        'iterations': len(lbfgs_history)
    }
    
    # 保存 L-BFGS 结果
    output_manager.save_history(lbfgs_history, 'lbfgs', {'method': 'L-BFGS'})
    output_manager.save_trajectory(lbfgs_history, 'lbfgs', molecule.atom_symbols)
    output_manager.save_iteration_details(lbfgs_history, 'lbfgs', molecule.atom_symbols)
    
    # ========== 运行 L-BFGS+GPR ==========
    print(f"\n{'='*70}")
    print("运行 L-BFGS+GPR (混合方法)")
    print(f"{'='*70}")
    
    # 重新创建分子（相同的初始结构）
    molecule_hybrid = Molecule.from_smiles(smiles, seed=seed, perturb_strength=perturb)
    
    hybrid_optimizer = HybridOptimizer(config)
    hybrid_history = hybrid_optimizer.optimize(molecule_hybrid, calculator)
    
    histories['L-BFGS+GPR'] = hybrid_history
    results['L-BFGS+GPR'] = {
        'converged': hybrid_history.converged,
        'best_energy': hybrid_history.get_best_iteration('energy').energy,
        'best_gradient_norm': hybrid_history.get_best_iteration('energy').gradient_norm,
        'iterations': len(hybrid_history)
    }
    
    # 保存混合方法结果
    output_manager.save_history(hybrid_history, 'hybrid', {'method': 'L-BFGS+GPR'})
    output_manager.save_trajectory(hybrid_history, 'hybrid', molecule.atom_symbols)
    output_manager.save_iteration_details(hybrid_history, 'hybrid', molecule.atom_symbols)
    
    # ========== 生成对比图表 ==========
    print(f"\n{'='*70}")
    print("生成对比图表")
    print(f"{'='*70}")
    
    vis_config = config.get('visualization', {})
    plotter = OptimizationPlotter(
        font_size=vis_config.get('font_size', 14),
        figure_size=tuple(vis_config.get('figure_size', [12, 8])),
        dpi=vis_config.get('dpi', 300)
    )
    
    plots_dir = os.path.join(output_dir, 'plots')
    
    # 能量对比
    plotter.plot_comparison(
        histories,
        title="L-BFGS vs L-BFGS+GPR - 能量对比",
        save_path=f"{plots_dir}/comparison_energy.png",
        plot_type='energy'
    )
    
    # 梯度对比
    plotter.plot_comparison(
        histories,
        title="L-BFGS vs L-BFGS+GPR - 梯度对比",
        save_path=f"{plots_dir}/comparison_gradient.png",
        plot_type='gradient'
    )
    
    # 组合对比
    plotter.plot_comparison(
        histories,
        title="L-BFGS vs L-BFGS+GPR - 完整对比",
        save_path=f"{plots_dir}/comparison_combined.png",
        plot_type='both'
    )
    
    # ========== 生成 3D 结构对比图 ==========
    vis = MoleculeVisualizer3D(
        font_size=vis_config.get('font_size', 14),
        figure_size=tuple(vis_config.get('figure_size', [10, 8])),
        dpi=vis_config.get('dpi', 300)
    )
    
    structures_dir = os.path.join(output_dir, 'structures')
    
    # 获取最优结构
    lbfgs_best = lbfgs_history.get_best_iteration('energy')
    hybrid_best = hybrid_history.get_best_iteration('energy')
    
    lbfgs_best_mol = Molecule(
        molecule.atom_symbols,
        lbfgs_best.coords.reshape(-1, 3),
        smiles,
        "L-BFGS Best"
    )
    
    hybrid_best_mol = Molecule(
        molecule.atom_symbols,
        hybrid_best.coords.reshape(-1, 3),
        smiles,
        "Hybrid Best"
    )
    
    # 三种方法对比
    vis.visualize_comparison(
        [molecule, lbfgs_best_mol, hybrid_best_mol],
        titles=[
            f'Initial\nE={calculator.calculate_energy(molecule.atom_symbols, molecule.coords):.6f}',
            f'L-BFGS\nE={lbfgs_best.energy:.6f}',
            f'Hybrid\nE={hybrid_best.energy:.6f}'
        ],
        save_path=f"{structures_dir}/all_methods_comparison.png"
    )
    
    # ========== 保存对比总结 ==========
    metadata = {
        'molecule': smiles,
        'perturb': perturb,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    output_manager.save_summary(histories, metadata)
    
    # ========== 打印结果 ==========
    print(f"\n{'='*70}")
    print("对比结果总结")
    print(f"{'='*70}")
    
    print(f"\n{'方法':<15} {'收敛':<8} {'最优能量':<20} {'梯度范数':<15} {'迭代次数':<10}")
    print(f"{'-'*70}")
    
    for method, result in results.items():
        converged_str = "是" if result['converged'] else "否"
        print(f"{method:<15} {converged_str:<8} {result['best_energy']:<20.10f} "
              f"{result['best_gradient_norm']:<15.6f} {result['iterations']:<10}")
    
    # 计算改进
    lbfgs_energy = results['L-BFGS']['best_energy']
    hybrid_energy = results['L-BFGS+GPR']['best_energy']
    energy_diff = hybrid_energy - lbfgs_energy
    
    print(f"\n能量差异：{energy_diff:.10f} Hartree")
    print(f"         = {energy_diff * 627.509:.6f} kcal/mol")
    
    if energy_diff < -0.001:
        print("结论：混合方法找到了更低的能量（更好的构型）")
    elif energy_diff > 0.001:
        print("结论：L-BFGS 找到了更低的能量")
    else:
        print("结论：两种方法找到了相似的能量")
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存至：{output_dir}")
    print(f"{'='*70}")
    
    return {
        'results': results,
        'histories': histories,
        'output_dir': output_dir
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='L-BFGS vs L-BFGS+GPR 对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--smiles', type=str, default='CCO',
                       help='分子 SMILES (default: CCO)')
    parser.add_argument('--perturb', type=float, default=0.0,
                       help='初始扰动强度 (Å) (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    results = run_comparison(
        smiles=args.smiles,
        perturb=args.perturb,
        seed=args.seed,
        config=config,
        output_dir=args.output
    )
    
    return results


if __name__ == '__main__':
    main()
