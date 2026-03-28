#!/usr/bin/env python3
"""
测试脚本
验证项目基本功能
"""
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.molecule import Molecule, IterationData, OptimizationHistory
from core.calculator import QuantumCalculator


def test_molecule():
    """测试分子类"""
    print("\n" + "="*50)
    print("测试 Molecule 类")
    print("="*50)
    
    # 从 SMILES 创建分子
    mol = Molecule.from_smiles('CCO', seed=42, perturb_strength=0.0)
    
    print(f"分子：{mol}")
    print(f"原子符号：{mol.atom_symbols}")
    print(f"原子数：{mol.n_atoms}")
    print(f"坐标形状：{mol.coords.shape}")
    print(f"初始能量坐标：\n{mol.coords}")
    
    # 测试坐标展平
    flat = mol.get_coords_flat()
    print(f"展平坐标形状：{flat.shape}")
    
    # 测试复制
    mol_copy = mol.copy()
    assert np.allclose(mol.coords, mol_copy.coords)
    print("复制测试：通过")
    
    # 测试 RMSD
    rmsd = mol.get_rmsd(mol_copy)
    assert rmsd < 1e-10
    print(f"RMSD 测试：通过 (RMSD={rmsd})")
    
    print("Molecule 类测试：通过 ✓")
    return True


def test_iteration_data():
    """测试迭代数据类"""
    print("\n" + "="*50)
    print("测试 IterationData 类")
    print("="*50)
    
    coords = np.random.randn(27)
    gradient = np.random.randn(27)
    
    data = IterationData(
        iteration=1,
        energy=-154.0,
        gradient=gradient,
        coords=coords,
        displacement=np.random.randn(27) * 0.1
    )
    
    print(f"迭代：{data.iteration}")
    print(f"能量：{data.energy}")
    print(f"梯度范数：{data.gradient_norm:.6f}")
    
    # 测试字典转换
    data_dict = data.to_dict()
    data_restored = IterationData.from_dict(data_dict)
    
    assert data_restored.iteration == data.iteration
    assert np.allclose(data_restored.coords, data.coords)
    print("字典转换测试：通过")
    
    print("IterationData 类测试：通过 ✓")
    return True


def test_optimization_history():
    """测试优化历史类"""
    print("\n" + "="*50)
    print("测试 OptimizationHistory 类")
    print("="*50)
    
    history = OptimizationHistory()
    
    # 添加一些模拟迭代
    for i in range(5):
        coords = np.random.randn(27)
        gradient = np.random.randn(27) * (1.0 / (i + 1))  # 梯度逐渐减小
        
        data = IterationData(
            iteration=i,
            energy=-154.0 - i * 0.01,  # 能量逐渐降低
            gradient=gradient,
            coords=coords
        )
        history.add_iteration(data)
    
    print(f"迭代次数：{len(history)}")
    print(f"能量历史：{history.get_energies()}")
    print(f"梯度范数历史：{history.get_gradient_norms()}")
    
    # 测试最优迭代
    best = history.get_best_iteration('energy')
    print(f"最优能量：{best.energy}")
    
    # 测试收敛检查
    history.converged = True
    history.convergence_iteration = 4
    print(f"收敛状态：{history.converged}")
    
    # 测试保存/加载
    history.save_json('/tmp/test_history.json')
    history_loaded = OptimizationHistory.from_json('/tmp/test_history.json')
    
    assert len(history_loaded) == len(history)
    print("保存/加载测试：通过")
    
    print("OptimizationHistory 类测试：通过 ✓")
    return True


def test_calculator():
    """测试量子化学计算器"""
    print("\n" + "="*50)
    print("测试 QuantumCalculator 类")
    print("="*50)
    
    # 创建小分子测试（水分子）
    mol = Molecule.from_smiles('O', seed=42)
    
    print(f"测试分子：{mol.atom_symbols}")
    print(f"坐标:\n{mol.coords}")
    
    # 创建计算器
    calculator = QuantumCalculator(basis='sto-3g', method='RHF', verbose=0)
    
    # 计算能量和梯度
    energy, gradient = calculator.calculate_energy_gradient(
        mol.atom_symbols, mol.coords
    )
    
    print(f"能量：{energy:.6f} Hartree")
    print(f"梯度范数：{np.linalg.norm(gradient):.6f}")
    
    # 验证梯度形状
    assert gradient.shape == (mol.n_atoms * 3,)
    print("梯度形状测试：通过")
    
    print("QuantumCalculator 类测试：通过 ✓")
    return True


def test_optimizers():
    """测试优化器"""
    print("\n" + "="*50)
    print("测试优化器")
    print("="*50)
    
    from optimizers.lbfgs import LBFGSOptimizer
    
    # 创建测试分子
    mol = Molecule.from_smiles('O', seed=42)
    
    # 配置
    config = {
        'optimizer': {
            'max_iterations': 50,
            'convergence_threshold': 1e-4,
            'verbose': False
        },
        'lbfgs': {
            'maxiter': 50,
            'gtol': 1e-4
        },
        'calculation': {
            'basis': 'sto-3g',
            'method': 'RHF'
        }
    }
    
    # 创建计算器
    calculator = QuantumCalculator(
        basis='sto-3g', method='RHF', verbose=0
    )
    
    # 运行 L-BFGS
    optimizer = LBFGSOptimizer(config)
    history = optimizer.optimize(mol, calculator)
    
    print(f"L-BFGS 迭代次数：{len(history)}")
    print(f"L-BFGS 收敛：{history.converged}")
    
    if len(history) > 0:
        best = history.get_best_iteration('energy')
        print(f"最优能量：{best.energy:.6f}")
        print(f"最优梯度范数：{best.gradient_norm:.6f}")
    
    print("优化器测试：通过 ✓")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("L-BFGS-GPR 项目测试套件")
    print("="*60)
    
    tests = [
        ("Molecule", test_molecule),
        ("IterationData", test_iteration_data),
        ("OptimizationHistory", test_optimization_history),
        ("QuantumCalculator", test_calculator),
        ("Optimizers", test_optimizers)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n{name} 测试失败：{e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计：{passed}/{total} 通过")
    
    if passed == total:
        print("\n所有测试通过！✓")
        return True
    else:
        print("\n部分测试失败！✗")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
