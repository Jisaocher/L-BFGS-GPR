"""
L-BFGS 优化器实现
使用 scipy.optimize.minimize 的 L-BFGS-B 方法
"""
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional, Tuple
import time

from core.molecule import Molecule, OptimizationHistory, IterationData
from core.calculator import QuantumCalculator, EnergyGradientFunction
from optimizers.base import BaseOptimizer


class LBFGSOptimizer(BaseOptimizer):
    """
    L-BFGS 优化器
    
    基于 scipy.optimize.minimize 的 L-BFGS-B 实现
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 L-BFGS 优化器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.name = "L-BFGS"
        
        # L-BFGS 特定参数
        lbfgs_config = config.get('lbfgs', {})
        self.maxiter = lbfgs_config.get('maxiter', 100)
        self.gtol = lbfgs_config.get('gtol', 1e-5)
        self.memory = lbfgs_config.get('memory', 10)
        
        # 内部状态
        self._energy_func = None
        self._prev_coords = None
    
    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行完整的 L-BFGS 优化
        
        Args:
            molecule: 初始分子结构
            calculator: 量子化学计算器
        
        Returns:
            OptimizationHistory: 优化历史
        """
        self.current_mol = molecule
        self.atom_symbols = molecule.atom_symbols
        self.calculator = calculator
        self.history = OptimizationHistory()
        
        # 初始化
        x0 = molecule.get_coords_flat()
        self._prev_coords = x0.copy()
        
        # 创建能量/梯度函数
        self._energy_func = EnergyGradientFunction(calculator, molecule.atom_symbols)
        
        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("L-BFGS 优化开始")
            print("=" * 70)
            print(f"初始能量：{self._energy_func.energy_only(x0):.10f} Hartree")
            print(f"原子数：{molecule.n_atoms}")
            print(f"自由度：{len(x0)}")
            print("=" * 70)
        
        self.history.start_time = time.time()
        
        # 执行优化
        result = minimize(
            fun=self._energy_func.energy_only,
            x0=x0,
            method='L-BFGS-B',
            jac=self._energy_func.gradient_only,
            callback=self._callback,
            options={
                'maxiter': self.maxiter,
                'gtol': self.gtol,
                'disp': False
            }
        )
        
        self.history.end_time = time.time()

        # 检查收敛
        final_grad_norm = np.linalg.norm(self._energy_func.gradient_only(result.x))
        self.history.converged = final_grad_norm < self.gtol
        if self.history.converged:
            self.history.convergence_iteration = len(self.history) - 1

        # 打印结束信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("优化完成！")
            print(f"最终能量：{result.fun:.10f} Hartree")
            print(f"最终梯度范数：{final_grad_norm:.6f}")
            print(f"迭代次数：{len(self.history)}")
            print(f"收敛状态：{'是' if self.history.converged else '否'}")
            print(f"计算时间：{self.history.end_time - self.history.start_time:.2f} 秒")
            print(f"能量/梯度调用次数：{self._energy_func.call_count}")
            print(f"优化终止原因：{result.message}")
            print(f"优化成功标志：{result.success}")
            print("=" * 70)

        return self.history
    
    def _callback(self, xk: np.ndarray) -> None:
        """
        scipy.optimize.minimize 的回调函数
        
        Args:
            xk: 当前坐标
        """
        # 计算当前能量和梯度
        energy, gradient = self._energy_func(xk)
        gradient_norm = np.linalg.norm(gradient)
        
        # 计算位移
        displacement = None
        if self._prev_coords is not None:
            displacement = np.linalg.norm(xk - self._prev_coords)
        
        # 创建迭代数据
        iteration = len(self.history)
        data = self.get_iteration_data(
            iteration=iteration,
            energy=energy,
            gradient=gradient,
            coords=xk,
            prev_coords=self._prev_coords
        )
        self.history.add_iteration(data)
        
        # 打印信息
        self.print_iteration(iteration, energy, gradient_norm, displacement)
        
        # 更新上一步坐标
        self._prev_coords = xk.copy()
    
    def step(self, coords_flat: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        执行单步 L-BFGS 优化
        
        这个方法用于混合优化器中，执行一步 L-BFGS
        
        Args:
            coords_flat: 当前坐标 (展平)
        
        Returns:
            new_coords: 新坐标
            energy: 能量
            gradient: 梯度
        """
        # 对于 L-BFGS，单步优化需要通过 scipy 的内部机制
        # 这里我们使用简化的方法：执行一次线搜索
        
        if self._energy_func is None:
            raise ValueError("需要先调用 optimize() 初始化")
        
        # 计算当前能量和梯度
        energy = self._energy_func.energy_only(coords_flat)
        gradient = self._energy_func.gradient_only(coords_flat)
        
        # 使用 BFGS 公式计算搜索方向
        # 这里简化处理，使用负梯度方向
        search_direction = -gradient
        
        # 简单的回溯线搜索
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        max_ls_iter = 20
        
        for _ in range(max_ls_iter):
            new_coords = coords_flat + alpha * search_direction
            
            # 确保坐标合理（防止原子太近）
            new_coords_reshaped = new_coords.reshape(-1, 3)
            min_dist = self._check_min_distance(new_coords_reshaped)
            
            if min_dist < 0.5:  # 原子间最小距离
                alpha *= rho
                continue
            
            new_energy = self._energy_func.energy_only(new_coords)
            
            # Armijo 条件
            if new_energy < energy + c1 * alpha * np.dot(gradient, search_direction):
                return new_coords, new_energy, self._energy_func.gradient_only(new_coords)
            
            alpha *= rho
        
        # 线搜索失败，返回小步长结果
        new_coords = coords_flat + 0.01 * search_direction
        new_energy = self._energy_func.energy_only(new_coords)
        new_gradient = self._energy_func.gradient_only(new_coords)
        return new_coords, new_energy, new_gradient
    
    def _check_min_distance(self, coords: np.ndarray) -> float:
        """检查最小原子间距离"""
        n_atoms = coords.shape[0]
        min_dist = float('inf')
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def run_fixed_steps(self, coords_flat: np.ndarray, n_steps: int,
                        calculator: QuantumCalculator) -> Tuple[np.ndarray, OptimizationHistory]:
        """
        执行固定步数的 L-BFGS 优化
        
        用于混合优化策略
        
        Args:
            coords_flat: 初始坐标
            n_steps: 步数
            calculator: 量子化学计算器
        
        Returns:
            final_coords: 最终坐标
            history: 这段优化的历史
        """
        self.current_mol = Molecule(['C'], coords_flat.reshape(-1, 3)[:1])  # 临时分子
        self.atom_symbols = ['C'] * (len(coords_flat) // 3)  # 假设
        self.calculator = calculator
        self.history = OptimizationHistory()
        
        self._energy_func = EnergyGradientFunction(calculator, self.atom_symbols)
        self._prev_coords = coords_flat.copy()
        
        # 使用 scipy 的 minimize 但限制步数
        # 通过 callback 控制步数
        self._step_count = 0
        self._max_steps = n_steps
        self._stop_optimization = False
        
        def limited_callback(xk):
            self._callback(xk)
            self._step_count += 1
            if self._step_count >= n_steps:
                self._stop_optimization = True
        
        # 执行优化（通过异常中断）
        try:
            result = minimize(
                fun=self._energy_func.energy_only,
                x0=coords_flat,
                method='L-BFGS-B',
                jac=self._energy_func.gradient_only,
                callback=limited_callback,
                options={
                    'maxiter': n_steps,
                    'gtol': self.gtol,
                    'disp': False
                }
            )
            final_coords = result.x
        except:
            final_coords = self._prev_coords
        
        return final_coords, self.history


def run_lbfgs_optimization(molecule: Molecule, config: Dict[str, Any]) -> OptimizationHistory:
    """
    便捷函数：运行 L-BFGS 优化
    
    Args:
        molecule: 初始分子
        config: 配置字典
    
    Returns:
        OptimizationHistory: 优化历史
    """
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )
    
    optimizer = LBFGSOptimizer(config)
    return optimizer.optimize(molecule, calculator)
