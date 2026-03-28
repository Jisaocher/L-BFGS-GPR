"""
L-BFGS+GPR 混合优化策略
每 m 步 L-BFGS 后加上 n 步 GPR 预测，选择表现最好的迭代结果
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time

from core.molecule import Molecule, OptimizationHistory, IterationData
from core.calculator import QuantumCalculator
from optimizers.base import BaseOptimizer
from optimizers.lbfgs import LBFGSOptimizer
from models.gradient_gpr import GradientGPRModel, SimpleGPRModel


class HybridOptimizer(BaseOptimizer):
    """
    L-BFGS+GPR 混合优化器
    
    策略：
    1. 执行 m 步 L-BFGS 优化
    2. 执行 n 步 GPR 预测（基于采集函数）
    3. 从这 m+n 步中选择表现最好的点
    4. 循环直至收敛
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化混合优化器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.name = "L-BFGS+GPR Hybrid"
        
        # 混合策略参数
        hybrid_config = config.get('hybrid', {})
        self.lbfgs_steps = hybrid_config.get('lbfgs_steps', 5)  # m
        self.gpr_steps = hybrid_config.get('gpr_steps', 2)      # n
        self.selection_metric = hybrid_config.get('selection_metric', 'combined')
        self.verify_local_minimum = hybrid_config.get('verify_local_minimum', True)
        self.verify_extra_steps = hybrid_config.get('verify_extra_steps', 3)
        
        # 组件
        self.lbfgs_optimizer = None
        self.gpr_model = None
        self.calculator = None
        
        # 状态
        self.current_round = 0
        self._bounds = None
    
    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行混合优化
        
        Args:
            molecule: 初始分子
            calculator: 量子化学计算器
        
        Returns:
            OptimizationHistory: 优化历史
        """
        self.current_mol = molecule
        self.calculator = calculator
        self.atom_symbols = molecule.atom_symbols
        self.history = OptimizationHistory()
        
        # 初始化 L-BFGS 优化器
        self.lbfgs_optimizer = LBFGSOptimizer(self.config)
        self.lbfgs_optimizer.current_mol = molecule
        self.lbfgs_optimizer.atom_symbols = molecule.atom_symbols
        self.lbfgs_optimizer.calculator = calculator
        
        # 初始化 GPR 模型
        dim = molecule.n_atoms * 3
        gpr_type = self.config.get('gpr', {}).get('type', 'gradient')
        if gpr_type == 'gradient':
            self.gpr_model = GradientGPRModel(self.config, dim)
        else:
            self.gpr_model = SimpleGPRModel(self.config, dim)
        
        # 设置边界（基于初始坐标的局部区域）
        self._setup_bounds(molecule)
        self.gpr_model.set_bounds(self._bounds)
        
        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("L-BFGS+GPR 混合优化开始")
            print("=" * 70)
            print(f"L-BFGS 步数 (m): {self.lbfgs_steps}")
            print(f"GPR 步数 (n): {self.gpr_steps}")
            print(f"选择标准：{self.selection_metric}")
            print(f"初始能量：{self._calculate_energy(molecule.get_coords_flat()):.10f} Hartree")
            print("=" * 70)
        
        self.history.start_time = time.time()
        
        # 初始采样：生成一些训练数据
        self._initial_sampling(molecule)
        
        # 主循环
        coords = molecule.get_coords_flat()
        max_iterations = self.config.get('optimizer', {}).get('max_iterations', 200)
        convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-5)
        
        iteration = 0
        no_improvement_count = 0
        max_no_improvement = 20  # 连续多少步无改进则停止
        
        while iteration < max_iterations:
            self.current_round += 1
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n{'='*70}")
                print(f"第 {self.current_round} 轮优化")
                print(f"{'='*70}")
            
            # 执行 L-BFGS 步骤
            lbfgs_coords, lbfgs_history = self._run_lbfgs_steps(coords, self.lbfgs_steps)
            self.history.iterations.extend(lbfgs_history.iterations)
            iteration += len(lbfgs_history)
            
            # 检查 L-BFGS 结果是否收敛
            if lbfgs_history.converged:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print("L-BFGS 已收敛，验证局部极小值...")
                
                if self.verify_local_minimum:
                    verified = self._verify_minimum(lbfgs_coords, convergence_threshold)
                    if verified:
                        self.history.converged = True
                        self.history.convergence_iteration = len(self.history) - 1
                        break
            
            # 执行 GPR 步骤
            gpr_coords, gpr_history = self._run_gpr_steps(lbfgs_coords, self.gpr_steps)
            self.history.iterations.extend(gpr_history.iterations)
            iteration += len(gpr_history)
            
            # 选择最佳点
            best_coords = self._select_best_in_round(lbfgs_history, gpr_history)
            
            if best_coords is not None:
                coords = best_coords
            else:
                coords = gpr_coords
            
            # 检查收敛
            current_gradient = self.calculator.calculate_gradient(
                self.atom_symbols, coords.reshape(-1, 3)
            )
            gradient_norm = np.linalg.norm(current_gradient)
            
            if gradient_norm < convergence_threshold:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n收敛！梯度范数：{gradient_norm:.6f}")
                break
            
            # 检查是否无改进
            if len(self.history) > 1:
                prev_best = self.history.get_best_iteration('energy')
                if prev_best is not None:
                    current_energy = self._calculate_energy(coords)
                    if abs(current_energy - prev_best.energy) < 1e-6:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
            
            if no_improvement_count >= max_no_improvement:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n早停：连续{max_no_improvement}步无显著改进")
                break
        
        self.history.end_time = time.time()
        
        # 打印结束信息
        if self.config.get('optimizer', {}).get('verbose', True):
            self._print_summary()
        
        return self.history
    
    def _setup_bounds(self, molecule: Molecule) -> None:
        """设置优化边界"""
        coords = molecule.coords
        radius = self.config.get('gpr', {}).get('local_radius', 0.5)
        
        self._bounds = []
        for i in range(molecule.n_atoms):
            for j in range(3):
                low = coords[i, j] - radius
                high = coords[i, j] + radius
                self._bounds.append((low, high))
    
    def _initial_sampling(self, molecule: Molecule) -> None:
        """初始采样生成 GPR 训练数据"""
        n_init = self.config.get('gpr', {}).get('n_init', 10)
        
        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n生成 {n_init} 个初始采样点...")
        
        coords = molecule.coords.copy()
        
        for i in range(n_init):
            # 在局部区域内随机采样
            if i == 0:
                # 第一个点使用初始坐标
                sampled_coords = coords.copy()
            else:
                # 其他点添加随机扰动
                np.random.seed(i + 42)
                perturbation = np.random.uniform(-0.3, 0.3, size=coords.shape)
                sampled_coords = coords + perturbation
            
            # 计算能量和梯度
            energy, gradient = self.calculator.calculate_energy_gradient(
                self.atom_symbols, sampled_coords
            )
            
            # 添加到 GPR 训练集
            self.gpr_model.add_data(
                sampled_coords.flatten(),
                energy,
                gradient
            )
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"  Init {i}: Energy = {energy:.8f} Hartree, "
                      f"|grad| = {np.linalg.norm(gradient):.6f}")
        
        # 训练 GPR 模型
        X, y, gradients = self.gpr_model.get_training_data()
        self.gpr_model.train(X, y, gradients)
        
        if self.config.get('optimizer', {}).get('verbose', True):
            print("GPR 模型训练完成")
    
    def _run_lbfgs_steps(self, coords: np.ndarray, n_steps: int) -> Tuple[np.ndarray, OptimizationHistory]:
        """
        执行固定步数的 L-BFGS 优化
        
        Args:
            coords: 初始坐标
            n_steps: 步数
        
        Returns:
            final_coords: 最终坐标
            history: 优化历史
        """
        # 创建临时的 L-BFGS 优化器
        temp_lbfgs = LBFGSOptimizer(self.config)
        temp_lbfgs.current_mol = self.current_mol
        temp_lbfgs.atom_symbols = self.atom_symbols
        temp_lbfgs.calculator = self.calculator
        temp_lbfgs.history = OptimizationHistory()
        temp_lbfgs._energy_func = temp_lbfgs._energy_func = \
            type('obj', (object,), {
                'energy_only': lambda self, x: self.calculator.calculate_energy(
                    self.atom_symbols, x.reshape(-1, 3)
                ),
                'gradient_only': lambda self, x: self.calculator.calculate_gradient(
                    self.atom_symbols, x.reshape(-1, 3)
                ),
                'call_count': 0
            })()
        
        from scipy.optimize import minimize
        
        step_count = [0]
        prev_coords = [coords.copy()]
        
        def callback(xk):
            energy, gradient = temp_lbfgs._energy_func.energy_only(xk), \
                              temp_lbfgs._energy_func.gradient_only(xk)
            gradient_norm = np.linalg.norm(gradient)
            
            displacement = np.linalg.norm(xk - prev_coords[0])
            
            data = temp_lbfgs.get_iteration_data(
                iteration=len(temp_lbfgs.history),
                energy=energy,
                gradient=gradient,
                coords=xk,
                prev_coords=prev_coords[0]
            )
            temp_lbfgs.history.add_iteration(data)
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"LBFGS {len(temp_lbfgs.history)-1}: E={energy:.8f}, |g|={gradient_norm:.6f}, d={displacement:.6f}")
            
            prev_coords[0] = xk.copy()
            step_count[0] += 1
            
            if step_count[0] >= n_steps:
                # 通过更新 GPR 训练数据
                self.gpr_model.add_data(xk, energy, gradient)
        
        result = minimize(
            fun=temp_lbfgs._energy_func.energy_only,
            x0=coords,
            method='L-BFGS-B',
            jac=temp_lbfgs._energy_func.gradient_only,
            callback=callback,
            options={'maxiter': n_steps, 'gtol': 1e-6, 'disp': False}
        )
        
        # 重新训练 GPR 模型
        X, y, gradients = self.gpr_model.get_training_data()
        if len(X) > 2:
            self.gpr_model.train(X, y, gradients)
        
        return result.x, temp_lbfgs.history
    
    def _run_gpr_steps(self, coords: np.ndarray, n_steps: int) -> Tuple[np.ndarray, OptimizationHistory]:
        """
        执行 GPR 预测步骤
        
        Args:
            coords: 初始坐标
            n_steps: 步数
        
        Returns:
            final_coords: 最终坐标
            history: 优化历史
        """
        history = OptimizationHistory()
        current_coords = coords.copy()
        y_min = min(self.gpr_model.y_train) if self.gpr_model.y_train else 0
        
        for i in range(n_steps):
            # 使用采集函数建议下一个点
            next_coords = self.gpr_model.suggest_next_point(self._bounds, y_min)
            
            # 计算真实能量和梯度
            energy, gradient = self.calculator.calculate_energy_gradient(
                self.atom_symbols, next_coords.reshape(-1, 3)
            )
            gradient_norm = np.linalg.norm(gradient)
            
            # 记录
            displacement = np.linalg.norm(next_coords - current_coords)
            data = self.get_iteration_data(
                iteration=len(history),
                energy=energy,
                gradient=gradient,
                coords=next_coords,
                prev_coords=current_coords
            )
            history.add_iteration(data)
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"GPR   {len(history)-1}: E={energy:.8f}, |g|={gradient_norm:.6f}, d={displacement:.6f}")
            
            # 添加到 GPR 训练集
            self.gpr_model.add_data(next_coords, energy, gradient)
            
            # 更新
            current_coords = next_coords
            y_min = min(y_min, energy)
        
        # 重新训练 GPR 模型
        X, y, gradients = self.gpr_model.get_training_data()
        if len(X) > 2:
            self.gpr_model.train(X, y, gradients)
        
        return current_coords, history
    
    def _select_best_in_round(self, lbfgs_history: OptimizationHistory,
                              gpr_history: OptimizationHistory) -> Optional[np.ndarray]:
        """
        从一轮中选择最佳点
        
        Args:
            lbfgs_history: L-BFGS 历史
            gpr_history: GPR 历史
        
        Returns:
            best_coords: 最佳坐标
        """
        if self.selection_metric == 'energy':
            metric = 'energy'
        elif self.selection_metric == 'gradient':
            metric = 'gradient'
        else:  # combined
            metric = 'combined'
        
        # 合并所有迭代
        all_iterations = lbfgs_history.iterations + gpr_history.iterations
        
        if not all_iterations:
            return None
        
        # 找到最佳迭代
        if metric == 'energy':
            best_idx = np.argmin([it.energy for it in all_iterations])
        elif metric == 'gradient':
            best_idx = np.argmin([it.gradient_norm for it in all_iterations])
        else:  # combined
            scores = [it.energy + 0.1 * it.gradient_norm for it in all_iterations]
            best_idx = np.argmin(scores)
        
        best = all_iterations[best_idx]
        
        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n本轮最佳：Iter {best.iteration}, E={best.energy:.8f}, |g|={best.gradient_norm:.6f}")
        
        return best.coords
    
    def _verify_minimum(self, coords: np.ndarray, threshold: float) -> bool:
        """
        验证是否为真正的局部极小值
        
        Args:
            coords: 坐标
            threshold: 收敛阈值
        
        Returns:
            is_minimum: 是否为局部极小值
        """
        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"执行 {self.verify_extra_steps} 步额外验证...")
        
        # 向前探索几步
        for i in range(self.verify_extra_steps):
            # 小扰动
            perturbation = np.random.uniform(-0.1, 0.1, size=coords.shape)
            perturbed_coords = coords + perturbation
            
            energy, gradient = self.calculator.calculate_energy_gradient(
                self.atom_symbols, perturbed_coords.reshape(-1, 3)
            )
            gradient_norm = np.linalg.norm(gradient)
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"  验证 {i+1}: E={energy:.8f}, |g|={gradient_norm:.6f}")
            
            # 如果扰动后梯度更小，说明可能不是真正的极小值
            if gradient_norm < threshold * 0.5:
                return False
        
        return True
    
    def _calculate_energy(self, coords_flat: np.ndarray) -> float:
        """计算能量"""
        return self.calculator.calculate_energy(
            self.atom_symbols, coords_flat.reshape(-1, 3)
        )
    
    def _print_summary(self) -> None:
        """打印优化总结"""
        print("\n" + "=" * 70)
        print("优化完成！")
        print("=" * 70)
        
        best = self.history.get_best_iteration('energy')
        if best:
            print(f"最优能量：{best.energy:.10f} Hartree")
            print(f"最优梯度范数：{best.gradient_norm:.6f}")
            print(f"最优迭代：{best.iteration}")
        
        print(f"总迭代次数：{len(self.history)}")
        print(f"收敛状态：{'收敛' if self.history.converged else '未收敛'}")
        print(f"计算时间：{self.history.end_time - self.history.start_time:.2f} 秒")
        print(f"GPR 训练点数：{self.gpr_model.n_training_points()}")
        print("=" * 70)
    
    def step(self, coords_flat: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """单步优化（用于兼容基类接口）"""
        # 混合优化器不直接支持单步，需要通过 optimize 方法
        raise NotImplementedError("HybridOptimizer 需要使用 optimize() 方法")


def run_hybrid_optimization(molecule: Molecule, config: Dict[str, Any]) -> OptimizationHistory:
    """
    便捷函数：运行混合优化
    
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
    
    optimizer = HybridOptimizer(config)
    return optimizer.optimize(molecule, calculator)
