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

        # 初始化 AI 代理模型（支持多种方法）
        dim = molecule.n_atoms * 3
        ai_method = self.config.get('gpr', {}).get('type', 'simple')

        if ai_method == 'random_forest':
            from models.random_forest import RandomForestModel
            self.gpr_model = RandomForestModel(self.config, dim)
            self.ai_method_name = "Random Forest"
        elif ai_method == 'gradient_predicting':
            # 梯度预测 GPR：直接预测梯度向量
            from models.gradient_predicting_gpr import GradientPredictingGPR
            self.gpr_model = GradientPredictingGPR(self.config, dim)
            self.ai_method_name = "Gradient-Predicting GPR"
        elif ai_method == 'gradient':
            self.gpr_model = GradientGPRModel(self.config, dim)
            self.ai_method_name = "Gradient GPR"
        else:  # simple 或默认
            self.gpr_model = SimpleGPRModel(self.config, dim)
            self.ai_method_name = "Simple GPR"

        # 设置边界（基于初始坐标的局部区域）
        self._setup_bounds(molecule)
        self.gpr_model.set_bounds(self._bounds)

        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print(f"L-BFGS+{self.ai_method_name} 混合优化开始")
            print("=" * 70)
            print(f"L-BFGS 步数 (m): {self.lbfgs_steps}")
            print(f"{self.ai_method_name} 步数 (n): {self.gpr_steps}")
            print(f"选择标准：{self.selection_metric}")
            print(f"初始能量：{self._calculate_energy(molecule.get_coords_flat()):.10f} Hartree")
            print("=" * 70)
        
        self.history.start_time = time.time()
        
        # 初始采样：生成训练数据
        self._initial_sampling(molecule)
        
        # 从初始采样点中选择最优作为起点
        coords = self._get_best_from_initial_samples()

        # 主循环
        max_iterations = self.config.get('optimizer', {}).get('max_iterations', 200)
        convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-5)

        iteration = 0
        no_improvement_count = 0
        max_no_improvement = 20  # 连续多少步无改进则停止
        
        # 记录全局最优点（基于梯度范数，而不是能量！）
        global_best_coords = coords.copy()
        global_best_energy = self._calculate_energy(coords)
        global_best_grad_norm = np.linalg.norm(
            self.calculator.calculate_gradient(self.atom_symbols, coords.reshape(-1, 3))
        )

        # GPR 配置
        use_gpr = self.config.get('gpr', {}).get('use_gpr', True)

        # 早停参数
        max_no_improvement = 50  # 20→50，允许更多轮次的小改进

        while iteration < max_iterations:
            self.current_round += 1

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n{'='*70}")
                print(f"第 {self.current_round} 轮优化")
                print(f"{'='*70}")
                print(f"本轮起点：Energy = {global_best_energy:.10f} Hartree")

            # 收集本轮所有点（包括起点）
            round_iterations = []
            
            # 创建起点迭代数据
            start_energy = self._calculate_energy(coords)
            start_gradient = self.calculator.calculate_gradient(
                self.atom_symbols, coords.reshape(-1, 3)
            )
            start_data = IterationData(
                iteration=-1,
                energy=start_energy,
                gradient=start_gradient,
                coords=coords
            )
            round_iterations.append(start_data)

            # 执行 L-BFGS 步骤（m 步）
            lbfgs_history = self._run_lbfgs_steps(coords, self.lbfgs_steps)
            round_iterations.extend(lbfgs_history.iterations)

            # 执行 GPR 步骤（n 步）- 每轮都执行，验证 AI 方法可行性
            if use_gpr:
                last_lbfgs_coords = lbfgs_history.iterations[-1].coords if lbfgs_history.iterations else coords
                gpr_history = self._run_gpr_steps(last_lbfgs_coords, self.gpr_steps)
                round_iterations.extend(gpr_history.iterations)

                # 记录 GPR 表现（不跳过，只记录）
                if gpr_history.iterations:
                    best_gpr_energy = min(it.energy for it in gpr_history.iterations)
                    best_lbfgs_energy = min(it.energy for it in lbfgs_history.iterations)
                    if best_gpr_energy < best_lbfgs_energy - 1e-4:
                        if self.config.get('optimizer', {}).get('verbose', True):
                            print(f"{self.ai_method_name} 找到更优点：E={best_gpr_energy:.8f}")
                    else:
                        if self.config.get('optimizer', {}).get('verbose', True):
                            print(f"{self.ai_method_name} 未找到更优点（当前轮次）")

            # 从本轮所有点中选择最优（起点 + m + n 个点）
            # 核心修改：选择梯度范数最小的点，而不是能量最低的点！
            # 分子几何构型优化的目标是找到梯度为零的稳定构型
            best_data = min(round_iterations, key=lambda x: x.gradient_norm)
            best_coords = best_data.coords

            # 添加到总历史
            self.history.iterations.extend(round_iterations)
            iteration += len(round_iterations)

            # 更新全局最优（基于梯度范数，而不是能量）
            current_best_grad_norm = best_data.gradient_norm
            if current_best_grad_norm < global_best_grad_norm:
                global_best_grad_norm = current_best_grad_norm
                global_best_coords = best_coords.copy()
                global_best_energy = best_data.energy  # 同时记录能量

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n本轮最佳：Iter {best_data.iteration}, E={best_data.energy:.8f}, |g|={best_data.gradient_norm:.6f}")
                print(f"全局最佳（梯度最小）：E={global_best_energy:.10f} Hartree, |g|={global_best_grad_norm:.6f}")

            # 每轮结束后统一训练 GPR（只训练 1 次）并应用滑动窗口
            X, y, gradients = self.gpr_model.get_training_data()
            if len(X) > 3:
                # 应用滑动窗口，只保留能量最好的 50% 的点（最多 max_training_points）
                self.gpr_model.limit_training_data_by_percentile(50.0)
                # 重新训练 GPR
                X, y, gradients = self.gpr_model.get_training_data()
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"GPR 训练点数：{len(X)}")
                self.gpr_model.train(X, y, gradients)

            # 下一轮从全局最优点开始
            coords = global_best_coords.copy()

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

            # 检查是否无改进（基于梯度范数，而不是能量）
            # 梯度范数变化小于 1e-6 认为无改进
            if len(self.history) > 1:
                prev_best = self.history.get_best_iteration('gradient')
                if prev_best is not None:
                    # 梯度改进阈值：1e-6
                    grad_diff = abs(current_best_grad_norm - prev_best.gradient_norm)
                    if grad_diff < 1e-6:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                        # 打印当前改进，让用户知道优化仍在进行
                        if grad_diff < 1e-3:  # 小改进也显示
                            if self.config.get('optimizer', {}).get('verbose', True):
                                print(f"梯度改进：Δ|g|={grad_diff:.2e}")

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
        """
        初始采样生成 GPR 训练数据
        
        使用 L-BFGS 迭代 k 次（默认 10 次）生成初始采样点，
        而不是随机扰动，以提供更高质量的训练数据
        """
        n_init = self.config.get('gpr', {}).get('n_init', 10)

        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n使用 L-BFGS 生成 {n_init} 个初始采样点...")

        coords = molecule.get_coords_flat()
        
        # 使用 L-BFGS 迭代生成初始样本点
        from scipy.optimize import minimize
        
        # 创建能量/梯度函数
        energy_func = lambda x: self.calculator.calculate_energy(self.atom_symbols, x.reshape(-1, 3))
        gradient_func = lambda x: self.calculator.calculate_gradient(self.atom_symbols, x.reshape(-1, 3))
        
        # 用于存储 L-BFGS 迭代点
        collected_points = []
        
        def callback(xk):
            """回调函数：收集 L-BFGS 迭代点"""
            energy = energy_func(xk)
            gradient = gradient_func(xk)
            collected_points.append((xk.copy(), energy, gradient.copy()))
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"  Init {len(collected_points)-1}: Energy = {energy:.8f} Hartree, "
                      f"|grad| = {np.linalg.norm(gradient):.6f}")
            
            # 收集到足够的点后停止
            if len(collected_points) >= n_init:
                pass  # 继续让 L-BFGS 完成当前迭代
        
        # 运行 L-BFGS 收集初始点
        result = minimize(
            fun=energy_func,
            x0=coords,
            method='L-BFGS-B',
            jac=gradient_func,
            callback=callback,
            options={'maxiter': n_init - 1, 'gtol': 1e-10, 'disp': False}
        )
        
        # 确保至少有 n_init 个点（包括初始点）
        # 如果 L-BFGS 提前收敛，添加一些扰动点
        if len(collected_points) < n_init:
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\nL-BFGS 提前收敛，补充 {n_init - len(collected_points)} 个扰动点...")
            
            base_coords = collected_points[-1][0] if collected_points else coords
            for i in range(len(collected_points), n_init):
                np.random.seed(i + 42)
                perturbation = np.random.uniform(-0.1, 0.1, size=base_coords.shape)
                sampled_coords = base_coords + perturbation
                
                energy = energy_func(sampled_coords)
                gradient = gradient_func(sampled_coords)
                collected_points.append((sampled_coords, energy, gradient.copy()))
                
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"  Init {i}: Energy = {energy:.8f} Hartree, "
                          f"|grad| = {np.linalg.norm(gradient):.6f} (perturbed)")

        # 将所有点添加到 GPR 训练集
        for sampled_coords, energy, gradient in collected_points:
            self.gpr_model.add_data(sampled_coords, energy, gradient)

        # 训练 GPR 模型
        X, y, gradients = self.gpr_model.get_training_data()
        self.gpr_model.train(X, y, gradients)

        if self.config.get('optimizer', {}).get('verbose', True):
            print("GPR 模型训练完成")

    def _get_best_from_initial_samples(self) -> np.ndarray:
        """
        从初始采样点中选择能量最低的作为主循环起点
        
        Returns:
            best_coords: 最优坐标
        """
        if not self.gpr_model.X_train:
            return self.current_mol.get_coords_flat()
        
        # 找到能量最低的点
        best_idx = np.argmin(self.gpr_model.y_train)
        best_coords = self.gpr_model.X_train[best_idx]
        best_energy = self.gpr_model.y_train[best_idx]
        
        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n初始采样最优：Energy = {best_energy:.10f} Hartree (Init {best_idx})")
        
        return best_coords

    def _run_lbfgs_steps(self, coords: np.ndarray, n_steps: int) -> OptimizationHistory:
        """
        执行固定步数的 L-BFGS 优化（使用 callback 机制，保持 Hessian 连续性）

        Args:
            coords: 初始坐标
            n_steps: 步数

        Returns:
            history: 优化历史
        """
        history = OptimizationHistory()
        calculator = self.calculator
        atom_symbols = self.atom_symbols
        
        # 用于记录步数
        step_count = [0]
        prev_coords = [coords.copy()]

        # 创建能量/梯度函数（闭包引用）
        def energy_only(x):
            return calculator.calculate_energy(atom_symbols, x.reshape(-1, 3))
        
        def gradient_only(x):
            return calculator.calculate_gradient(atom_symbols, x.reshape(-1, 3))

        def callback(xk):
            """scipy 的 callback，在每步后调用"""
            energy = energy_only(xk)
            gradient = gradient_only(xk)
            gradient_norm = np.linalg.norm(gradient)
            displacement = np.linalg.norm(xk - prev_coords[0])
            
            # 记录迭代数据
            data = IterationData(
                iteration=step_count[0],
                energy=energy,
                gradient=gradient,
                coords=xk.copy(),
                displacement=prev_coords[0] - xk
            )
            history.add_iteration(data)
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"LBFGS {step_count[0]}: E={energy:.8f}, |g|={gradient_norm:.6f}, d={displacement:.6f}")
            
            # 更新上一步坐标
            prev_coords[0] = xk.copy()
            step_count[0] += 1
            
            # 添加到 GPR 训练集（每步都加）
            self.gpr_model.add_data(xk.copy(), energy, gradient.copy())
            
            # 如果达到步数上限，无法直接停止，但可以通过标记让外部知道
            if step_count[0] >= n_steps:
                pass  # 继续让 scipy 完成当前迭代

        # 执行 L-BFGS 优化（连续运行 n_steps 步）
        from scipy.optimize import minimize
        result = minimize(
            fun=energy_only,
            x0=coords,
            method='L-BFGS-B',
            jac=gradient_only,
            callback=callback,
            options={
                'maxiter': n_steps,  # 连续运行 n_steps
                'gtol': 1e-10,
                'disp': False
            }
        )
        
        # 注意：不在这里训练 GPR，等到轮结束后统一训练

        return history
    
    def _run_gpr_steps(self, coords: np.ndarray, n_steps: int) -> OptimizationHistory:
        """
        执行 GPR 预测步骤

        Args:
            coords: 初始坐标
            n_steps: 步数

        Returns:
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

        return history
    
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
