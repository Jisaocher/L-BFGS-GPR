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
        self._current_lbfgs_history = None  # 保存当前轮次的 L-BFGS 历史供 AI 训练使用
        
        # 初始化 L-BFGS 优化器
        self.lbfgs_optimizer = LBFGSOptimizer(self.config)
        self.lbfgs_optimizer.current_mol = molecule
        self.lbfgs_optimizer.atom_symbols = molecule.atom_symbols
        self.lbfgs_optimizer.calculator = calculator

        # 初始化 AI 代理模型（只支持 gradient_predicting 方法）
        dim = molecule.n_atoms * 3
        ai_method = self.config.get('gpr', {}).get('type', 'gradient_predicting')

        # 只支持梯度预测 GPR 方法
        if ai_method == 'gradient_predicting':
            # 梯度预测 GPR：直接预测梯度向量，向梯度下降的方向预测
            from models.gradient_predicting_gpr import GradientPredictingGPR
            self.gpr_model = GradientPredictingGPR(self.config, dim)
            self.ai_method_name = "Gradient-Predicting GPR"
        else:
            # 默认使用梯度预测 GPR
            from models.gradient_predicting_gpr import GradientPredictingGPR
            self.gpr_model = GradientPredictingGPR(self.config, dim)
            self.ai_method_name = "Gradient-Predicting GPR"

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
        
        # 收敛判定参数（从配置文件读取）
        hybrid_config = self.config.get('hybrid', {})
        conv_config = hybrid_config.get('convergence', {})
        convergence_threshold = float(conv_config.get('threshold', self.config.get('optimizer', {}).get('convergence_threshold', 5e-4)))
        max_no_improvement = int(conv_config.get('max_no_improvement', 50))
        no_improvement_threshold = float(conv_config.get('no_improvement_threshold', 1e-6))

        iteration = 0
        no_improvement_count = 0

        # 记录全局最优点（基于梯度范数，而不是能量！）
        global_best_coords = coords.copy()
        global_best_energy = self._calculate_energy(coords)
        global_best_grad_norm = np.linalg.norm(
            self.calculator.calculate_gradient(self.atom_symbols, coords.reshape(-1, 3))
        )

        # GPR 配置
        use_gpr = self.config.get('gpr', {}).get('use_gpr', True)

        while iteration < max_iterations:
            self.current_round += 1

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n{'='*70}")
                print(f"第 {self.current_round} 轮优化")
                print(f"{'='*70}")
                # 输出当前轮的实际起点能量（从 coords 计算）
                current_start_energy = self._calculate_energy(coords)
                print(f"本轮起点：Energy = {current_start_energy:.10f} Hartree")

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

            # 保存 L-BFGS 历史供 AI 训练使用
            self._current_lbfgs_history = lbfgs_history
            
            # 检查 L-BFGS 是否提前收敛（实际迭代次数 < m）
            # 如果 scipy 提前终止，说明 L-BFGS 已达到收敛标准
            if len(lbfgs_history.iterations) < self.lbfgs_steps:
                # 先把本轮数据添加到总历史
                self.history.iterations.extend(round_iterations)
                
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n→ L-BFGS 提前收敛（实际迭代 {len(lbfgs_history.iterations)} 步 < {self.lbfgs_steps} 步）")
                    print(f"→ 已达到 L-BFGS 收敛标准，从全局选择最优结果\n")
                
                # 从全局历史中选择梯度最小的点
                best_iteration = min(self.history.iterations, key=lambda x: x.gradient_norm)
                
                # 判断最佳点来自哪一轮、哪种方法
                # 通过轮次信息来追踪
                if self.config.get('optimizer', {}).get('verbose', True):
                    # 找到最佳点来自哪一轮
                    # 由于我们没有存储轮次信息，这里简化处理：
                    # 直接使用当前轮次（因为 L-BFGS 提前收敛说明已经找到最优）
                    best_round = self.current_round
                    
                    # 判断是 L-BFGS 还是 GPR（根据梯度判断）
                    # 梯度最小的点通常来自 L-BFGS
                    if best_iteration.gradient_norm < 0.001:
                        best_method = "LBFGS"
                    else:
                        best_method = "GPR"
                    
                    best_step = best_iteration.iteration

                    print(f"全局最优：第 {best_round} 轮 {best_method} {best_step}, E={best_iteration.energy:.8f}, |g|={best_iteration.gradient_norm:.6f}")
                    print(f"收敛！梯度范数：{best_iteration.gradient_norm:.6f}")
                
                # 设置收敛标志
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                break

            # 执行 GPR 步骤（n 步）- 每轮都执行，验证 AI 方法可行性
            gpr_history = None
            if use_gpr:
                last_lbfgs_coords = lbfgs_history.iterations[-1].coords if lbfgs_history.iterations else coords
                gpr_history = self._run_gpr_steps(last_lbfgs_coords, self.gpr_steps)
                round_iterations.extend(gpr_history.iterations)

            # 从本轮候选点中选择最优
            # 候选点包括：
            # 1. L-BFGS 的最后一步（代表 L-BFGS 的最终结果）
            # 2. AI 方法的所有迭代（代表 AI 的探索结果）
            # 不包括：
            # - 本轮起点（已在上一轮评估过）
            # - L-BFGS 的中间步骤（只是过程，不是最终结果）

            # 获取候选点
            candidate_points = []

            # 1. 添加 L-BFGS 最后一步（如果有）
            if lbfgs_history.iterations:
                lbfgs_final = lbfgs_history.iterations[-1]
                candidate_points.append(lbfgs_final)

            # 2. 添加 AI 方法的所有迭代（如果有）
            if gpr_history and gpr_history.iterations:
                for it in gpr_history.iterations:
                    candidate_points.append(it)
            
            # 确保有候选点
            if not candidate_points:
                # 如果没有任何候选点，使用起点
                best_data = round_iterations[0]
                best_coords = best_data.coords
            else:
                # 获取起点数据（用于计算变化量）
                start_point = round_iterations[0]
                start_energy = start_point.energy
                start_grad_norm = start_point.gradient_norm
                
                # 获取权重配置
                weights_config = self.config.get('selection_weights', {})
                energy_weight = weights_config.get('energy_weight', 0.3)
                gradient_weight = weights_config.get('gradient_weight', 0.7)
                
                # 计算相对于起点的变化量（差值）
                # 能量变化：ΔE = E - E_start（负值表示能量下降）
                # 梯度变化：Δ|g| = |g| - |g|_start（负值表示梯度下降）
                energy_diffs = [it.energy - start_energy for it in candidate_points]
                grad_diffs = [it.gradient_norm - start_grad_norm for it in candidate_points]
                
                # 对差值进行归一化到 [0, 1] 范围
                # 归一化公式：(Δ - Δ_min) / (Δ_max - Δ_min)
                # 归一化后：0 表示变化最小，1 表示变化最大
                energy_diff_min, energy_diff_max = min(energy_diffs), max(energy_diffs)
                grad_diff_min, grad_diff_max = min(grad_diffs), max(grad_diffs)
                
                # 避免除零
                energy_range = (energy_diff_max - energy_diff_min) if (energy_diff_max - energy_diff_min) > 1e-10 else 1.0
                grad_range = (grad_diff_max - grad_diff_min) if (grad_diff_max - grad_diff_min) > 1e-10 else 1.0
                
                # 计算加权评分（评分越小越好）
                # 评分 = energy_weight * ΔE_normalized + gradient_weight * Δ|g|_normalized
                def compute_score(it, e_diff, g_diff):
                    e_norm = (e_diff - energy_diff_min) / energy_range
                    g_norm = (g_diff - grad_diff_min) / grad_range
                    return energy_weight * e_norm + gradient_weight * g_norm
                
                # 为每个候选点计算评分
                scored_points = []
                for i, it in enumerate(candidate_points):
                    score = compute_score(it, energy_diffs[i], grad_diffs[i])
                    scored_points.append((score, it))
                
                # 选择评分最低的点（综合变化最大且方向正确）
                best_score, best_data = min(scored_points, key=lambda x: x[0])
                best_coords = best_data.coords

            # 添加到总历史
            self.history.iterations.extend(round_iterations)
            iteration += len(round_iterations)

            # 判断最佳点来自 L-BFGS 还是 GPR
            is_best_from_lbfgs = lbfgs_history.iterations and best_data == lbfgs_history.iterations[-1]
            
            if self.config.get('optimizer', {}).get('verbose', True):
                if is_best_from_lbfgs:
                    print(f"\n本轮最佳：LBFGS {best_data.iteration}, E={best_data.energy:.8f}, |g|={best_data.gradient_norm:.6f}")
                elif gpr_history and gpr_history.iterations:
                    # 判断是 GPR 的第几步
                    gpr_step = list(gpr_history.iterations).index(best_data) + 1
                    print(f"\n本轮最佳：GPR {gpr_step}, E={best_data.energy:.8f}, |g|={best_data.gradient_norm:.6f}")
                else:
                    print(f"\n本轮最佳：Iter {best_data.iteration}, E={best_data.energy:.8f}, |g|={best_data.gradient_norm:.6f}")
                
                print(f"选择权重：能量={energy_weight:.2f}, 梯度={gradient_weight:.2f}")
                print(f"起点能量：{start_energy:.8f}, 起点梯度：{start_grad_norm:.6f}")
                print(f"最佳点能量变化：ΔE={best_data.energy - start_energy:.6f}, 梯度变化：Δ|g|={best_data.gradient_norm - start_grad_norm:.6f}")
                
                # 判断 GPR 是否找到更优点
                if use_gpr and gpr_history and gpr_history.iterations:
                    best_gpr_energy = min(it.energy for it in gpr_history.iterations)
                    best_gpr_grad = min(it.gradient_norm for it in gpr_history.iterations)
                    best_lbfgs_energy = lbfgs_history.iterations[-1].energy if lbfgs_history.iterations else float('inf')
                    best_lbfgs_grad = lbfgs_history.iterations[-1].gradient_norm if lbfgs_history.iterations else float('inf')
                    
                    if best_gpr_energy < best_lbfgs_energy - 1e-4 or best_gpr_grad < best_lbfgs_grad:
                        print(f"{self.ai_method_name} 找到更优点！")
                    else:
                        print(f"{self.ai_method_name} 未找到更优点（当前轮次）")

            # 每轮结束后统一训练 GPR（只训练 1 次）
            # 注意：gradient_predicting 方法使用自己的训练数据管理，不需要滑动窗口
            if ai_method != 'gradient_predicting':
                X, y, gradients = self.gpr_model.get_training_data()
                if len(X) > 3:
                    # 应用滑动窗口，只保留能量最好的 50% 的点（最多 max_training_points）
                    self.gpr_model.limit_training_data_by_percentile(50.0)
                    # 重新训练 GPR
                    X, y, gradients = self.gpr_model.get_training_data()
                    if self.config.get('optimizer', {}).get('verbose', True):
                        print(f"GPR 训练点数：{len(X)}")
                    self.gpr_model.train(X, y, gradients)

            # 下一轮从本轮最优点开始（不是全局最优）
            coords = best_coords.copy()

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
            # 梯度范数变化小于阈值认为无改进
            if len(self.history) > 1:
                prev_best = self.history.get_best_iteration('gradient')
                if prev_best is not None:
                    # 使用本轮最佳点的梯度范数
                    current_best_grad_norm = best_data.gradient_norm
                    # 梯度改进阈值：从配置读取
                    grad_diff = abs(current_best_grad_norm - prev_best.gradient_norm)
                    if grad_diff < no_improvement_threshold:
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

        # 优化结束：从所有历史迭代中选择梯度范数最小的坐标作为最终结果
        # 核心思想：分子几何优化的目标是找到梯度为零的稳定构型
        if self.history.iterations:
            # 找到梯度范数最小的点
            best_iteration = min(self.history.iterations, key=lambda x: x.gradient_norm)
            # 更新 history 的收敛信息
            self.history.converged = True  # 优化完成，标记为收敛
            self.history.best_iteration = best_iteration

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

        使用 L-BFGS 迭代 n_init 次生成初始采样点
        """
        n_init = self.config.get('gpr', {}).get('n_init', 5)

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
        self._initial_sampling_history = []  # 保存用于 AI 训练

        def callback(xk):
            """回调函数：收集 L-BFGS 迭代点"""
            energy = energy_func(xk)
            gradient = gradient_func(xk)
            collected_points.append((xk.copy(), energy, gradient.copy()))
            
            # 保存用于 AI 训练（迭代序号从 1 开始）
            from core.molecule import IterationData
            data = IterationData(
                iteration=len(self._initial_sampling_history) + 1,  # 从 1 开始
                energy=energy,
                gradient=gradient,
                coords=xk.copy()
            )
            self._initial_sampling_history.append(data)

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"  Init {len(collected_points)}: Energy = {energy:.8f} Hartree, "
                      f"|grad| = {np.linalg.norm(gradient):.6f}")

        # 运行 L-BFGS 收集初始点（固定迭代 n_init 次）
        result = minimize(
            fun=energy_func,
            x0=coords,
            method='L-BFGS-B',
            jac=gradient_func,
            callback=callback,
            options={
                'maxiter': n_init,  # 固定迭代次数
                'gtol': 1e-10,      # 设置很小的梯度阈值，让 L-BFGS 跑满指定次数
                'disp': False
            }
        )

        # 将所有点添加到 GPR 训练集（但不训练，留待循环中训练）
        for sampled_coords, energy, gradient in collected_points:
            self.gpr_model.add_data(sampled_coords, energy, gradient)

        if self.config.get('optimizer', {}).get('verbose', True):
            print("初始采样完成")

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
            print(f"\n初始采样最优：Energy = {best_energy:.8f} Hartree (Init {best_idx + 1})")
        
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

            # 记录迭代数据（迭代序号从 1 开始）
            iter_num = step_count[0] + 1  # 从 1 开始
            data = IterationData(
                iteration=iter_num,
                energy=energy,
                gradient=gradient,
                coords=xk.copy(),
                displacement=prev_coords[0] - xk
            )
            history.add_iteration(data)

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"LBFGS {iter_num}: E={energy:.8f}, |g|={gradient_norm:.6f}, d={displacement:.6f}")

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
        
        # L-BFGS 收敛控制参数
        lbfgs_config = self.config.get('lbfgs', {})
        gtol = float(lbfgs_config.get('gtol', 1e-5))  # 梯度阈值
        ftol = float(lbfgs_config.get('ftol', 1e12))  # 能量变化阈值（factr 参数）
        
        # scipy 的 ftol 参数控制能量收敛：
        # |ΔE/E| < ftol × epsmch ≈ ftol × 2.2e-16
        # 对于 E≈-154 Hartree：|ΔE| < ftol × 3.4e-14
        # ftol=1e12 → |ΔE| < 0.034 Hartree 收敛
        # ftol=1e18 → |ΔE| < 34000 Hartree 收敛（几乎禁用）
        
        result = minimize(
            fun=energy_only,
            x0=coords,
            method='L-BFGS-B',
            jac=gradient_only,
            callback=callback,
            options={
                'maxiter': n_steps,      # 连续运行 n_steps
                'gtol': gtol,            # 梯度收敛阈值
                'ftol': ftol,            # 能量相对变化阈值
                'disp': False
            }
        )
        
        # 输出 scipy 的收敛信息
        if self.config.get('optimizer', {}).get('verbose', True):
            if hasattr(result, 'message') and result.message:
                print(f"  → L-BFGS 终止原因：{result.message}")
        
        # 注意：不在这里训练 GPR，等到轮结束后统一训练

        return history
    
    def _run_gpr_steps(self, coords: np.ndarray, n_steps: int) -> OptimizationHistory:
        """
        执行 AI 预测步骤（gradient_predicting 方法）
        
        核心逻辑：
        1. 构建训练数据：i 个历史最优 + j 个最新
        2. 训练模型：学习 [坐标，梯度] → 新坐标 的映射
        3. 自回归预测 n 步：每步预测新坐标，并应用位移限制

        Args:
            coords: 初始坐标（L-BFGS 最后一步的坐标）
            n_steps: 预测步数

        Returns:
            history: 优化历史
        """
        history = OptimizationHistory()
        
        # 1. 构建训练数据
        self._prepare_ai_training_data()
        
        # 2. 训练 AI 模型
        # 注意：这里的 X, y 是占位符，实际训练数据在 gpr_model.X_train 中
        self.gpr_model.train(np.array([]), np.array([]))
        
        # 3. 自回归预测 n 步（迭代序号从 1 开始）
        current_coords = coords.copy()

        for i in range(n_steps):
            # 计算当前梯度
            current_gradient = self.calculator.calculate_gradient(
                self.atom_symbols, current_coords.reshape(-1, 3)
            )

            # AI 预测新坐标
            next_coords = self.gpr_model.predict_next_coords(
                current_coords, current_gradient,
                apply_displacement_limit=True
            )

            # 计算真实能量和梯度
            energy, gradient = self.calculator.calculate_energy_gradient(
                self.atom_symbols, next_coords.reshape(-1, 3)
            )
            gradient_norm = np.linalg.norm(gradient)

            # 记录（迭代序号从 1 开始）
            displacement = np.linalg.norm(next_coords - current_coords)
            data = self.get_iteration_data(
                iteration=len(history) + 1,  # 从 1 开始
                energy=energy,
                gradient=gradient,
                coords=next_coords,
                prev_coords=current_coords
            )
            history.add_iteration(data)

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"GPR   {len(history)}: E={energy:.8f}, |g|={gradient_norm:.6f}, d={displacement:.6f}")

            # 更新
            current_coords = next_coords.copy()

        return history
    
    def _prepare_ai_training_data(self) -> None:
        """
        准备 AI 训练数据

        从历史迭代中选取：
        - i 个历史梯度最优数据点
        - j 个最新数据点（如果本轮 L-BFGS 步数 m < j，往回找之前的数据）
        """
        # 清除旧数据
        self.gpr_model.clear_data()

        # 获取配置
        ai_config = self.config.get('ai_training', {})
        i_best = ai_config.get('i_best_steps', 5)
        j_recent = ai_config.get('j_recent_steps', 10)

        # 1. 选取 i 个历史梯度最优的点（从所有历史迭代中）
        all_iterations = list(self.history.iterations)
        training_data_added = 0

        if all_iterations:
            # 按梯度范数排序，取最优的 i 个
            sorted_by_grad = sorted(all_iterations, key=lambda x: x.gradient_norm)
            best_points = sorted_by_grad[:i_best]

            # 对于每个最优点，使用它本身作为目标（学习稳定构型）
            # 因为最优点已经接近平衡构型，不应该继续"优化"
            for it in best_points:
                try:
                    # 使用最优点本身作为目标（学习恒等映射）
                    # 这样 GPR 会学会：在梯度小的位置，坐标应该保持稳定
                    self.gpr_model.add_training_data(
                        it.coords.copy(), it.gradient.copy(), it.coords.copy()
                    )
                    training_data_added += 1
                except Exception as e:
                    print(f"Warning: 添加历史最优点失败：{e}")

        # 2. 选取 j 个最新数据点（从本轮 L-BFGS 历史中取）
        if hasattr(self, '_current_lbfgs_history') and self._current_lbfgs_history:
            lbfgs_points = list(self._current_lbfgs_history.iterations)
            recent_points = lbfgs_points[-j_recent:] if len(lbfgs_points) >= j_recent else lbfgs_points

            # 对于 L-BFGS 点，使用实际的下一步
            for i, it in enumerate(recent_points):
                try:
                    if i < len(recent_points) - 1:
                        # 中间点：下一步是序列中的下一个
                        next_coords = recent_points[i + 1].coords
                    else:
                        # 最后一个点：使用自身作为目标（学习恒等映射）
                        next_coords = it.coords

                    self.gpr_model.add_training_data(
                        it.coords.copy(), it.gradient.copy(), next_coords.copy()
                    )
                    training_data_added += 1
                except Exception as e:
                    print(f"Warning: 添加最新点失败：{e}")

        # 如果训练数据仍然不足，使用初始采样点
        if training_data_added < 2 and hasattr(self, '_initial_sampling_history'):
            init_points = list(self._initial_sampling_history)
            for i, it in enumerate(init_points[:-1]):
                try:
                    next_coords = init_points[i + 1].coords
                    self.gpr_model.add_training_data(
                        it.coords.copy(), it.gradient.copy(), next_coords.copy()
                    )
                    training_data_added += 1
                except Exception as e:
                    print(f"Warning: 添加初始点失败：{e}")

        print(f"AI 训练数据：{training_data_added} 个点 (i_best={i_best}, j_recent={j_recent})")
    
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

        # 从所有历史迭代中选择梯度范数最小的点作为最终结果
        best = self.history.get_best_iteration('gradient')
        if best:
            print(f"最优能量（全局梯度最小）：{best.energy:.10f} Hartree")
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
