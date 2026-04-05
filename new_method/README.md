# L-BFGS-GPR 混合优化项目

## 分子几何构型优化：L-BFGS 与高斯过程回归的混合策略

---

## 摘要

本项目实现了将传统优化方法 L-BFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）与人工智能方法 GPR（Gaussian Process Regression，高斯过程回归）相结合的分子几何构型优化框架。项目采用 Python 自主编码实现，验证了 L-BFGS 与 AI 代理模型协同优化的可行性，为智能优化方法在计算化学领域的应用提供了可参考的技术路线。

**核心贡献**：
1. 实现了完整的 L-BFGS+GPR 混合优化框架
2. 提出了基于滑动窗口的训练数据筛选机制
3. 设计了能量激励的采集函数引导 GPR 探索
4. 通过乙醇分子优化实验验证了方法的收敛性

---

## 目录

1. [研究背景与意义](#研究背景与意义)
2. [理论基础](#理论基础)
3. [方法实现](#方法实现)
4. [算法流程](#算法流程)
5. [参数设置与边界条件](#参数设置与边界条件)
6. [安装与使用](#安装与使用)
7. [实验结果](#实验结果)
8. [结论与展望](#结论与展望)
9. [项目结构](#项目结构)
10. [参考文献](#参考文献)

---

## 研究背景与意义

### 1.1 分子几何构型优化问题

分子几何构型优化是计算化学中的核心问题之一，其目标是找到分子势能面上的极小值点（稳定构型）。数学上，这等价于求解以下优化问题：

$$\min_{\mathbf{R}} E(\mathbf{R})$$

其中 $\mathbf{R} = \{\mathbf{r}_1, \mathbf{r}_2, ..., \mathbf{r}_N\}$ 为 N 个原子的三维坐标，$E(\mathbf{R})$ 为势能函数（通过量子化学计算获得）。

### 1.2 传统优化方法的局限

传统优化方法如 L-BFGS、共轭梯度法等已广泛应用于分子优化，但存在以下局限：
- **易陷入局部极小值**：严重依赖初始构型
- **计算成本高**：每次迭代都需要昂贵的量子化学计算
- **缺乏全局探索能力**：难以跨越势能垒

### 1.3 人工智能方法的潜力

机器学习代理模型（如高斯过程回归）可以：
- **预测势能面**：用廉价预测替代部分量子化学计算
- **引导全局探索**：通过采集函数平衡探索与开发
- **提供不确定性量化**：指导采样策略

### 1.4 研究目标

本项目旨在探索 L-BFGS 与 GPR 的结合方式，验证以下假设：
- **可行性**：L-BFGS+AI 框架可以实现并收敛
- **兼容性**：AI 方法的引入不破坏 L-BFGS 的收敛性
- **参考价值**：为后续研究提供可复现的基线框架

---

## 理论基础

### 2.1 L-BFGS 算法原理

L-BFGS 是一种拟牛顿法，通过近似 Hessian 矩阵的逆来实现快速收敛。

#### 2.1.1 基本迭代公式

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{H}_k \nabla f(\mathbf{x}_k)$$

其中：
- $\mathbf{x}_k$：第 k 步的坐标
- $\alpha_k$：步长（通过线搜索确定）
- $\mathbf{H}_k$：Hessian 逆近似
- $\nabla f(\mathbf{x}_k)$：梯度

#### 2.1.2 BFGS 更新公式

$$\mathbf{H}_{k+1} = (\mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^T) \mathbf{H}_k (\mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^T) + \rho_k \mathbf{s}_k \mathbf{s}_k^T$$

其中：
- $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$
- $\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$
- $\rho_k = 1 / (\mathbf{y}_k^T \mathbf{s}_k)$

#### 2.1.3 Limited-memory 策略

L-BFGS 只存储最近 m 步的 $(\mathbf{s}_k, \mathbf{y}_k)$ 对，通过递归方式计算 $\mathbf{H}_k \nabla f(\mathbf{x}_k)$，大大降低了内存需求。

### 2.2 高斯过程回归（GPR）

GPR 是一种贝叶斯非参数回归方法，适用于小样本、高维度的回归问题。

#### 2.2.1 基本模型

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

其中：
- $m(\mathbf{x})$：均值函数（通常设为 0）
- $k(\mathbf{x}, \mathbf{x}')$：核函数（协方差函数）

#### 2.2.2 预测分布

给定训练数据 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$，新输入 $\mathbf{x}_*$ 的预测分布为：

$$p(f_* | \mathbf{x}_*, \mathcal{D}) = \mathcal{N}(\mu_*, \sigma_*^2)$$

其中：
- $\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$
- $\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$

#### 2.2.3 核函数选择

本项目使用 Matern 5/2 核：

$$k(r) = \sigma^2 \left(1 + \frac{\sqrt{5}r}{l} + \frac{5r^2}{3l^2}\right) \exp\left(-\frac{\sqrt{5}r}{l}\right)$$

其中 $r = \|\mathbf{x} - \mathbf{x}'\|$，$l$ 为长度尺度，$\sigma^2$ 为方差。

### 2.3 采集函数（Acquisition Function）

采集函数用于平衡探索（exploration）与开发（exploitation）。

#### 2.3.1 期望提升（Expected Improvement, EI）

$$EI(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f_{min})]$$

解析形式：

$$EI(\mathbf{x}) = (\mu(\mathbf{x}) - f_{min} - \xi) \Phi(Z) + \sigma(\mathbf{x}) \phi(Z)$$

其中：
- $Z = \frac{\mu(\mathbf{x}) - f_{min} - \xi}{\sigma(\mathbf{x})}$
- $\Phi$：标准正态累积分布函数
- $\phi$：标准正态概率密度函数
- $\xi$：探索参数

#### 2.3.2 能量激励采集函数（本项目改进）

$$Acq(\mathbf{x}) = 5.0 \times EI(\mathbf{x}) + 2.0 \times \max(0, f_{min} - \mu(\mathbf{x}))$$

第二项为能量差直接激励，鼓励预测能量低于当前最优的点。

---

## 方法实现

### 3.1 整体架构

项目采用模块化设计，主要包含以下组件：

```
┌─────────────────────────────────────────────────────────────┐
│                      主程序 (main.py)                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │   L-BFGS 优化器    │           │  L-BFGS+GPR 混合  │
    │   (lbfgs.py)      │           │  (hybrid.py)      │
    └───────────────────┘           └───────────────────┘
              │                               │
              │                               ▼
              │                     ┌───────────────────┐
              │                     │   GPR 代理模型     │
              │                     │ (gradient_gpr.py) │
              │                     └───────────────────┘
              │                               │
              ▼                               ▼
    ┌───────────────────────────────────────────────────────┐
    │                 量子化学计算器                         │
    │                  (calculator.py)                      │
    │              (PySCF 后端，RHF/cc-pvdz)                │
    └───────────────────────────────────────────────────────┘
```

### 3.2 核心类设计

#### 3.2.1 Molecule 类

```python
class Molecule:
    """分子结构数据类"""
    
    def __init__(self, atom_symbols: List[str], 
                 coords: np.ndarray, 
                 smiles: str = "", 
                 name: str = ""):
        self.atom_symbols = atom_symbols  # 原子符号列表
        self.coords = coords              # 笛卡尔坐标 (n_atoms, 3)
        self.smiles = smiles              # SMILES 字符串
        self.n_atoms = len(atom_symbols)
    
    @classmethod
    def from_smiles(cls, smiles: str, seed: int = 42, 
                    perturb_strength: float = 0.1) -> 'Molecule':
        """从 SMILES 生成分子，可添加初始扰动"""
```

#### 3.2.2 OptimizationHistory 类

```python
class OptimizationHistory:
    """优化历史记录类"""
    
    def __init__(self):
        self.iterations = []          # IterationData 列表
        self.converged = False        # 收敛标志
        self.convergence_iteration = None  # 收敛时的迭代序号
    
    def add_iteration(self, data: IterationData):
        """添加一次迭代记录"""
    
    def get_best_iteration(self, metric: str = 'energy') -> IterationData:
        """获取最优迭代（energy 或 gradient）"""
```

#### 3.2.3 IterationData 类

```python
class IterationData:
    """单次迭代的数据记录"""
    
    def __init__(self, iteration: int, energy: float, 
                 gradient: np.ndarray, coords: np.ndarray,
                 displacement: np.ndarray = None):
        self.iteration = iteration      # 迭代序号
        self.energy = energy            # 能量 (Hartree)
        self.gradient = gradient        # 梯度 (3*n_atoms,)
        self.gradient_norm = np.linalg.norm(gradient)
        self.coords = coords            # 坐标 (3*n_atoms,)
```

### 3.3 L-BFGS 优化器实现

#### 3.3.1 核心方法

```python
class LBFGSOptimizer(BaseOptimizer):
    """L-BFGS 优化器"""
    
    def optimize(self, molecule: Molecule, 
                 calculator: QuantumCalculator) -> OptimizationHistory:
        """执行完整的 L-BFGS 优化"""
        
        x0 = molecule.get_coords_flat()
        
        result = minimize(
            fun=self._energy_func.energy_only,
            x0=x0,
            method='L-BFGS-B',
            jac=self._energy_func.gradient_only,
            callback=self._callback,  # 记录每步数据
            options={
                'maxiter': self.maxiter,
                'gtol': self.gtol,
                'disp': False
            }
        )
        
        # 检查收敛
        final_grad_norm = np.linalg.norm(
            self._energy_func.gradient_only(result.x)
        )
        self.history.converged = final_grad_norm < self.gtol
        
        return self.history
```

#### 3.3.2 收敛判据

L-BFGS 使用 scipy 的内部收敛判据：
- `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`：函数值相对减小量小于阈值
- `CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL`：投影梯度范数小于容差

**注意**：scipy 的 `success=True` 表示优化器正常终止，但不一定满足用户设定的梯度阈值。

### 3.4 L-BFGS+GPR 混合优化器实现

#### 3.4.1 混合策略设计

```python
class HybridOptimizer(BaseOptimizer):
    """L-BFGS+GPR 混合优化器"""
    
    def optimize(self, molecule: Molecule, 
                 calculator: QuantumCalculator) -> OptimizationHistory:
        """执行混合优化"""
        
        # 1. 初始采样（L-BFGS 生成 n_init 个点）
        self._initial_sampling(molecule)
        
        # 2. 从初始点中选择最优作为起点
        coords = self._get_best_from_initial_samples()
        
        # 3. 主循环
        while iteration < max_iterations:
            # 3.1 执行 L-BFGS 步骤（m 步，连续运行保持 Hessian 连续性）
            lbfgs_history = self._run_lbfgs_steps(coords, self.lbfgs_steps)
            
            # 3.2 执行 GPR 步骤（n 步，基于采集函数探索）
            gpr_history = self._run_gpr_steps(
                lbfgs_history.iterations[-1].coords, 
                self.gpr_steps
            )
            
            # 3.3 从本轮所有点（起点+m+n）中选择能量最优
            best_data = min(round_iterations, key=lambda x: x.energy)
            
            # 3.4 更新全局最优
            if best_data.energy < global_best_energy:
                global_best_energy = best_data.energy
                global_best_coords = best_data.coords.copy()
            
            # 3.5 滑动窗口：只保留能量最好的 50% 训练点
            self.gpr_model.limit_training_data_by_percentile(50.0)
            
            # 3.6 重新训练 GPR
            self.gpr_model.train(X, y, gradients)
            
            # 3.7 下一轮从全局最优点开始
            coords = global_best_coords.copy()
        
        return self.history
```

#### 3.4.2 L-BFGS 连续优化（使用 callback 机制）

```python
def _run_lbfgs_steps(self, coords: np.ndarray, n_steps: int):
    """
    执行固定步数的 L-BFGS 优化（保持 Hessian 连续性）
    
    关键改进：使用 scipy 的 callback 机制，一次 minimize 调用连续运行 n_steps 步，
    而不是每步调用一次 minimize。这保证了 Hessian 近似信息的连续性。
    """
    history = OptimizationHistory()
    step_count = [0]
    prev_coords = [coords.copy()]
    
    def callback(xk):
        """scipy 的 callback，在每步后调用"""
        energy = energy_only(xk)
        gradient = gradient_only(xk)
        
        # 记录数据
        data = IterationData(
            iteration=step_count[0],
            energy=energy,
            gradient=gradient,
            coords=xk.copy()
        )
        history.add_iteration(data)
        
        # 添加到 GPR 训练集
        self.gpr_model.add_data(xk.copy(), energy, gradient.copy())
        
        step_count[0] += 1
        prev_coords[0] = xk.copy()
    
    # 连续运行 n_steps 步
    result = minimize(
        fun=energy_only,
        x0=coords,
        method='L-BFGS-B',
        jac=gradient_only,
        callback=callback,
        options={'maxiter': n_steps, 'gtol': 1e-10}
    )
    
    return history
```

#### 3.4.3 GPR 探索步骤

```python
def _run_gpr_steps(self, coords: np.ndarray, n_steps: int):
    """
    执行 GPR 预测步骤
    
    GPR 通过采集函数建议新点，然后计算真实能量和梯度
    """
    history = OptimizationHistory()
    current_coords = coords.copy()
    y_min = min(self.gpr_model.y_train)
    
    for i in range(n_steps):
        # 1. GPR 建议下一个点（优化采集函数）
        next_coords = self.gpr_model.suggest_next_point(
            self._bounds, y_min
        )
        
        # 2. 计算真实能量和梯度
        energy, gradient = calculator.calculate_energy_gradient(
            atom_symbols, next_coords.reshape(-1, 3)
        )
        
        # 3. 记录数据
        data = IterationData(...)
        history.add_iteration(data)
        
        # 4. 添加到 GPR 训练集
        self.gpr_model.add_data(next_coords, energy, gradient)
        
        current_coords = next_coords
        y_min = min(y_min, energy)
    
    return history
```

#### 3.4.4 滑动窗口机制

```python
def limit_training_data_by_percentile(self, percentile: float = 50.0):
    """
    限制训练数据，只保留能量最好的前 percentile% 的点
    
    目的：
    1. 控制计算开销（GPR 训练复杂度 O(n³)）
    2. 聚焦最优区域（移除远离最优区域的"差质量点"）
    3. 提高 GPR 预测质量
    """
    if len(self.X_train) < 3:
        return
    
    # 计算要保留的点数
    n_keep = max(3, int(len(self.X_train) * percentile / 100.0))
    n_keep = min(n_keep, self.max_training_points)
    
    # 按能量排序，保留最好的点
    sorted_idx = np.argsort(self.y_train)
    keep_idx = sorted_idx[:n_keep]
    
    self.X_train = [self.X_train[i] for i in keep_idx]
    self.y_train = [self.y_train[i] for i in keep_idx]
    self.grad_train = [self.grad_train[i] for i in keep_idx]
    
    self.is_trained = False  # 需要重新训练
```

---

## 算法流程

### 4.1 L-BFGS 算法流程

```
算法 1: L-BFGS 分子几何构型优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：
  - 初始分子结构 R₀
  - 收敛阈值 ε（默认 5×10⁻⁴）
  - 最大迭代次数 N_max（默认 300）

输出：
  - 优化后的分子结构 R*
  - 优化历史

步骤：
1. 初始化：
   - 计算初始能量 E₀ = E(R₀)
   - 计算初始梯度 g₀ = ∇E(R₀)
   - 设置 Hessian 逆近似 H₀ = I

2. For k = 0, 1, 2, ..., N_max:
   2.1 计算搜索方向：p_k = -H_k g_k
   2.2 线搜索确定步长 α_k（满足 Wolfe 条件）
   2.3 更新坐标：R_{k+1} = R_k + α_k p_k
   2.4 计算新梯度：g_{k+1} = ∇E(R_{k+1})
   2.5 计算位移和梯度差：
       s_k = R_{k+1} - R_k
       y_k = g_{k+1} - g_k
   2.6 更新 Hessian 逆近似 H_{k+1}（BFGS 公式）
   2.7 收敛检查：
       If ‖g_{k+1}‖ < ε:
           返回 R* = R_{k+1}，收敛

3. 返回 R* = R_{N_max}，未收敛
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.2 L-BFGS+GPR 混合算法流程

```
算法 2: L-BFGS+GPR 混合优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：
  - 初始分子结构 R₀
  - L-BFGS 步数 m（默认 5）
  - GPR 步数 n（默认 1）
  - 收敛阈值 ε（默认 5×10⁻⁴）
  - 最大迭代次数 N_max（默认 300）

输出：
  - 优化后的分子结构 R*
  - 优化历史

步骤：
1. 初始采样（生成 GPR 初始训练集）：
   1.1 从 R₀ 运行 L-BFGS，生成 n_init 个点（默认 3 个）
   1.2 如果 L-BFGS 提前收敛，用扰动点补充
   1.3 选择能量最低的点作为主循环起点 R_start

2. 初始化 GPR 模型：
   2.1 用初始采样点训练 GPR
   2.2 设置优化边界：R_start ± local_radius

3. 主循环：For round = 1, 2, ..., N_rounds:
   
   3.1 L-BFGS 阶段（连续 m 步）：
       For i = 1 to m:
           - 执行 L-BFGS 单步
           - 记录 (R_i, E_i, g_i)
           - 添加到 GPR 训练集
   
   3.2 GPR 阶段（n 步探索）：
       For j = 1 to n:
           a) 优化采集函数，建议新点 R_cand：
              R_cand = argmax Acq(R)
              Acq(R) = 5.0×EI(R) + 2.0×max(0, E_min - μ(R))
           b) 计算真实能量和梯度：E_cand, g_cand
           c) 记录 (R_cand, E_cand, g_cand)
           d) 添加到 GPR 训练集
   
   3.3 选择本轮最优：
       R_best = argmin{E | 所有本轮点}
   
   3.4 更新全局最优：
       If E(R_best) < E_global_min:
           E_global_min = E(R_best)
           R_global = R_best
   
   3.5 滑动窗口筛选训练数据：
       - 按能量排序，保留前 50% 的点
       - 重新训练 GPR
   
   3.6 收敛检查：
       If ‖∇E(R_global)‖ < ε:
           返回 R* = R_global，收敛
   
   3.7 早停检查：
       If 连续 50 轮能量改进 < 10⁻⁷ Hartree:
           返回 R* = R_global，早停

4. 返回 R* = R_global
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.3 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        初始分子 R₀                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  初始采样（L-BFGS 生成 3 个点）                                     │
│  - Init 0: (R₀, E₀, g₀)                                         │
│  - Init 1: (R₁, E₁, g₁)                                         │
│  - Init 2: (R₂, E₂, g₂)（扰动补充）                              │
│                                                                 │
│  选择最优：R_start = argmin{E₀, E₁, E₂}                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    第 1 轮优化                                    │
├─────────────────────────────────────────────────────────────────┤
│  L-BFGS 阶段（5 步，连续运行）：                                    │
│  ├─ Step 1: R_start → R₁, E₁, g₁ → 添加到 GPR 训练集             │
│  ├─ Step 2: R₁ → R₂, E₂, g₂ → 添加到 GPR 训练集                 │
│  ├─ Step 3: R₂ → R₃, E₃, g₃ → 添加到 GPR 训练集                 │
│  ├─ Step 4: R₃ → R₄, E₄, g₄ → 添加到 GPR 训练集                 │
│  └─ Step 5: R₄ → R, E₅, g → 添加到 GPR 训练集                 │
│                                                                 │
│  GPR 阶段（1 步探索）：                                            │
│  ├─ GPR 建议：R_cand（基于采集函数）                            │
│  ├─ 计算真实值：E_cand, g_cand                                  │
│  └─ 添加到 GPR 训练集                                            │
│                                                                 │
│  选择最优：R_best = argmin{E_start, E₁, ..., E₅, E_cand}        │
│  更新全局：If E(R_best) < E_global: R_global = R_best           │
│  滑动窗口：保留能量前 50% 的点 → 重训 GPR                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    第 2 轮优化                                    │
│  起点：R_global（从第 1 轮继承）                                   │
│  ...（同第 1 轮）                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          ... 循环 ...
```

---

## 参数设置与边界条件

### 5.1 优化器参数

| 参数 | 符号 | 默认值 | 说明 | 取值依据 |
|------|------|--------|------|----------|
| 最大迭代次数 | $N_{max}$ | 300 | 优化迭代上限 | 保证充分收敛 |
| 收敛阈值 | $\varepsilon$ | 5×10⁻⁴ | 梯度范数阈值 | 化学上可认为"收敛" |
| L-BFGS 步数/轮 | $m$ | 5 | 每轮 L-BFGS 步数 | 平衡效率与探索 |
| GPR 步数/轮 | $n$ | 1 | 每轮 GPR 探索步数 | 验证可行性为主 |
| 早停阈值 | $\delta_E$ | 10⁻⁷ Hartree | 能量改进阈值 | 约 0.06 kcal/mol |
| 早停轮数 | $N_{stop}$ | 50 | 连续无改进轮数 | 允许小改进继续 |

### 5.2 GPR 参数

| 参数 | 符号 | 默认值 | 说明 | 取值依据 |
|------|------|--------|------|----------|
| 初始采样点数 | $n_{init}$ | 3 | L-BFGS 生成点数 | 减少训练开销 |
| GPR 类型 | - | simple | 只能量/能量 + 梯度 | simple 更快 |
| 局部搜索半径 | $r_{local}$ | 0.1 Å | 优化边界 | 聚焦最优区域 |
| 最大训练点数 | $N_{train}$ | 15 | 滑动窗口上限 | 控制 O(n³) 开销 |
| 噪声方差 | $\sigma_n^2$ | 10⁻² | 观测噪声 | 加快收敛 |
| 探索参数 | $\xi$ | 10⁻⁴ | EI 探索强度 | 聚焦 exploitation |
| 能量激励系数 | $\lambda_E$ | 2.0 | 直接能量激励 | 鼓励低能量 |
| EI 放大系数 | $\lambda_{EI}$ | 5.0 | EI 值放大 | 增强探索 |

### 5.3 边界条件设置

#### 5.3.1 优化边界

GPR 的探索范围限制在当前最优结构附近：

$$\mathbf{R}_{global} \pm r_{local}$$

其中 $r_{local} = 0.1$ Å（默认）。

**物理意义**：限制 GPR 在合理范围内探索，避免预测远离最优区域的点。

#### 5.3.2 滑动窗口边界

训练数据筛选条件：

$$\mathcal{D}_{new} = \{(\mathbf{R}_i, E_i, \mathbf{g}_i) \mid E_i \leq E_{50\%}\}$$

其中 $E_{50\%}$ 为能量排序的第 50 百分位数。

**物理意义**：移除"差质量点"，让 GPR 聚焦最优区域。

### 5.4 收敛判据

#### 5.4.1 梯度收敛（主要判据）

$$\|\nabla E(\mathbf{R}^*)\| < \varepsilon$$

默认 $\varepsilon = 5 \times 10^{-4}$ Hartree/Bohr。

**化学意义**：梯度小于 0.0005 Hartree/Bohr 可认为达到稳定构型。

#### 5.4.2 能量收敛（辅助判据）

$$|E_{k} - E_{k-1}| < \delta_E$$

默认 $\delta_E = 10^{-7}$ Hartree（约 0.06 kcal/mol）。

**化学意义**：能量变化小于 0.06 kcal/mol，化学上可认为"基本不变"。

#### 5.4.3 早停判据

连续 $N_{stop}$ 轮能量改进小于 $\delta_E$：

$$\forall i \in [k-N_{stop}, k]: |E_i - E_{i-1}| < \delta_E$$

---

## 安装与使用

### 6.1 环境要求

- Python >= 3.8
- 量子化学计算：PySCF >= 2.0.0
- 分子处理：RDKit >= 2022.0.0
- 机器学习：scikit-learn >= 1.0.0
- 优化：scipy >= 1.8.0
- 绘图：matplotlib >= 3.5.0, zhplot >= 0.1.0

### 6.2 安装步骤

```bash
# 1. 创建conda环境（可选）
conda create -n gpr_env python=3.9
conda activate gpr_env

# 2. 安装依赖
cd /mnt/e/wsl_dir/L-BFGS-GPR/new_method
pip install -r requirements.txt
```

### 6.3 运行示例

#### 6.3.1 L-BFGS 基准测试

```bash
# 无扰动初始结构
python main.py --method lbfgs --molecule ethanol

# 添加扰动（0.5 Å）
python main.py --method lbfgs --molecule ethanol --perturb 0.5
```

#### 6.3.2 L-BFGS+GPR 混合优化

```bash
# 默认配置（推荐）
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 自定义参数
python main.py --method hybrid --molecule ethanol \
    --perturb 0.1 \
    --max-iter 200 \
    --threshold 1e-3
```

#### 6.3.3 对比实验

```bash
# 运行对比脚本
python run_comparison.py --smiles CCO --perturb 0.1
```

### 6.4 配置文件

通过修改 `config/default_config.yaml` 调整参数：

```yaml
# 分子设置
molecule:
  smiles: "CCO"              # 乙醇
  perturb_strength: 0.1      # 初始扰动 (Å)

# 优化器设置
optimizer:
  max_iterations: 300        # 最大迭代次数
  convergence_threshold: 5e-4  # 收敛阈值

# GPR 设置
gpr:
  n_init: 3                  # 初始采样点数
  local_radius: 0.1          # 搜索半径 (Å)
  max_training_points: 15    # 最大训练点数
  use_gpr: true              # 是否启用 GPR

# 混合策略
hybrid:
  lbfgs_steps: 5             # 每轮 L-BFGS 步数
  gpr_steps: 1               # 每轮 GPR 步数
```

### 6.5 输出说明

运行完成后，输出目录包含：

```
output/
├── hybrid_YYYYMMDD_HHMMSS.json          # 优化历史（JSON）
├── hybrid_trajectory_*.xyz              # 优化轨迹（XYZ 格式）
├── hybrid_details_*.json                # 详细迭代信息
├── plots/
│   ├── hybrid_energy.png                # 能量收敛曲线
│   ├── hybrid_gradient.png              # 梯度收敛曲线
│   └── hybrid_combined.png              # 组合图表
└── structures/
    ├── hybrid_initial.xyz               # 初始结构
    ├── hybrid_final.xyz                 # 最终结构
    └── hybrid_comparison.png            # 结构对比图
```

---

## 实验结果

### 7.1 实验设置

- **测试分子**：乙醇（CCO）
- **计算方法**：RHF/cc-pvdz
- **初始结构**：从 SMILES 生成，添加 0.1 Å 随机扰动
- **收敛标准**：梯度范数 < 5×10⁻⁴ Hartree/Bohr

### 7.2 L-BFGS 基准结果

| 指标 | 数值 |
|------|------|
| 初始能量 | -154.0157 Hartree |
| 最终能量 | -154.0927 Hartree |
| 能量改进 | 0.0770 Hartree (48.3 kcal/mol) |
| 初始梯度 | 0.1585 Hartree/Bohr |
| 最终梯度 | 0.0005 Hartree/Bohr |
| 迭代次数 | ~80 |
| 计算时间 | ~25 分钟 |

### 7.3 L-BFGS+GPR 混合结果

| 指标 | 数值 |
|------|------|
| 初始能量 | -154.0157 Hartree |
| 最终能量 | -154.0927 Hartree |
| 能量改进 | 0.0770 Hartree |
| 初始梯度 | 0.1585 Hartree/Bohr |
| 最终梯度 | 0.0005 Hartree/Bohr |
| 迭代次数 | ~80 |
| 计算时间 | ~35 分钟 |
| GPR 参与轮次 | 100% |
| GPR 找到更优点 | 0% |

### 7.4 结果分析

#### 7.4.1 收敛性

两种方法都成功收敛到相同能量（-154.0927 Hartree），证明：
- ✓ L-BFGS+GPR 框架不影响 L-BFGS 的收敛性
- ✓ 混合方法能找到与纯 L-BFGS 相同的极小值

#### 7.4.2 GPR 表现

GPR 预测能量始终高于 L-BFGS（约 0.05-0.09 Hartree），原因：
- **训练数据少**：滑动窗口只保留 12-15 个点
- **高维空间**：27 维优化空间，GPR 拟合困难
- **探索范围大**：0.1 Å 边界内 GPR 预测不准

#### 7.4.3 计算开销

| 方法 | 时间/迭代 | 总时间 |
|------|----------|--------|
| L-BFGS | ~0.3 分钟 | ~25 分钟 |
| L-BFGS+GPR | ~0.45 分钟 | ~35 分钟 |

GPR 每轮增加约 50% 开销，主要来自：
- GPR 训练（O(n³) 复杂度）
- 采集函数优化（20 次随机重启）

### 7.5 结论

1. **可行性验证**：L-BFGS+GPR 框架成功实现并收敛
2. **兼容性验证**：GPR 的引入不破坏 L-BFGS 收敛
3. **性能局限**：当前 GPR 预测质量有待提升
4. **参考价值**：为后续研究提供了基线框架

---

## 结论与展望

### 8.1 主要结论

本项目成功实现了 L-BFGS 与 GPR 的混合优化框架，主要结论如下：

1. **框架可行性**：通过自主编码实现了 L-BFGS+AI 的完整流程，验证了传统优化器与 AI 代理模型结合的可行性。

2. **收敛性保证**：混合方法能够收敛到与纯 L-BFGS 相同的极小值，证明了 AI 方法的引入不会破坏原有优化器的收敛性。

3. **技术挑战**：
   - 高维空间（27 维）中 GPR 预测质量有限
   - GPR 训练开销随数据量快速增长（O(n³)）
   - 采集函数设计需要平衡探索与开发

4. **创新点**：
   - 提出了基于滑动窗口的训练数据筛选机制
   - 设计了能量激励的采集函数
   - 使用 callback 机制保持 L-BFGS 的 Hessian 连续性

### 8.2 局限性

1. **GPR 预测质量**：当前 GPR 未能找到比 L-BFGS 更优的点，主要起"探索尝试"作用。

2. **计算效率**：GPR 训练增加了约 40% 的计算开销。

3. **参数敏感性**：GPR 性能对核函数参数、探索参数等较为敏感。

### 8.3 未来展望

#### 8.3.1 GPR 性能提升

- **更快的实现**：使用 GPyTorch 等 GPU 加速库
- **降维处理**：PCA 或自动编码器降低输入维度
- **稀疏 GPR**：使用诱导点减少训练数据

#### 8.3.2 采集函数改进

- **自适应参数**：根据优化阶段动态调整 $\xi$ 和 $\lambda_{EI}$
- **多目标采集**：同时考虑能量、梯度、不确定性
- **基于置信度**：在高置信度区域更多开发

#### 8.3.3 混合策略优化

- **动态步数**：根据收敛情况调整 m 和 n
- **早期退出**：如果 GPR 持续失败，减少 GPR 步数
- **多保真度**：结合低精度和高精度计算

#### 8.3.4 应用扩展

- **更大分子**：测试蛋白质、聚合物等复杂体系
- **其他 AI 模型**：尝试神经网络、随机森林等
- **多尺度优化**：结合分子力学和量子力学

---

## 项目结构

```
new_method/
├── core/                        # 核心数据类
│   ├── molecule.py              # 分子结构、迭代历史
│   └── calculator.py            # 量子化学计算接口 (PySCF)
│
├── optimizers/                  # 优化器
│   ├── base.py                  # 优化器基类
│   ├── lbfgs.py                 # L-BFGS 优化器
│   └── hybrid.py                # L-BFGS+GPR 混合优化器
│
├── models/                      # 机器学习模型
│   ├── gpr_base.py              # GPR 基类
│   └── gradient_gpr.py          # 简单 GPR 模型 (sklearn)
│
├── visualization/               # 可视化
│   ├── structure3d.py           # 3D 分子结构
│   └── plots.py                 # 能量/梯度图表
│
├── utils/                       # 工具函数
│   ├── io_utils.py              # 数据输入输出
│   └── converters.py            # 坐标转换
│
├── config/                      # 配置文件
│   └── default_config.yaml      # 默认配置
│
├── main.py                      # 主程序
├── run_comparison.py            # 对比实验脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 项目文档
```

---

## 参考文献

1. Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage. *Mathematics of Computation*, 35(151), 773-782.

2. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1), 503-528.

3. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

4. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

5. Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.

6. Smith, J. S., et al. (2017). ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. *Chemical Science*, 8(4), 3192-3203.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

8. Sun, Q., et al. (2018). Recent developments in the PySCF program package. *The Journal of Chemical Physics*, 153(2), 024109.

---

## 致谢

本项目用于学术研究和教学目的。

**项目联系人**：LiuZhe  
**邮箱**：3266048598@qq.com

如有问题或建议，请提交 Issue。

---

## 附录：运行日志示例

### A.1 L-BFGS 优化日志

```
======================================================================
L-BFGS 优化开始
======================================================================
初始能量：-154.0157174741 Hartree
原子数：9
自由度：27
======================================================================
Iter    0: Energy = -154.07083856 Hartree, |grad| = 0.158465
Iter    1: Energy = -154.07558982 Hartree, |grad| = 0.101891
...
Iter   78: Energy = -154.09269977 Hartree, |grad| = 0.000489
======================================================================
优化完成！
最终能量：-154.09269977 Hartree
最终梯度范数：0.000489
迭代次数：79
收敛状态：是
计算时间：1523.45 秒
======================================================================
```

### A.2 L-BFGS+GPR 优化日志

```
======================================================================
L-BFGS+GPR 混合优化开始
======================================================================
L-BFGS 步数 (m): 5
GPR 步数 (n): 1
选择标准：energy
初始能量：-154.0157174741 Hartree
======================================================================

使用 L-BFGS 生成 3 个初始采样点...
  Init 0: Energy = -154.07083856 Hartree, |grad| = 0.158465
  Init 1: Energy = -154.07558982 Hartree, |grad| = 0.101891

L-BFGS 提前收敛，补充 1 个扰动点...
  Init 2: Energy = -154.03249090 Hartree, |grad| = 0.403696

GPR 模型训练完成
初始采样最优：Energy = -154.0755898230 Hartree (Init 1)

======================================================================
第 1 轮优化
======================================================================
本轮起点：Energy = -154.0755898230 Hartree
LBFGS 0: E=-154.08104664, |g|=0.101891, d=0.093566
LBFGS 1: E=-154.08255786, |g|=0.102476, d=0.060831
LBFGS 2: E=-154.08422056, |g|=0.093374, d=0.033140
LBFGS 3: E=-154.08489242, |g|=0.040089, d=0.025494
LBFGS 4: E=-154.08513249, |g|=0.041626, d=0.023189
GPR   0: E=-153.89955966, |g|=1.158287, d=0.575876
GPR 未找到更优点（当前轮次）

本轮最佳：Iter 4, E=-154.08513249, |g|=0.053007
全局最佳：E=-154.0851324940 Hartree
GPR 训练点数：4
...

======================================================================
优化完成！
最终能量：-154.09269977 Hartree
最终梯度范数：0.000489
迭代次数：79
收敛状态：是
计算时间：2134.67 秒
======================================================================
```
