# L-BFGS+GPR 混合优化策略设计文档

## 1. 核心逻辑概述

### 1.1 抽象核心逻辑


核心逻辑：使用 AI 方法（gradient_predicting）**模仿 L-BFGS 优化器的轨迹，向前预测**，从而帮助 L-BFGS 更快完成优化。

```
┌─────────────────────────────────────────────────────────────────┐
│                    L-BFGS+GPR 混合优化流程                       │
├─────────────────────────────────────────────────────────────────┤
│  初始 L-BFGS (n_init 步) → 生成初始训练数据                        │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              主循环 (每轮)                               │   │
│  │  ┌─────────────────────┐                               │   │
│  │  │  L-BFGS 步骤 (m 步)    │ → 收集训练数据                  │   │
│  │  └─────────────────────┘                               │   │
│  │         ↓                                               │   │
│  │  ┌─────────────────────┐                               │   │
│  │  │  GPR 训练            │ ← 使用 i_best + j_recent 数据    │   │
│  │  └─────────────────────┘                               │   │
│  │         ↓                                               │   │
│  │  ┌─────────────────────┐                               │   │
│  │  │  GPR 预测 (n 步)      │ → 向前探路                      │   │
│  │  └─────────────────────┘                               │   │
│  │         ↓                                               │   │
│  │  ┌─────────────────────┐                               │   │
│  │  │  最优点选择          │ ← 加权评分 (能量 30% + 梯度 70%)   │   │
│  │  └─────────────────────┘                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  收敛检查 → 满足则停止，否则继续下一轮                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计思想

1. **轨迹模仿学习**：GPR 学习 L-BFGS 的优化轨迹，而非直接预测能量/梯度
2. **向前探路**：GPR 预测的坐标点相当于"探路"，帮助探索势能面
3. **数据选择策略**：
   - `i_best_steps`：全局梯度范数最小的点（学习稳定构型）
   - `j_recent_steps`：最新 L-BFGS 迭代点（学习优化方向）
4. **加权选择**：每轮从 L-BFGS 终点和 GPR 预测点中，按能量/梯度加权选择最优

---

## 2. 详细设计

### 2.1 混合优化器架构

```
/mnt/e/wsl_dir/L-BFGS-GPR/new_method/
├── optimizers/
│   ├── hybrid.py           # 混合优化器 (HybridOptimizer)
│   └── lbfgs.py            # L-BFGS 优化器 (LBFGSOptimizer)
├── models/
│   ├── gradient_predicting_gpr.py  # 梯度预测 GPR 模型
│   ├── gradient_gpr.py             # 梯度 GPR 基类
│   └── gpr_base.py                 # GPR 抽象基类
├── core/
│   ├── molecule.py         # 分子数据结构
│   └── calculator.py       # 量子化学计算器
└── config/
    └── default_config.yaml # 配置文件
```

### 2.2 核心类关系图

```
┌──────────────────────┐
│  HybridOptimizer     │  混合优化器（主控制器）
│  - lbfgs_optimizer   │
│  - gpr_model         │
│  - calculator        │
└──────────┬───────────┘
           │
           ├─────────────────────────────────┐
           │                                 │
           ▼                                 ▼
┌──────────────────────┐        ┌──────────────────────┐
│  LBFGSOptimizer      │        │ GradientPredictingGPR│
│  (scipy L-BFGS-B)    │        │ (AI 预测模型)          │
│  - 执行固定步数优化   │        │ - 输入：[坐标，梯度]   │
│  - callback 收集数据  │        │ - 输出：新坐标        │
└──────────────────────┘        └──────────────────────┘
           │                                 │
           └────────────────┬────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  QuantumCalculator   │
                  │  (PySCF 封装)         │
                  │  - 计算能量           │
                  │  - 计算梯度           │
                  └──────────────────────┘
```

### 2.3 算法流程详解

#### 2.3.1 初始化阶段

```python
# 1. 初始化 L-BFGS 优化器
self.lbfgs_optimizer = LBFGSOptimizer(config)

# 2. 初始化 GPR 模型（维度 = 3 * n_atoms * 2 = 54 维输入）
self.gpr_model = GradientPredictingGPR(config, dim)

# 3. 设置边界（基于初始坐标的局部区域）
self._setup_bounds(molecule)

# 4. 初始采样：L-BFGS 运行 n_init 步生成训练数据
self._initial_sampling(molecule)
```

#### 2.3.2 主循环（每轮）

```python
while iteration < max_iterations:
    # ========== 步骤 1: L-BFGS 优化 (m 步) ==========
    lbfgs_history = self._run_lbfgs_steps(coords, self.lbfgs_steps)
    
    # ========== 步骤 2: 准备 AI 训练数据 ==========
    self._prepare_ai_training_data()
    # - 选取 i_best_steps 个全局梯度最优的点
    # - 选取 j_recent_steps 个最新 L-BFGS 迭代点
    
    # ========== 步骤 3: 训练 GPR 模型 ==========
    self.gpr_model.train(X, y)  # 实际数据在 gpr_model.X_train 中
    
    # ========== 步骤 4: GPR 预测 (n 步) ==========
    gpr_history = self._run_gpr_steps(last_lbfgs_coords, self.gpr_steps)
    
    # ========== 步骤 5: 选择本轮最优点 ==========
    # 候选点：L-BFGS 最后一步 + 所有 GPR 预测点
    # 评分 = energy_weight * ΔE_norm + gradient_weight * Δ|g|_norm
    best_coords = select_best_from_round()
    
    # ========== 步骤 6: 收敛检查 ==========
    if gradient_norm < convergence_threshold:
        break
```

#### 2.3.3 GPR 训练数据构建

```python
def _prepare_ai_training_data(self):
    """
    训练数据构建策略：
    
    1. 历史最优点 (i_best_steps 个)
       - 从所有历史迭代中按梯度范数排序
       - 使用最优点本身作为目标（学习恒等映射）
       - 目的：学会在梯度小的位置保持稳定
    
    2. 最新 L-BFGS 点 (j_recent_steps 个)
       - 从本轮 L-BFGS 历史中取最新 j 个点
       - 使用实际的下一步作为目标（学习优化轨迹）
       - 目的：学会 L-BFGS 的优化方向
    """
    # 示例：对于乙醇 (9 原子，27 自由度)
    # 输入：[坐标 (27 维), 梯度 (27 维)] = 54 维
    # 输出：新坐标 (27 维)
```

#### 2.3.4 GPR 预测机制

```python
def predict_next_coords(self, coords, gradient):
    """
    预测下一步坐标（自回归方式）
    
    1. 构建输入：input_vec = [coords, gradient] (54 维)
    2. 对每个坐标分量预测：next_coords[i] = model[i].predict(input_vec)
    3. 应用位移限制：
       - 最大位移：max_displacement = 0.3 Å
       - 最小位移：min_displacement = 0.001 Å
    """
    input_vec = np.concatenate([coords, gradient]).reshape(1, -1)
    
    # 27 个独立 GPR 模型，每个预测一个坐标分量
    for i in range(self.dim):
        next_coords[i] = self.models[i].predict(input_vec)[0]
    
    # 位移限制（防止预测点偏离太远）
    displacement = next_coords - coords
    if ||displacement|| > max_displacement:
        displacement *= (max_displacement / ||displacement||)
    
    return coords + displacement
```

#### 2.3.5 最优点选择策略

```python
def select_best_from_round():
    """
    加权评分选择最优点
    
    候选点：
    1. L-BFGS 最后一步
    2. 所有 GPR 预测点（n 步）
    
    评分计算：
    1. 计算相对于起点的变化量：
       - ΔE = E - E_start
       - Δ|g| = |g| - |g|_start
    
    2. 归一化到 [0, 1]：
       - ΔE_norm = (ΔE - ΔE_min) / (ΔE_max - ΔE_min)
       - Δ|g|_norm = (Δ|g| - Δ|g|_min) / (Δ|g|_max - Δ|g|_min)
    
    3. 加权评分（越小越好）：
       - score = 0.3 * ΔE_norm + 0.7 * Δ|g|_norm
    """
```

---

## 3. 梯度范数 (Gradient Norm)

### 3.1 数学定义

在分子几何优化中，**梯度范数**是衡量系统偏离平衡构型程度的关键指标。

#### 3.1.1 梯度向量

对于包含 $N$ 个原子的分子，总能量 $E$ 是 $3N$ 个坐标的函数：

$$E = E(\mathbf{R}) = E(x_1, y_1, z_1, x_2, y_2, z_2, \ldots, x_N, y_N, z_N)$$

**梯度向量**（一阶导数）：

$$\mathbf{g} = \nabla E(\mathbf{R}) = \left[ \frac{\partial E}{\partial x_1}, \frac{\partial E}{\partial y_1}, \frac{\partial E}{\partial z_1}, \ldots, \frac{\partial E}{\partial z_N} \right]^T \in \mathbb{R}^{3N}$$

#### 3.1.2 梯度范数（欧几里得范数）

$$\|\mathbf{g}\| = \sqrt{\sum_{i=1}^{N} \left[ \left(\frac{\partial E}{\partial x_i}\right)^2 + \left(\frac{\partial E}{\partial y_i}\right)^2 + \left(\frac{\partial E}{\partial z_i}\right)^2 \right]}$$

**物理意义**：
- $\|\mathbf{g}\| = 0$：系统处于**稳定点**（极小值、极大值或鞍点）
- $\|\mathbf{g}\| > 0$：系统受力，原子会沿负梯度方向移动以降低能量
- $\|\mathbf{g}\|$ 越小：越接近平衡构型

### 3.2 代码实现

```python
# /mnt/e/wsl_dir/L-BFGS-GPR/new_method/core/molecule.py
class IterationData:
    def __init__(self, iteration: int, energy: float, gradient: np.ndarray,
                 coords: np.ndarray, displacement: Optional[np.ndarray] = None):
        self.iteration = iteration
        self.energy = energy
        self.gradient = gradient.copy()
        # 梯度范数计算（L2 范数）
        self.gradient_norm = np.linalg.norm(gradient)
        # ...
```

```python
# /mnt/e/wsl_dir/L-BFGS-GPR/new_method/core/calculator.py
def calculate_gradient(self, atom_symbols, coords):
    """
    计算梯度（PySCF 解析梯度）
    
    返回：gradient (3*N 维数组)
    单位：Hartree/Angstrom（如果坐标单位是 Angstrom）
    """
    _, gradient = self.calculate_energy_gradient(atom_symbols, coords)
    return gradient
```

### 3.3 收敛标准

```yaml
# config/default_config.yaml
optimizer:
  convergence_threshold: 1.0e-4  # 梯度范数收敛阈值

lbfgs:
  gtol: 1.0e-4  # L-BFGS 投影梯度范数阈值
```

**收敛判定**：
$$\|\mathbf{g}\| < 10^{-4} \text{ Hartree/Å} \quad \Rightarrow \quad \text{优化收敛}$$

---

## 4. 理论清单

### 4.1 高斯过程回归 (GPR)

#### 4.1.1 基本原理

高斯过程回归是一种**非参数贝叶斯回归方法**：

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

- $m(\mathbf{x})$：均值函数（通常设为 0）
- $k(\mathbf{x}, \mathbf{x}')$：核函数（协方差函数）

#### 4.1.2 核函数（Kernel Functions）

本项目使用的核函数：

```yaml
# config/default_config.yaml
gpr:
  kernel_type: "matern52"
  noise_variance: 1.0e-2
```

**Matérn 5/2 核**：

$$k_{\text{Matérn-5/2}}(r) = \sigma^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)$$

其中：
- $r = \|\mathbf{x} - \mathbf{x}'\|$：两点间距离
- $\ell$：长度尺度（length scale）
- $\sigma^2$：方差

**组合核**（代码实现）：

```python
kernel = (
    ConstantKernel(1.0, (1e-1, 1e1)) *
    Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 100.0)) +
    WhiteKernel(1e-3, (1e-4, 1e-1))
)
```

- `ConstantKernel`：常数核（缩放因子）
- `Matern`：Matérn 5/2 核（平滑性适中）
- `WhiteKernel`：白噪声核（处理噪声）

#### 4.1.3 GPR 预测

给定训练数据 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$，新输入 $\mathbf{x}_*$ 的预测：

预测均值：

$$\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

预测方差：

$$\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

其中：
- $\mathbf{K}$：训练数据协方差矩阵，$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
- $\mathbf{k}_*$：测试点与训练点的协方差向量，$k_*(\mathbf{x}_*) = [k(\mathbf{x}_1, \mathbf{x}_*), \ldots, k(\mathbf{x}_n, \mathbf{x}_*)]^T$
- $\sigma_n^2$：噪声方差
- $\mathbf{I}$：单位矩阵

### 4.2 L-BFGS 优化算法

#### 4.2.1 BFGS 算法

**拟牛顿法**：用近似 Hessian 矩阵 $\mathbf{B}_k$ 替代真实 Hessian

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{B}_k^{-1} \nabla E(\mathbf{x}_k)$$

**BFGS 更新公式**：

$$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^T \mathbf{B}_k}{\mathbf{s}_k^T \mathbf{B}_k \mathbf{s}_k}$$

其中：
- $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$
- $\mathbf{y}_k = \nabla E(\mathbf{x}_{k+1}) - \nabla E(\mathbf{x}_k)$

#### 4.2.2 L-BFGS（有限内存 BFGS）

**核心思想**：不存储完整的 Hessian 近似，只保留最近 $m$ 步的 $\{\mathbf{s}_i, \mathbf{y}_i\}$ 对

**Two-loop recursion**（两圈递归）：

```python
# scipy.optimize.minimize(method='L-BFGS-B')
# 参数：
# - memory: 记忆步数（默认 10）
# - gtol: 梯度收敛阈值
# - ftol: 能量变化阈值
```

#### 4.2.3 投影梯度（L-BFGS-B）

对于有边界约束的问题，使用**投影梯度**：

$$\|\mathbf{g}_{\text{proj}}\| = \sqrt{\sum_i g_{\text{proj},i}^2}$$

$$g_{\text{proj},i} = \begin{cases}
\frac{\partial E}{\partial x_i} & \text{if } x_i \text{ 在边界内} \\
\min(0, \frac{\partial E}{\partial x_i}) & \text{if } x_i \text{ 在下界} \\
\max(0, \frac{\partial E}{\partial x_i}) & \text{if } x_i \text{ 在上界}
\end{cases}$$

### 4.3 量子化学计算

#### 4.3.1 电子结构方法

```yaml
# config/default_config.yaml
calculation:
  method: "RHF"      # 限制 Hartree-Fock
  basis: "cc-pvdz"   # 相关一致性极化双ζ基组
```

**RHF（Restrict Hartree-Fock）**：
- 闭壳层分子
- 自旋轨道配对
- 能量：$E_{\text{RHF}} = \langle \Psi | \hat{H} | \Psi \rangle$

#### 4.3.2 基组（Basis Set）

**cc-pVDZ**（correlation-consistent polarized Valence Double Zeta）：
- 价层：2 组基函数
- 极化函数：d 轨道（对 C、N、O 等）
- 精度：适中，适合中小分子

#### 4.3.3 解析梯度（Analytical Gradient）

PySCF 计算的梯度是**解析梯度**（不是数值微分）：

$$\frac{\partial E}{\partial \mathbf{R}} = \left\langle \Psi \left| \frac{\partial \hat{H}}{\partial \mathbf{R}} \right| \Psi \right\rangle + \text{波函数响应项}$$

**单位转换**：
- PySCF 内部：Hartree/Bohr
- 输出（Angstrom 坐标）：Hartree/Å

$$1 \text{ Hartree/Bohr} = \frac{1}{0.529177} \text{ Hartree/Å} \approx 1.8897 \text{ Hartree/Å}$$

### 4.4 Python 库调用

#### 4.4.1 核心依赖

| 库 | 用途 | 关键模块 |
|---|---|---|
| **PySCF** | 量子化学计算 | `pyscf.gto`, `pyscf.scf`, `pyscf.grad` |
| **scikit-learn** | GPR 模型 | `sklearn.gaussian_process` |
| **scipy** | L-BFGS 优化 | `scipy.optimize.minimize` |
| **RDKit** | 分子结构生成 | `rdkit.Chem.AllChem` |
| **numpy** | 数值计算 | `numpy.linalg.norm` |

#### 4.4.2 关键 API 调用

```python
# 1. PySCF 能量/梯度计算
from pyscf import gto, scf, grad

mol = gto.Mole()
mol.atom = "C 0 0 0; H 0 0 1.09; ..."
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()  # SCF 收敛

g = grad.RHF(mf)
gradient = g.kernel()  # Hartree/Bohr

# 2. scipy L-BFGS-B 优化
from scipy.optimize import minimize

result = minimize(
    fun=energy_func,
    x0=coords,
    method='L-BFGS-B',
    jac=gradient_func,
    callback=callback,
    options={'maxiter': n_steps, 'gtol': 1e-4}
)

# 3. scikit-learn GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
gpr.fit(X_train, y_train)
```

### 4.5 AI 方法理论

#### 4.5.1 模仿学习（Imitation Learning）

本项目 GPR 的本质是**模仿 L-BFGS 的优化轨迹**：

```
状态：s = [坐标，梯度] ∈ ℝ^(2×3N)
动作：a = 新坐标 ∈ ℝ^(3N)
策略：π(s) → a  (GPR 学习从状态到动作的映射)
```

#### 4.5.2 滑动窗口（Sliding Window）

```python
# 保留能量最好的前 50% 的点（最多 max_training_points）
def limit_training_data_by_percentile(self, percentile=50.0):
    n_keep = max(3, int(len(self.X_train) * percentile / 100.0))
    sorted_idx = np.argsort(self.y_train)
    keep_idx = sorted_idx[:n_keep]
```

**目的**：
- 减少计算量（GPR 复杂度 $O(n^3)$）
- 保留高质量数据
- 适应势能面局部特征

#### 4.5.3 自回归预测（Autoregressive Prediction）

```python
# GPR 预测 n 步（迭代方式）
current_coords = coords
for i in range(n_steps):
    next_coords = gpr_model.predict_next_coords(current_coords, gradient)
    current_coords = next_coords  # 反馈到下一步
```

**特点**：
- 每步预测依赖前一步结果
- 误差会累积（需要位移限制）

---

## 5. 变量说明

### 5.1 配置文件变量

| 变量 | 含义 | 默认值 |
|---|---|---|
| `gpr.n_init` | 初始 L-BFGS 采样步数 | 5 |
| `hybrid.lbfgs_steps` | 每轮 L-BFGS 步数 (m) | 10 |
| `hybrid.gpr_steps` | 每轮 GPR 预测步数 (n) | 2 |
| `ai_training.i_best_steps` | 历史最优训练点数 | 5 |
| `ai_training.j_recent_steps` | 最新训练点数 | 10 |
| `selection_weights.energy_weight` | 能量选择权重 | 0.3 |
| `selection_weights.gradient_weight` | 梯度选择权重 | 0.7 |
| `ai_prediction.max_displacement` | GPR 最大位移限制 | 0.3 Å |
| `gpr.max_training_points` | GPR 最大训练点数 | 15 |

### 5.2 核心变量

| 变量 | 维度 | 含义 |
|---|---|---|
| `coords` | $(3N,)$ | 分子坐标（展平） |
| `gradient` | $(3N,)$ | 能量梯度 |
| `gradient_norm` | 标量 | $\|\nabla E\|_2$ |
| `X_train` | $(n, 6N)$ | GPR 输入：[坐标，梯度] |
| `y_train` | $(n, 3N)$ | GPR 输出：新坐标 |

---

## 6. 代码调用流程

### 6.1 主入口

```python
# main.py
from optimizers.hybrid import HybridOptimizer
from core.calculator import QuantumCalculator
from core.molecule import Molecule

# 1. 创建分子
mol = Molecule.from_smiles("CCO", seed=42, perturb_strength=0.1)

# 2. 加载配置
with open("config/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 3. 创建优化器
optimizer = HybridOptimizer(config)

# 4. 执行优化
history = optimizer.optimize(mol, calculator)
```

### 6.2 关键方法调用链

```
HybridOptimizer.optimize()
├── _initial_sampling()          # 初始 L-BFGS 采样
│   └── scipy.optimize.minimize()
├── _run_lbfgs_steps()           # 每轮 L-BFGS
│   └── scipy.optimize.minimize()
│       └── callback() → 收集数据
├── _prepare_ai_training_data()  # 准备训练数据
│   ├── 选取 i_best_steps
│   └── 选取 j_recent_steps
├── gpr_model.train()            # 训练 GPR
│   └── sklearn GPR.fit()
├── _run_gpr_steps()             # GPR 预测
│   └── gpr_model.predict_next_coords()
└── _select_best_from_round()    # 选择最优点
```

---

## 7. 总结

### 7.1 创新点

1. **轨迹模仿**：GPR 不预测能量/梯度，而是学习 L-BFGS 的优化轨迹
2. **混合策略**：L-BFGS（局部精细优化）+ GPR（全局探路）
3. **数据选择**：结合历史最优（稳定点）和最新数据（优化方向）

### 7.2 适用场景

- 分子几何优化
- 势能面搜索
- 需要避免局部极小值的问题

### 7.3 局限性

- GPR 训练复杂度 $O(n^3)$，训练点数受限
- 自回归预测误差累积
- 依赖 L-BFGS 初始轨迹质量
