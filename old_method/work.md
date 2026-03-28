# 分子几何构型优化：L-BFGS 与 GPR 方法研究

本项目基于量子化学计算（PySCF）和机器学习方法（高斯过程回归 GPR），实现了乙醇分子的几何构型优化。通过对比传统 L-BFGS 算法与基于梯度增强的 GPR 贝叶斯优化，探索了机器学习在势能面搜索中的潜力。

## 目录

1. [项目背景](#1-项目背景)
2. [实验环境与依赖](#2-实验环境与依赖)
3. [分子模型与计算方法](#3-分子模型与计算方法)
4. [方法原理](#4-方法原理)
5. [代码实现](#5-代码实现)
6. [实验结果与分析](#6-实验结果与分析)
7. [结论与展望](#7-结论与展望)
8. [参考文献](#8-参考文献)

## 项目文件结构

```
work/
├── work.md                    # 项目总览文档
├── L-BFGS.ipynb              # L-BFGS 优化实现 notebook
├── GPR.ipynb                 # 普通 GPR 优化实现 notebook（scikit-learn）
├── GPR_gpy.ipynb             # 梯度增强 GPR 优化实现 notebook（GPy）
├── lbfgs_ethanol.png         # L-BFGS 优化过程可视化
├── gpr_no_grad_ethanol.png   # 普通 GPR 优化过程可视化
├── gpy_gpr_ethanol.png       # 梯度增强 GPR 优化过程可视化（无扰动）
└── gpy_gpr_ethanol_perturb05.png  # 梯度增强 GPR 优化过程可视化（扰动 0.5Å）
```

---

## 1. 项目背景

分子几何构型优化是计算化学的基础问题，旨在找到分子势能面上的局部极小点（稳定构型）。传统方法（如 L-BFGS）利用梯度信息高效收敛，但对初始点敏感，且在远距离探索时可能陷入局部极小。近年来，基于高斯过程回归（GPR）的贝叶斯优化被引入，利用代理模型以少量量子化学计算快速逼近全局极小，尤其适用于初始结构远离平衡的情况。

**研究目标**：
- 对比 L-BFGS 与普通 GPR 的优化性能
- 探索梯度增强 GPR 在高维分子优化中的应用
- 评估不同初始结构（有无扰动）对优化结果的影响

**模型体系**：乙醇（C₂H₆O），在 RHF/cc-pVDZ 理论水平下进行优化。

---

## 2. 实验环境与依赖

- **操作系统**：Ubuntu 20.04 (WSL2)
- **Python 版本**：3.9
- **主要依赖库**：
  - `pyscf`：量子化学计算（能量、梯度）
  - `rdkit`：分子构型生成与内坐标转换
  - `scipy`：L-BFGS 优化器
  - `scikit-learn`：普通 GPR 模型
  - `GPy`：梯度增强多输出 GPR 模型
  - `matplotlib`：可视化

**安装命令**：
```bash
conda create -n molopt python=3.9
conda activate molopt
conda install -c conda-forge pyscf rdkit scipy scikit-learn matplotlib
pip install gpy
```

---

## 3. 分子模型与计算方法

### 3.1 分子结构

- **分子**：乙醇（C₂H₆O）
- **原子数**：9 个原子
- **自由度**：
  - 笛卡尔坐标：27 个自由度（9×3）
  - 内坐标：21 个自由度（消除整体平移旋转）

### 3.2 计算方法

- **理论水平**：RHF/cc-pVDZ
- **梯度类型**：解析梯度
- **初始结构**：由 RDKit 从 SMILES 'CCO' 生成三维结构
- **扰动设置**：可添加 ±0.5 Å 随机扰动模拟远距离起点

### 3.3 乙醇分子连接关系

```
原子序号  元素  连接关系
1        C     -
2        C     1
3        O     2
4-6      H     1 (3 个氢连接第一个碳)
7-8      H     2 (2 个氢连接第二个碳)
9        H     3 (1 个氢连接氧)
```

---

## 4. 方法原理

### 4.1 L-BFGS 算法

L-BFGS（Limited-memory BFGS）是一种拟牛顿法，适用于高维无约束优化。

**核心思想**：
- 仅存储最近 \(m\) 步的位移 \(\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k\) 和梯度变化 \(\mathbf{y}_k = \nabla f_{k+1} - \nabla f_k\)
- 通过两循环递归算法隐式计算搜索方向 \(\mathbf{p}_k = -\mathbf{H}_k \nabla f_k\)
- 结合 Wolfe 条件线搜索，保证全局收敛

**优点**：
- 内存占用 \(O(mn)\)
- 超线性收敛
- 适合高维问题

### 4.2 GPR 代理模型优化

GPR（高斯过程回归）是一种非参数贝叶斯模型，用于拟合未知函数 \(f(\mathbf{x})\)。

**预测公式**：
\[
\begin{aligned}
\mu(\mathbf{x}_*) &= \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y} \\
\sigma^2(\mathbf{x}_*) &= k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
\end{aligned}
\]

**采集函数（EI）**：
\[
EI(\mathbf{x}) = (f_{\min} - \mu(\mathbf{x})) \Phi\left(\frac{f_{\min} - \mu(\mathbf{x})}{\sigma(\mathbf{x})}\right) + \sigma(\mathbf{x}) \phi\left(\frac{f_{\min} - \mu(\mathbf{x})}{\sigma(\mathbf{x})}\right)
\]

### 4.3 梯度增强 GPR

梯度增强 GPR 将能量和梯度联合建模，每个样本提供 \(1+3N\) 个观测值，显著提升样本效率。

**核函数**：Matern52 × Coregionalize（建模输出间的相关性）

**复合采集函数**：
\[
\text{Acq}(\mathbf{x}) = EI(\mathbf{x}) - \lambda \cdot \|\nabla E_{\text{pred}}(\mathbf{x})\|
\]

其中 \(\lambda\) 控制梯度惩罚项，鼓励模型推荐低能量且小梯度的点。

---

## 5. 代码实现

### 5.1 L-BFGS 优化（L-BFGS.ipynb）

**核心代码流程**：

```python
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, scf, grad
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 生成乙醇初始分子（可带扰动）
def get_ethanol_init(seed=42, perturb_strength=0.0):
    mol = Chem.MolFromSmiles('CCO')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    if perturb_strength > 0:
        np.random.seed(seed)
        coords += np.random.uniform(-perturb_strength, perturb_strength, size=coords.shape)
    return atom_symbols, coords

# 2. PySCF 能量和梯度计算
def energy_grad_pyscf(atom_symbols, coords_ang):
    atom_str = '\n'.join([f"{atom_symbols[i]}  {coords_ang[i,0]:.8f}  {coords_ang[i,1]:.8f}  {coords_ang[i,2]:.8f}" 
                          for i in range(len(atom_symbols))])
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = 'cc-pvdz'
    mol.unit = 'angstrom'
    mol.build(verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    energy = mf.e_tot
    g = grad.RHF(mf).kernel()
    return energy, g.flatten()

# 3. L-BFGS 优化
x0 = init_coords_ang.flatten()
history = {'energy': [], 'grad_norm': []}

def callback(xk):
    e, g = energy_grad_pyscf(atom_symbols, xk.reshape(natom, 3))
    history['energy'].append(e)
    history['grad_norm'].append(np.linalg.norm(g))

res = minimize(fun, x0, method='L-BFGS-B', jac=jac, callback=callback,
               options={'maxiter': 100, 'gtol': 1e-5})
```

**关键参数**：
- `perturb_strength`：初始扰动强度（0 或 0.5 Å）
- `gtol`：梯度收敛阈值（1e-5）
- `maxiter`：最大迭代次数（100）

### 5.2 普通 GPR 优化（GPR.ipynb）

使用 scikit-learn 的 `GaussianProcessRegressor`，在内坐标空间进行优化。

**特点**：
- 使用内坐标（Z-matrix）表示分子
- 核函数：Matern52 + WhiteKernel
- 采集函数：EI（Expected Improvement）
- 局部搜索半径：0.5 Å

**局限性**：
- 缺乏梯度信息
- 在高维空间样本效率低

### 5.3 梯度增强 GPR 优化（GPR_gpy.ipynb）

使用 GPy 库构建多输出模型（能量 + 梯度）。

**核心实现**：

```python
import GPy

class GPyGradientOptimizer:
    def __init__(self, atom_symbols, z0, bounds_z, local_radius=0.5, n_init=10):
        # 初始化：生成 n_init 个样本点
        # 核函数：Matern52 × Coregionalize
        input_kernel = GPy.kern.Matern52(dim, ARD=True)
        coreg = GPy.kern.Coregionalize(input_dim=1, output_dim=output_dim, rank=5)
        kernel = input_kernel * coreg
        self.model = GPy.models.GPRegression(X, Y, kernel)
    
    def _composite_acq(self, x, lambda_grad=0.1, xi=0.001):
        # 复合采集函数：EI - λ·||∇E_pred||
        mu, var = self.model.predict(x.reshape(1,-1))
        ei = calculate_ei(mu[0,0], var[0,0], y_min)
        pred_grad_norm = np.linalg.norm(mu[0,1:])
        return ei - lambda_grad * pred_grad_norm
```

**关键参数**：
- `n_init`：初始采样点数（10）
- `local_radius`：局部搜索半径（0.5 Å）
- `lambda_grad`：梯度惩罚权重（0.1）
- `xi`：EI 探索参数（0.001）

---

## 6. 实验结果与分析

### 6.1 实验设置

所有实验均采用 RHF/cc-pVDZ 理论水平，乙醇分子。

**初始结构**：
- **无扰动**：RDKit 生成的平衡结构
- **扰动 0.5**：添加 ±0.5 Å 随机扰动

### 6.2 L-BFGS 优化结果

| 初始条件 | 迭代次数 | 最终能量 (Hartree) | 最终梯度范数 | 收敛状态 |
|---------|---------|-------------------|-------------|---------|
| 无扰动   | 42      | -154.09271898     | 0.00032     | 收敛     |
| 扰动 0.5  | 64      | -153.91088477     | 0.00001     | 收敛     |

**分析**：
- L-BFGS 在近平衡点时收敛迅速（42 步）
- 远距离初始点（扰动 0.5）需要更多迭代（64 步）
- 扰动情况下落入不同的局部极小（能量高约 0.18 Hartree）

### 6.3 普通 GPR 优化结果

| 初始条件 | 迭代次数 | 最优能量 (Hartree) | 收敛状态 |
|---------|---------|-------------------|---------|
| 无扰动   | 10      | -154.07946221     | 早停     |
| 扰动 0.5  | 10      | -153.51568333     | 早停     |

**分析**：
- 普通 GPR 因缺乏梯度信息，样本效率低
- 在局部区域无法精确建模势能面
- 连续 10 步无改进后早停

### 6.4 梯度增强 GPR 优化结果

| 初始条件 | 初始采样 | 优化迭代 | 最优能量 (Hartree) | 收敛状态 |
|---------|---------|---------|-------------------|---------|
| 扰动 0.5  | 10      | 11      | -153.51568333     | 早停     |

**分析**：
- 梯度信息被引入，但参数设置过于保守
- 局部搜索半径（0.5 Å）过小
- 采集函数参数（xi=0.001）限制探索
- 初始样本不足（10 点）

### 6.5 结果对比

**能量对比图**：
- L-BFGS（无扰动）：最优，-154.0927 Hartree
- L-BFGS（扰动 0.5）：次优，-153.9109 Hartree
- GPR 方法：未能收敛到极小点

**收敛速度**：
- L-BFGS：线性收敛，稳定下降
- GPR：早期改进快，后期停滞

---

## 7. 结论与展望

### 7.1 主要结论

1. **L-BFGS**：
   - 在近平衡区域高效可靠
   - 对初始点敏感，远距离时易陷入局部极小
   - 适合作为精细优化方法

2. **普通 GPR**：
   - 难以单独完成高维势能面优化
   - 必须引入梯度信息

3. **梯度增强 GPR**：
   - 有潜力但参数敏感
   - 需仔细调优（搜索范围、样本量、采集函数）

### 7.2 未来改进方向

1. **动态调整局部半径**：根据优化进程自适应调整搜索范围
2. **优化采集函数参数**：采用自适应λ和ξ参数
3. **丰富训练集**：记录 L-BFGS 中间步以增强 GPR 训练
4. **先进算法**：引入 TuRBO（Trust Region Bayesian Optimization）
5. **混合策略**：L-BFGS 与 GPR 交替执行，兼顾局部收敛和全局探索

### 7.3 计算化学与机器学习的结合

本项目展示了机器学习方法在分子优化中的应用潜力。虽然当前 GPR 方法尚未超越传统 L-BFGS，但梯度增强策略和混合优化方向值得进一步探索。

---

## 8. 参考文献

1. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1), 503-528.

2. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

3. Shahriari, B., et al. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.

4. Sun, S., et al. (2019). Gradient-enhanced Gaussian process regression for molecular potential energy surfaces. *J. Chem. Phys.*, 150(10), 104102.

5. Ahuja, K., et al. (2021). Reinforcement learning for molecular geometry optimization. *NeurIPS 2021 AI for Science Workshop*.

---

## 附录：关键代码文件说明

### L-BFGS.ipynb
- 完整的 L-BFGS 优化实现
- 包含能量/梯度记录
- 可视化优化轨迹

### GPR.ipynb
- 基于 scikit-learn 的普通 GPR 实现
- 内坐标转换
- EI 采集函数优化

### GPR_gpy.ipynb
- 基于 GPy 的梯度增强 GPR 实现
- 多输出建模（能量 + 梯度）
- 复合采集函数
