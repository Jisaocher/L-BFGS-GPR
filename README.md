# L-BFGS-GPR 分子几何构型优化项目

**分子几何构型优化 - L-BFGS 与 GPR 混合策略研究**

---

## 项目信息

- **项目维护者**: 刘喆 (Liu Zhe)
- **联系邮箱**: 3266048598@qq.com
- **研究日期**: 2024
- **项目类型**: 计算化学 + 机器学习

---

## 项目概述

本项目研究基于机器学习的分子几何构型优化方法，通过对比传统 L-BFGS 算法与基于高斯过程回归 (GPR) 的贝叶斯优化，探索机器学习在分子势能面搜索中的应用潜力。

项目包含两个版本：
- **old_method**: 初始研究版本，包含基础实现和实验验证
- **new_method**: 重构优化版本，实现模块化设计和混合优化策略

---

## 项目结构

```
L-BFGS-GPR/
├── old_method/              # 初始研究版本
│   ├── L-BFGS.ipynb        # L-BFGS 优化 Notebook
│   ├── GPR.ipynb           # 普通 GPR 优化 Notebook
│   ├── GPR_gpy.ipynb       # 梯度增强 GPR 优化 Notebook
│   └── ...                 # 相关输出和可视化
│
├── new_method/              # 重构优化版本 ⭐
│   ├── core/               # 核心数据类
│   ├── optimizers/         # 优化器实现
│   ├── models/             # GPR 模型
│   ├── visualization/      # 可视化模块
│   ├── utils/              # 工具函数
│   ├── config/             # 配置文件
│   ├── main.py             # 主程序
│   ├── run_comparison.py   # 对比脚本
│   └── README.md           # 详细文档
│
├── README.md               # 本文件（项目总览）
└── ...
```

---

## 研究方法对比

### old_method（初始版本）

**特点**：
- Jupyter Notebook 形式，便于交互式实验
- 三种方法独立实现：L-BFGS、普通 GPR、梯度增强 GPR
- 使用 PySCF 进行量子化学计算
- 初步验证了混合策略的可行性

**局限性**：
- 代码耦合度高，难以复用
- 输出格式不统一，不利于结果对比
- 缺乏模块化设计，扩展困难
- 可视化功能有限

### new_method（优化版本）⭐

**核心改进**：

| 方面 | old_method | new_method |
|------|------------|------------|
| 代码结构 | Notebook 脚本 | 模块化 Python 包 |
| 优化策略 | 独立运行 | L-BFGS+GPR 混合 |
| 数据输出 | 简单打印 | JSON/CSV/XYZ 多格式 |
| 可视化 | 基础图表 | 3D 结构 + 对比图 |
| 可扩展性 | 困难 | 易于扩展新 AI 方法 |
| 配置管理 | 硬编码 | YAML 配置文件 |

---

## new_method 核心优化功能

### 1. 混合优化策略

**设计思路**：每 m 步 L-BFGS 后加上 n 步 GPR 预测，选择表现最好的迭代结果。

```
┌─────────────────────────────────────────────────────────────┐
│                    混合优化流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   初始采样 → L-BFGS(m 步) → GPR(n 步) → 选择最佳 → 收敛检查   │
│      ↑                                          │          │
│      └──────────────────────────────────────────┘          │
│                         循环                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键参数**（可在 `config/default_config.yaml` 中调整）：
- `hybrid.lbfgs_steps`: 每轮 L-BFGS 步数 (m)，默认 5
- `hybrid.gpr_steps`: 每轮 GPR 步数 (n)，默认 2
- `hybrid.selection_metric`: 选择标准（energy/gradient/combined）

### 2. 梯度增强 GPR 优化

**核心改进**：GPR 预测方向明确指向"能量低、梯度为零"的稳定构型。

**复合采集函数**：
```
Acq(x) = EI(x) - λ·||∇E_pred(x)||
```

其中：
- `EI(x)`: 期望提升（Expected Improvement），鼓励探索低能量区域
- `||∇E_pred(x)||`: 预测梯度范数，鼓励选择梯度小的点
- `λ`: 梯度惩罚权重，平衡两项贡献

**设计原理**：
- 最稳定的几何构型具有"低能量 + 梯度为零"的特点
- 传统 EI 只考虑能量，可能收敛到梯度较大的点
- 复合采集函数同时优化能量和梯度，加速收敛

### 3. 局部极小值避免策略

**问题**：优化可能陷入局部极小值（梯度为零但能量不是全局最低）

**解决方案**：
```python
# 当检测到梯度接近零时，触发验证
if gradient_norm < threshold:
    # 1. 在附近添加小扰动 (±0.1 Å)
    # 2. 执行 extra_steps 额外探索
    # 3. 如果扰动后找到梯度更小的点，说明未收敛
    # 4. 继续优化直至找到真正的极小值
```

**配置参数**：
- `hybrid.verify_local_minimum`: 是否启用验证，默认 true
- `hybrid.verify_extra_steps`: 验证额外步数，默认 3

### 4. 详细数据输出

**每步迭代输出内容**：
- 能量 (Hartree)
- 梯度范数
- 坐标 (3×N_atoms)
- 与上一步的位移
- 梯度矩阵 (3×N_atoms)
- 时间戳

**输出格式**：
- JSON: 完整历史记录，便于后续分析
- CSV: 表格格式，便于 Excel 查看
- XYZ: 分子轨迹，便于可视化软件打开

**输出目录结构**：
```
output/
├── {method}_*.json           # 优化历史
├── {method}_trajectory_*.xyz # 优化轨迹
├── {method}_details_*.json   # 详细信息
├── plots/
│   ├── {method}_energy.png   # 能量收敛图
│   ├── {method}_gradient.png # 梯度收敛图
│   └── {method}_combined.png # 组合图
└── structures/
    ├── {method}_initial.xyz  # 初始结构
    ├── {method}_final.xyz    # 最终结构
    └── {method}_comparison.png
```

### 5. 可视化增强

**3D 分子结构**：
- CPK 配色方案（C=黑，O=红，H=白）
- 化学键自动检测
- 原子标签显示
- 多视角对比

**收敛曲线**：
- 大字体（14pt+），便于论文使用
- 高分辨率（300 DPI）
- 对数坐标（梯度图）
- 多方法对比图

### 6. 模块化与可扩展性

**设计原则**：
- 优化器基类定义统一接口
- 各组件独立，可单独替换
- 配置驱动，无需修改代码

**扩展新 AI 方法示例**：
```python
# 1. 创建新的优化器
from optimizers.base import BaseOptimizer

class NeuralNetworkOptimizer(BaseOptimizer):
    def optimize(self, molecule, calculator):
        # 实现神经网络优化逻辑
        pass

# 2. 在混合策略中使用
from optimizers.neural_net import NeuralNetworkOptimizer
# 替换 HybridOptimizer 中的 GPR 模型
```

---

## 快速开始

### 安装依赖

```bash
cd new_method
pip install -r requirements.txt
```

### 运行示例

```bash
# L-BFGS 基准测试
python main.py --method lbfgs --molecule ethanol

# L-BFGS+GPR 混合优化
python main.py --method hybrid --molecule ethanol --perturb 0.5

# 对比实验（同时运行两种方法）
python run_comparison.py --smiles CCO --perturb 0.5

# 运行测试
python test_project.py
```

### 配置调整

编辑 `new_method/config/default_config.yaml`：

```yaml
# 调整混合策略参数
hybrid:
  lbfgs_steps: 5           # L-BFGS 步数
  gpr_steps: 2             # GPR 步数
  selection_metric: "combined"

# 调整 GPR 参数
gpr:
  n_init: 10               # 初始采样点数
  local_radius: 0.5        # 搜索半径
  lambda_grad: 0.1         # 梯度惩罚权重
```

---

## 实验结果（示例）

### 乙醇分子优化（RHF/cc-pVDZ）

| 方法 | 初始能量 | 最优能量 | 迭代次数 | 收敛 |
|------|---------|---------|---------|------|
| L-BFGS (无扰动) | -154.0803 | -154.0927 | 42 | ✓ |
| L-BFGS (扰动 0.5) | -153.0956 | -153.9109 | 64 | ✓ |
| L-BFGS+GPR (测试中) | - | - | - | - |

---

## 技术栈

| 类别 | 工具/库 |
|------|--------|
| 量子化学 | PySCF |
| 分子处理 | RDKit, ASE |
| 优化算法 | SciPy (L-BFGS-B) |
| 机器学习 | GPy, scikit-learn |
| 可视化 | Matplotlib |
| 配置管理 | PyYAML |

---

## 文档说明

| 文档 | 位置 | 说明 |
|------|------|------|
| 项目总览 | `README.md` (本文件) | 项目整体介绍 |
| new_method 详细文档 | `new_method/README.md` | 新版本使用说明 |
| 使用指南 | `new_method/USAGE.md` | 详细配置和 API 说明 |

---

## 项目维护

**维护者**: 刘喆 (Liu Zhe)

**联系方式**:
- 邮箱：3266048598@qq.com
- 项目目录：`/mnt/e/wsl_dir/L-BFGS-GPR/`

**版本历史**:
- v1.0 (old_method): 初始研究版本，验证基本概念
- v2.0 (new_method): 重构优化版本，实现混合策略

---

## 许可证

本项目用于学术研究和教学目的。

---

## 致谢

感谢所有为本项目提供支持和帮助的人员。

---

*最后更新：2024 年*
