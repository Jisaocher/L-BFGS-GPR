# L-BFGS-GPR 混合优化项目

分子几何构型优化 - L-BFGS 与 GPR 混合策略

## 项目概述

本项目实现了基于 L-BFGS（拟牛顿法）和 GPR（高斯过程回归）的分子几何构型优化方法。

### 核心特性

1. **混合优化策略**：每 m 步 L-BFGS 后加上 n 步 GPR 预测，选择表现最好的迭代结果
2. **梯度增强 GPR**：同时建模能量和梯度，提升样本效率
3. **详细输出**：每步迭代保存能量、梯度、坐标、位移等完整信息
4. **3D 可视化**：分子结构 3D 展示 + 能量/梯度收敛曲线
5. **模块化设计**：易于扩展其他 AI 优化策略

### 优化策略对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| L-BFGS | 局部收敛快，稳定可靠 | 对初始点敏感，易陷入局部极小 |
| L-BFGS+GPR | 兼顾全局探索，可能找到更好的极小点 | 计算开销较大，参数敏感 |

## 快速开始

### 安装依赖

```bash
cd /mnt/e/wsl_dir/L-BFGS-GPR/new_method
pip install -r requirements.txt
```

### 运行示例

```bash
# L-BFGS 基准测试
python main.py --method lbfgs --molecule ethanol

# L-BFGS+GPR 混合优化
python main.py --method hybrid --molecule ethanol --perturb 0.5

# 对比实验
python run_comparison.py --smiles CCO --perturb 0.5
```

### 测试项目

```bash
python test_project.py
```

## 项目结构

```
new_method/
├── core/                    # 核心数据类
│   ├── molecule.py         # 分子结构、迭代历史
│   └── calculator.py       # 量子化学计算接口 (PySCF)
├── optimizers/              # 优化器
│   ├── base.py             # 优化器基类
│   ├── lbfgs.py            # L-BFGS 优化器
│   └── hybrid.py           # L-BFGS+GPR 混合优化器
├── models/                  # 机器学习模型
│   ├── gpr_base.py         # GPR 基类
│   └── gradient_gpr.py     # 梯度增强 GPR (GPy)
├── visualization/           # 可视化
│   ├── structure3d.py      # 3D 分子结构
│   └── plots.py            # 能量/梯度图表
├── utils/                   # 工具
│   ├── io_utils.py         # 数据输出
│   └── converters.py       # 坐标转换
├── config/                  # 配置
│   └── default_config.yaml
├── main.py                  # 主程序
├── run_comparison.py        # 对比脚本
└── test_project.py          # 测试脚本
```

## 混合策略设计

### 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      开始优化                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  初始采样：生成 n_init 个样本点，训练 GPR 模型                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  L-BFGS 阶段：执行 m 步 (默认 5 步) L-BFGS 优化                    │
│  - 每步记录能量、梯度、坐标、位移                              │
│  - 将新点添加到 GPR 训练集                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  GPR 阶段：执行 n 步 (默认 2 步) GPR 预测                          │
│  - 使用复合采集函数 EI - λ·||∇E||选择新点                      │
│  - 计算真实能量和梯度                                         │
│  - 更新 GPR 模型                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  选择最佳：从 m+n 步中选择能量/梯度综合最优的点                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  收敛检查：梯度范数 < 阈值？                                   │
│  - 是 → 验证局部极小值 → 收敛                                 │
│  - 否 → 继续下一轮                                            │
└─────────────────────────────────────────────────────────────┘
```

### 采集函数

使用复合采集函数平衡能量下降和梯度减小：

```
Acq(x) = EI(x) - λ·||∇E_pred(x)||
```

其中：
- `EI(x)`: 期望提升（Expected Improvement）
- `λ`: 梯度惩罚权重（默认 0.1）
- `||∇E_pred(x)||`: 预测梯度范数

### 局部极小值验证

为避免陷入局部极小，设计验证策略：
1. 当梯度范数 < 阈值时，触发验证
2. 在当前位置添加小扰动（±0.1 Å）
3. 执行 extra_steps（默认 3 步）额外探索
4. 如果扰动后找到梯度更小的点，说明未收敛

## 输出说明

### 输出文件

| 文件 | 说明 |
|------|------|
| `{method}_*.json` | 优化历史（JSON 格式） |
| `{method}_trajectory_*.xyz` | 优化轨迹（XYZ 格式） |
| `{method}_details_*.json` | 详细迭代信息（梯度矩阵、位移等） |
| `plots/*_energy.png` | 能量收敛曲线 |
| `plots/*_gradient.png` | 梯度收敛曲线 |
| `plots/*_combined.png` | 组合图表 |
| `structures/*_initial.xyz` | 初始结构 |
| `structures/*_final.xyz` | 最终结构 |
| `structures/*_comparison.png` | 结构对比图 |

### 输出示例

```
输出目录：./output/
├── lbfgs_20240101_120000.json
├── lbfgs_trajectory_20240101_120000.xyz
├── lbfgs_details_20240101_120000.json
├── plots/
│   ├── lbfgs_plot_energy.png
│   ├── lbfgs_plot_gradient.png
│   └── lbfgs_plot_combined.png
└── structures/
    ├── lbfgs_initial.xyz
    ├── lbfgs_final.xyz
    └── lbfgs_comparison.png
```

## 配置参数

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hybrid.lbfgs_steps` | 5 | 每轮 L-BFGS 步数 (m) |
| `hybrid.gpr_steps` | 2 | 每轮 GPR 步数 (n) |
| `hybrid.selection_metric` | "combined" | 选择标准 (energy/gradient/combined) |
| `gpr.n_init` | 10 | 初始采样点数 |
| `gpr.local_radius` | 0.5 | 局部搜索半径 (Å) |
| `gpr.xi` | 0.01 | EI 探索参数 |
| `gpr.lambda_grad` | 0.1 | 梯度惩罚权重 |
| `optimizer.convergence_threshold` | 1e-5 | 梯度收敛阈值 |
| `optimizer.max_iterations` | 200 | 最大迭代次数 |

## 扩展其他 AI 方法

项目采用模块化设计，易于扩展：

```python
# 1. 创建新的优化器
from optimizers.base import BaseOptimizer

class MyAIOptimizer(BaseOptimizer):
    def optimize(self, molecule, calculator):
        # 实现优化逻辑
        pass

# 2. 在混合策略中使用
from optimizers.my_ai import MyAIOptimizer
# 替换 HybridOptimizer 中的 GPR 模型
```

## 依赖

- Python >= 3.8
- pyscf >= 2.0.0
- rdkit >= 2022.0.0
- scipy >= 1.8.0
- scikit-learn >= 1.0.0
- GPy >= 1.10.0
- matplotlib >= 3.5.0
- numpy >= 1.20.0
- pyyaml >= 6.0

## 使用文档

详细使用说明请参阅 [USAGE.md](./USAGE.md)

## 许可证

本项目用于学术研究和教学目的。

## 联系方式

如有问题或建议，请提交 Issue 
项目联系人：LiuZhe,3266048598@qq.com。
