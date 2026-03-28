# L-BFGS-GPR 混合优化项目使用文档

## 项目概述

本项目实现了基于 L-BFGS 和 GPR（高斯过程回归）的分子几何构型优化方法，支持两种优化策略：
- **L-BFGS**：传统拟牛顿法，作为基准方法
- **L-BFGS+GPR**：混合优化策略，结合 L-BFGS 的局部快速收敛和 GPR 的全局探索能力

## 项目结构

```
new_method/
├── core/                    # 核心数据类
│   ├── molecule.py         # 分子结构、迭代历史类
│   └── calculator.py       # 量子化学计算接口
├── optimizers/              # 优化器实现
│   ├── base.py             # 优化器基类
│   ├── lbfgs.py            # L-BFGS 优化器
│   └── hybrid.py           # L-BFGS+GPR 混合优化器
├── models/                  # 机器学习模型
│   ├── gpr_base.py         # GPR 基类
│   └── gradient_gpr.py     # 梯度增强 GPR
├── visualization/           # 可视化模块
│   ├── structure3d.py      # 3D 分子结构可视化
│   └── plots.py            # 能量/梯度图表
├── utils/                   # 工具函数
│   ├── io_utils.py         # 输入输出工具
│   └── converters.py       # 坐标转换工具
├── config/                  # 配置文件
│   └── default_config.yaml # 默认配置
├── main.py                  # 主程序入口
├── run_comparison.py        # 对比运行脚本
└── test_project.py          # 测试脚本
```

## 安装依赖

```bash
cd /mnt/e/wsl_dir/L-BFGS-GPR/new_method
pip install -r requirements.txt
```

### 依赖说明

- `pyscf`: 量子化学计算
- `rdkit`: 分子结构生成
- `scipy`: L-BFGS 优化器
- `scikit-learn`: 简单 GPR 模型
- `GPy`: 梯度增强 GPR 模型
- `matplotlib`: 可视化
- `pyyaml`: 配置文件解析
- `ase`: 分子结构处理（可选）

## 快速开始

### 1. 运行 L-BFGS 优化（基准方法）

```bash
python main.py --method lbfgs --molecule ethanol
```

### 2. 运行 L-BFGS+GPR 混合优化

```bash
python main.py --method hybrid --molecule ethanol
```

### 3. 运行对比实验

```bash
python run_comparison.py --smiles CCO --perturb 0.5
```

## 命令行参数

### main.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 优化方法 (`lbfgs` 或 `hybrid`) | `lbfgs` |
| `--molecule` | 分子名称 (`ethanol`, `water`, `methane`) | `ethanol` |
| `--smiles` | SMILES 字符串（覆盖 `--molecule`） | `None` |
| `--perturb` | 初始扰动强度 (Å) | `0.0` |
| `--seed` | 随机种子 | `42` |
| `--config` | 配置文件路径 | `None` |
| `--output` | 输出目录 | `./output` |
| `--max-iter` | 最大迭代次数 | `200` |
| `--threshold` | 收敛阈值 | `1e-5` |

### run_comparison.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--smiles` | 分子 SMILES | `CCO` |
| `--perturb` | 初始扰动强度 (Å) | `0.0` |
| `--seed` | 随机种子 | `42` |
| `--config` | 配置文件路径 | `None` |
| `--output` | 输出目录 | `./output/comparison_<timestamp>` |

## 配置说明

编辑 `config/default_config.yaml` 调整参数：

### 分子设置
```yaml
molecule:
  smiles: "CCO"           # 乙醇 SMILES
  seed: 42                # 随机种子
  perturb_strength: 0.0   # 初始扰动强度 (Å)
```

### 计算方法
```yaml
calculation:
  basis: "cc-pvdz"        # 基组
  method: "RHF"           # 量子化学方法
  unit: "angstrom"        # 坐标单位
```

### 混合策略设置
```yaml
hybrid:
  lbfgs_steps: 5          # 每轮 L-BFGS 步数 (m)
  gpr_steps: 2            # 每轮 GPR 步数 (n)
  selection_metric: "combined"  # 选择标准
  verify_local_minimum: true    # 验证局部极小值
  verify_extra_steps: 3         # 确认极小值额外步数
```

### GPR 设置
```yaml
gpr:
  n_init: 10              # 初始采样点数
  local_radius: 0.5       # 局部搜索半径 (Å)
  xi: 0.01                # EI 采集函数探索参数
  lambda_grad: 0.1        # 梯度惩罚权重
```

## 输出说明

### 输出目录结构

```
output/
├── lbfgs_20240101_120000.json      # 优化历史（JSON）
├── lbfgs_trajectory_*.xyz          # 优化轨迹（XYZ 格式）
├── lbfgs_details_*.json            # 详细迭代信息
├── plots/
│   ├── lbfgs_plot_energy.png       # 能量收敛图
│   ├── lbfgs_plot_gradient.png     # 梯度收敛图
│   └── lbfgs_plot_combined.png     # 组合图
└── structures/
    ├── lbfgs_initial.xyz           # 初始结构
    ├── lbfgs_final.xyz             # 最终结构
    └── lbfgs_comparison.png        # 结构对比图
```

### 输出文件格式

#### JSON 历史文件
```json
{
  "iterations": [
    {
      "iteration": 0,
      "energy": -154.08034759,
      "gradient_norm": 0.0746305,
      "coords": [...],
      "gradient": [...],
      "displacement": [...]
    },
    ...
  ],
  "converged": true,
  "convergence_iteration": 41,
  "statistics": {
    "total_iterations": 42,
    "initial_energy": -154.08034759,
    "final_energy": -154.09271898,
    "energy_improvement": 0.01237139
  }
}
```

#### XYZ 轨迹文件
```
9
Iteration 0, Energy=-154.08034759, |grad|=0.074630
C    -0.953455    0.047804    0.042499
C     0.487919   -0.321539   -0.189050
...
```

## 混合优化策略说明

### 工作流程

1. **初始采样**：在局部区域内生成 n_init 个样本点，训练 GPR 模型
2. **L-BFGS 阶段**：执行 m 步（默认 5 步）L-BFGS 优化
3. **GPR 阶段**：执行 n 步（默认 2 步）GPR 预测
4. **选择最佳**：从 m+n 步中选择表现最好的点
5. **验证收敛**：检查是否达到收敛标准
6. **循环**：重复步骤 2-5 直至收敛

### 采集函数

使用复合采集函数：
```
Acq(x) = EI(x) - λ·||∇E_pred(x)||
```

其中：
- `EI(x)`: 期望提升（Expected Improvement）
- `λ`: 梯度惩罚权重
- `||∇E_pred(x)||`: 预测梯度范数

该函数鼓励模型推荐低能量且小梯度的点。

### 局部极小值验证

当检测到梯度接近零的点时，执行额外验证：
1. 在附近添加小扰动
2. 计算扰动后的梯度
3. 如果扰动后梯度更小，说明可能不是真正的极小值
4. 继续优化直至找到真正的极小值

## 扩展其他 AI 方法

项目采用模块化设计，易于扩展其他 AI 优化策略：

### 1. 创建新的优化器

```python
# optimizers/my_ai_method.py
from optimizers.base import BaseOptimizer

class MyAIOptimizer(BaseOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.name = "MyAI"
    
    def optimize(self, molecule, calculator):
        # 实现优化逻辑
        pass
    
    def step(self, coords_flat):
        # 实现单步优化
        pass
```

### 2. 在 hybrid.py 中使用

```python
from optimizers.my_ai_method import MyAIOptimizer

# 在 HybridOptimizer 中替换 GPR 模型
self.ai_model = MyAIOptimizer(config)
```

### 3. 更新配置

```yaml
# config/default_config.yaml
my_ai:
  param1: value1
  param2: value2
```

## 常见问题

### Q: 优化不收敛怎么办？

A: 尝试以下调整：
- 增加 `max_iterations`
- 放宽 `convergence_threshold`
- 调整 GPR 的 `local_radius` 和 `n_init`
- 检查初始结构是否合理

### Q: GPR 训练失败怎么办？

A: 可能原因：
- 训练点太少：增加 `n_init`
- 核函数不合适：尝试不同的 `kernel_type`
- 数值不稳定：调整 `noise_variance`

### Q: 如何加速计算？

A: 建议：
- 使用较小的基组（如 `sto-3g`）进行测试
- 减少 `max_iterations`
- 减少 GPR 的 `n_init` 和 `gpr_steps`

## 性能基准

以下为乙醇分子（RHF/cc-pVDZ）的典型结果：

| 方法 | 初始能量 | 最优能量 | 迭代次数 | 收敛 |
|------|---------|---------|---------|------|
| L-BFGS (无扰动) | -154.0803 | -154.0927 | 42 | 是 |
| L-BFGS (扰动 0.5) | -153.0956 | -153.9109 | 64 | 是 |
| L-BFGS+GPR (测试中) | - | - | - | - |

## 参考文献

1. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization.
2. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
3. Sun, S., et al. (2019). Gradient-enhanced Gaussian process regression for molecular potential energy surfaces.
