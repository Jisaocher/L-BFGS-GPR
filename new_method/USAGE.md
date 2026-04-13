# L-BFGS-GPR 混合优化项目使用文档

## 项目概述

本项目实现了基于 L-BFGS 和**梯度预测 GPR**的分子几何构型优化方法。

**核心思想**：
- **L-BFGS**：传统拟牛顿法，作为主要优化引擎
- **梯度预测 GPR**：直接预测梯度向量，引导优化向梯度下降方向进行
- **综合选择策略**：每轮综合考虑能量和梯度，选择最优起点

**AI 方法说明**：
- 项目**仅支持 `gradient_predicting` 方法**（梯度预测 GPR）
- 该方法**直接预测梯度向量**，而非能量
- 预测方向：**向梯度下降的方向**（梯度更小的区域）
- 旧版方法（`simple`、`gradient`、`random_forest`）已移除，因为它们预测能量而非梯度

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
# 无扰动初始结构
python main.py --method lbfgs --molecule ethanol

# 添加扰动（0.5 Å）
python main.py --method lbfgs --molecule ethanol --perturb 0.5
```

### 2. 运行 L-BFGS+梯度预测 GPR 混合优化

**默认配置（推荐）**：
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

**自定义权重配置**：
```bash
# 编辑 config/default_config.yaml，调整 selection_weights
# selection_weights:
#   energy_weight: 0.3    # 能量权重
#   gradient_weight: 0.7  # 梯度权重（推荐 > 能量权重）

python main.py --method hybrid --molecule ethanol --perturb 0.1
```

### 3. 运行对比实验

```bash
# L-BFGS vs L-BFGS+梯度预测 GPR 对比
python run_comparison.py --smiles CCO --perturb 0.1
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
| `--max-iter` | 最大迭代次数 | `300` |
| `--threshold` | 收敛阈值 | `5e-4` |

---

## 完整执行方式

### 方法 1：L-BFGS 基准方法

**用途**：作为对比基准，验证 AI 方法的有效性

```bash
# 乙醇分子，无扰动
python main.py --method lbfgs --molecule ethanol

# 乙醇分子，添加 0.1 Å 扰动
python main.py --method lbfgs --molecule ethanol --perturb 0.1

# 自定义收敛阈值
python main.py --method lbfgs --molecule ethanol --threshold 1e-3
```

**输出文件**：
```
output/
├── lbfgs_YYYYMMDD_HHMMSS.json
├── lbfgs_trajectory_*.xyz
├── lbfgs_details_*.json
...
```

---

### 方法 2：L-BFGS+简单 GPR（默认 AI 方法）

**用途**：快速验证 AI 方法可行性，训练速度中等

**配置文件** (`config/default_config.yaml`)：
```yaml
gpr:
  type: "simple"             # AI 方法类型
  n_init: 3                  # 初始采样点数
  local_radius: 0.1          # 搜索半径 (Å)
  max_training_points: 15    # 最大训练点数
  use_gpr: true              # 启用 GPR

hybrid:
  lbfgs_steps: 5             # 每轮 L-BFGS 步数
  gpr_steps: 1               # 每轮 GPR 步数
```

**运行命令**：
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

**输出文件**：
```
output/
├── hybrid_gpr_YYYYMMDD_HHMMSS.json    # 注意后缀：gpr
├── hybrid_gpr_trajectory_*.xyz
├── hybrid_gpr_details_*.json
...
```

---

### 方法 3：L-BFGS+梯度 GPR

**用途**：精确建模，研究 GPR 潜力（最慢）

**配置文件**：
```yaml
gpr:
  type: "gradient"           # 梯度 GPR
  n_init: 5                  # 需要更多初始点
  local_radius: 0.1
  max_training_points: 20    # 更多训练数据

hybrid:
  lbfgs_steps: 5
  gpr_steps: 1
```

**运行命令**：
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

**输出文件**：
```
output/
├── hybrid_ggpr_YYYYMMDD_HHMMSS.json   # 注意后缀：ggpr
├── hybrid_ggpr_trajectory_*.xyz
├── hybrid_ggpr_details_*.json
...
```

---

### 方法 4：L-BFGS+随机森林（推荐）

**用途**：**推荐用于快速收敛**，训练和预测都很快

**配置文件**：
```yaml
gpr:
  type: "random_forest"      # 随机森林
  n_init: 3
  local_radius: 0.1
  max_training_points: 15

# 随机森林专用参数
random_forest:
  n_estimators: 100          # 树的数量
  max_depth: 10              # 树的最大深度
  min_samples_split: 2       # 分裂所需最小样本数
  min_samples_leaf: 1        # 叶子节点最小样本数

hybrid:
  lbfgs_steps: 5
  gpr_steps: 1
```

**运行命令**：
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

**输出文件**：
```
output/
├── hybrid_rf_YYYYMMDD_HHMMSS.json     # 注意后缀：rf
├── hybrid_rf_trajectory_*.xyz
├── hybrid_rf_details_*.json
...
```

---

### 方法 5：对比实验（自动运行所有方法）

**用途**：一次性运行 L-BFGS 和所有 AI 方法，生成对比报告

**运行命令**：
```bash
python run_comparison.py --smiles CCO --perturb 0.1
```

**输出目录**：
```
output/comparison_YYYYMMDD_HHMMSS/
├── comparison_summary_*.json    # 对比总结
├── lbfgs/                       # L-BFGS 结果
├── hybrid_gpr/                  # 简单 GPR 结果
├── hybrid_rf/                   # 随机森林结果
└── plots/                       # 对比图表
```

---

## AI 方法说明

### 唯一支持的 AI 方法：梯度预测 GPR

项目**仅支持 `gradient_predicting` 方法**，因为：

| 方法 | 训练目标 | 预测目标 | 是否符合优化目标 | 状态 |
|------|---------|---------|----------------|------|
| `simple` (简单 GPR) | 能量 E | 能量 E | ❌ 能量最低≠梯度为零 | 已移除 |
| `gradient` (梯度 GPR) | 能量 + 梯度 | 能量 + 梯度 | ⚠️ 目标不纯粹 | 已移除 |
| `random_forest` | 能量 E | 能量 E | ❌ 能量最低≠梯度为零 | 已移除 |
| **`gradient_predicting`** | **梯度 ∇E** | **梯度 ∇E** | ✅ **直接预测梯度** | **唯一支持** |

### 核心优势

**梯度预测 GPR 的核心优势**：
1. ✅ **训练目标明确**：直接使用梯度向量作为训练目标
2. ✅ **预测方向正确**：向梯度下降的方向预测（梯度更小的区域）
3. ✅ **采集函数合理**：使用预测梯度范数作为采集函数
4. ✅ **物理意义清晰**：最小化梯度范数 = 接近稳定构型

### 权重配置建议

**综合选择策略**：
```python
score = energy_weight * E_normalized + gradient_weight * ||g||_normalized
```

**推荐配置**：
```yaml
selection_weights:
  energy_weight: 0.3       # 能量权重（0-1）
  gradient_weight: 0.7     # 梯度权重（0-1，推荐 > 能量权重）
```

**原因**：
- 分子几何优化的**目标是梯度为零**，不是能量最低
- 但**纯梯度选择**可能选择能量过高但梯度小的点
- **综合选择**平衡两者，避免极端情况

**特殊情况**：
```yaml
# 纯梯度选择（激进）
selection_weights:
  energy_weight: 0.0
  gradient_weight: 1.0

# 纯能量选择（不推荐）
selection_weights:
  energy_weight: 1.0
  gradient_weight: 0.0

# 平衡选择（折中）
selection_weights:
  energy_weight: 0.5
  gradient_weight: 0.5
```

---

## 配置文件说明

编辑 `config/default_config.yaml` 调整参数：

### 分子设置
```yaml
molecule:
  smiles: "CCO"              # 乙醇 SMILES
  seed: 42                   # 随机种子
  perturb_strength: 0.1      # 初始扰动强度 (Å)
```

### 计算方法
```yaml
calculation:
  basis: "cc-pvdz"           # 基组
  method: "RHF"              # 量子化学方法
  unit: "angstrom"           # 坐标单位
```

### 优化器设置
```yaml
optimizer:
  max_iterations: 300        # 最大迭代次数
  convergence_threshold: 5e-4  # 梯度收敛阈值
  verbose: true              # 输出详细信息
```

### AI 方法设置（关键）
```yaml
gpr:
  type: "random_forest"      # AI 方法类型：simple/gradient/random_forest
  n_init: 3                  # 初始采样点数
  local_radius: 0.1          # 局部搜索半径 (Å)
  max_training_points: 15    # 最大训练点数（滑动窗口）
  use_gpr: true              # 是否启用 AI 方法

# 随机森林专用参数（当 type="random_forest" 时）
random_forest:
  n_estimators: 100          # 树的数量
  max_depth: 10              # 树的最大深度
  min_samples_split: 2       # 分裂所需最小样本数
  min_samples_leaf: 1        # 叶子节点最小样本数
```

### 混合策略设置
```yaml
hybrid:
  lbfgs_steps: 5             # 每轮 L-BFGS 步数 (m)
  gpr_steps: 1               # 每轮 AI 步数 (n)
  selection_metric: "energy" # 选择标准：energy/gradient/combined
  verify_local_minimum: false  # 验证局部极小值（关闭以加快）
```

## 输出说明

### 输出目录结构

**L-BFGS 基准方法**：
```
output/
├── lbfgs_20240101_120000.json      # 优化历史（JSON）
├── lbfgs_trajectory_*.xyz          # 优化轨迹（XYZ 格式）
├── lbfgs_details_*.json            # 详细迭代信息
├── plots/
│   ├── lbfgs_energy.png            # 能量收敛图
│   ├── lbfgs_gradient.png          # 梯度收敛图
│   └── lbfgs_combined.png          # 组合图
└── structures/
    ├── lbfgs_initial.xyz           # 初始结构
    ├── lbfgs_final.xyz             # 最终结构
    └── lbfgs_comparison.png        # 结构对比图
```

**L-BFGS+简单 GPR**（`gpr.type = "simple"`）：
```
output/
├── hybrid_gpr_20240101_120000.json # AI 方法后缀：gpr
├── hybrid_gpr_trajectory_*.xyz
├── hybrid_gpr_details_*.json
├── plots/
│   ├── hybrid_gpr_energy.png
│   ├── hybrid_gpr_gradient.png
│   └── hybrid_gpr_combined.png
└── structures/
    ├── hybrid_gpr_initial.xyz
    ├── hybrid_gpr_final.xyz
    └── hybrid_gpr_comparison.png
```

**L-BFGS+梯度 GPR**（`gpr.type = "gradient"`）：
```
output/
├── hybrid_ggpr_20240101_120000.json # AI 方法后缀：ggpr
├── hybrid_ggpr_trajectory_*.xyz
├── hybrid_ggpr_details_*.json
├── plots/
│   ├── hybrid_ggpr_energy.png
│   ├── hybrid_ggpr_gradient.png
│   └── hybrid_ggpr_combined.png
└── structures/
    ├── hybrid_ggpr_initial.xyz
    ├── hybrid_ggpr_final.xyz
    └── hybrid_ggpr_comparison.png
```

**L-BFGS+随机森林**（`gpr.type = "random_forest"`）：
```
output/
├── hybrid_rf_20240101_120000.json   # AI 方法后缀：rf
├── hybrid_rf_trajectory_*.xyz
├── hybrid_rf_details_*.json
├── plots/
│   ├── hybrid_rf_energy.png
│   ├── hybrid_rf_gradient.png
│   └── hybrid_rf_combined.png
└── structures/
    ├── hybrid_rf_initial.xyz
    ├── hybrid_rf_final.xyz
    └── hybrid_rf_comparison.png
```

### AI 方法文件命名规则

| 文件类型 | L-BFGS | 简单 GPR | 梯度 GPR | 随机森林 |
|---------|--------|---------|---------|---------|
| JSON 历史 | `lbfgs_*.json` | `hybrid_gpr_*.json` | `hybrid_ggpr_*.json` | `hybrid_rf_*.json` |
| XYZ 轨迹 | `lbfgs_trajectory_*.xyz` | `hybrid_gpr_trajectory_*.xyz` | `hybrid_ggpr_trajectory_*.xyz` | `hybrid_rf_trajectory_*.xyz` |
| 详细数据 | `lbfgs_details_*.json` | `hybrid_gpr_details_*.json` | `hybrid_ggpr_details_*.json` | `hybrid_rf_details_*.json` |
| 能量图 | `lbfgs_energy.png` | `hybrid_gpr_energy.png` | `hybrid_ggpr_energy.png` | `hybrid_rf_energy.png` |
| 梯度图 | `lbfgs_gradient.png` | `hybrid_gpr_gradient.png` | `hybrid_ggpr_gradient.png` | `hybrid_rf_gradient.png` |
| 组合图 | `lbfgs_combined.png` | `hybrid_gpr_combined.png` | `hybrid_ggpr_combined.png` | `hybrid_rf_combined.png` |
| 结构对比 | `lbfgs_comparison.png` | `hybrid_gpr_comparison.png` | `hybrid_ggpr_comparison.png` | `hybrid_rf_comparison.png` |

| AI 方法 | 配置值 | 文件后缀 | 示例文件名 |
|---------|--------|---------|-----------|
| 简单 GPR | `simple` | `gpr` | `hybrid_gpr_*.json` |
| 梯度 GPR | `gradient` | `ggpr` | `hybrid_ggpr_*.json` |
| 随机森林 | `random_forest` | `rf` | `hybrid_rf_*.json` |
| 神经网络 | `neural_network` | `nn` | `hybrid_nn_*.json` |
| 纯 L-BFGS | - | 无 | `lbfgs_*.json` |

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

---

## 3D 分子结构可视化

优化完成后，可以使用 `draw_structure3D.py` 批量生成 HTML 格式的 3D 分子结构图。

### 执行方式

**方法 1：使用默认路径（推荐）**
```bash
# 自动从配置文件读取 output 目录
python draw_structure3D.py
```

**方法 2：指定 structures 目录**
```bash
# 直接指定 structures 目录路径
python draw_structure3D.py /path/to/output/structures
```

### 输出文件

脚本会将 `structures` 目录下所有 `.xyz` 文件转换为同名的 `.html` 文件：

```
output/
├── structures/
│   ├── lbfgs_initial.xyz         ← 原始 XYZ 文件
│   ├── lbfgs_initial.html        ← 生成的 HTML
│   ├── lbfgs_final.xyz
│   ├── lbfgs_final.html
│   ├── hybrid_gpr_initial.xyz
│   ├── hybrid_gpr_initial.html
│   ├── hybrid_gpr_final.xyz
│   └── hybrid_gpr_final.html
```

### HTML 文件说明

- **标题**：使用文件名（如 "lbfgs_final"）
- **显示**：球棍模型（原子=球体，化学键=棍子）
- **交互**：
  - 旋转：鼠标拖动
  - 缩放：鼠标滚轮
  - 平移：右键拖动

### 示例输出

```bash
$ python draw_structure3D.py

使用配置文件定义的目录：./output/structures

找到 4 个 XYZ 文件:
  - hybrid_gpr_final.xyz
  - hybrid_gpr_initial.xyz
  - lbfgs_final.xyz
  - lbfgs_initial.xyz

开始生成 HTML 文件...
分子结构图已保存：./output/structures/hybrid_gpr_final.html
分子结构图已保存：./output/structures/hybrid_gpr_initial.html
分子结构图已保存：./output/structures/lbfgs_final.html
分子结构图已保存：./output/structures/lbfgs_initial.html

完成！生成了 4 个 HTML 文件:
  ✓ hybrid_gpr_final.html
  ✓ hybrid_gpr_initial.html
  ✓ lbfgs_final.html
  ✓ lbfgs_initial.html

提示：用浏览器打开 HTML 文件可查看交互式 3D 分子结构。
```

### 依赖

需要安装 `py3Dmol`：
```bash
pip install py3Dmol
```

---

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
