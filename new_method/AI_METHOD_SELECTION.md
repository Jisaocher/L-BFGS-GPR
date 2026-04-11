# AI 方法选择功能说明

## 功能概述

现在 L-BFGS+AI 混合优化支持通过配置文件选择不同的人工智能方法：

| AI 方法 | 配置值 | 特点 |
|---------|--------|------|
| 简单 GPR | `simple` | 默认，只能量预测，速度快 |
| 梯度 GPR | `gradient` | 能量 + 梯度联合建模，精确但慢 |
| 随机森林 | `random_forest` | **推荐**，训练预测都快，鲁棒 |

## 使用方法

### 方法 1：修改配置文件

编辑 `config/default_config.yaml`：

```yaml
gpr:
  type: "random_forest"    # 改为随机森林
```

然后运行：
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

### 方法 2：保持默认（简单 GPR）

```yaml
gpr:
  type: "simple"           # 默认值
```

## 输出示例

### 使用随机森林
```
======================================================================
L-BFGS+Random Forest 混合优化开始
======================================================================
L-BFGS 步数 (m): 5
Random Forest 步数 (n): 1
选择标准：energy
...
```

### 使用简单 GPR
```
======================================================================
L-BFGS+Simple GPR 混合优化开始
======================================================================
L-BFGS 步数 (m): 5
Simple GPR 步数 (n): 1
选择标准：energy
...
```

### 使用梯度 GPR
```
======================================================================
L-BFGS+Gradient GPR 混合优化开始
======================================================================
L-BFGS 步数 (m): 5
Gradient GPR 步数 (n): 1
选择标准：energy
...
```

## 实现细节

### 1. 新增文件

- `models/random_forest.py`：随机森林模型实现

### 2. 修改文件

- `optimizers/hybrid.py`：添加 AI 方法选择逻辑
- `config/default_config.yaml`：添加 `gpr.type` 参数和随机森林配置
- `README.md`：更新使用说明

### 3. 核心代码

```python
# optimizers/hybrid.py 第 78-88 行
ai_method = self.config.get('gpr', {}).get('type', 'simple')

if ai_method == 'random_forest':
    from models.random_forest import RandomForestModel
    self.gpr_model = RandomForestModel(self.config, dim)
    self.ai_method_name = "Random Forest"
elif ai_method == 'gradient':
    self.gpr_model = GradientGPRModel(self.config, dim)
    self.ai_method_name = "Gradient GPR"
else:  # simple 或默认
    self.gpr_model = SimpleGPRModel(self.config, dim)
    self.ai_method_name = "Simple GPR"
```

## 推荐配置

### 快速测试（随机森林）

```yaml
gpr:
  type: "random_forest"
  n_init: 5
  local_radius: 0.1

random_forest:
  n_estimators: 100
  max_depth: 10
```

**预期性能**：
- 乙醇（cc-pvdz）：~20 分钟收敛
- 比 GPR 快约 40%

### 精确研究（梯度 GPR）

```yaml
gpr:
  type: "gradient"
  n_init: 10
  local_radius: 0.1
  max_training_points: 20
```

**预期性能**：
- 乙醇（cc-pvdz）：~40-50 分钟收敛
- 最精确但最慢

### 默认配置（简单 GPR）

```yaml
gpr:
  type: "simple"
  n_init: 3
  local_radius: 0.1
  max_training_points: 15
```

**预期性能**：
- 乙醇（cc-pvdz）：~30-35 分钟收敛
- 平衡速度与精度

## 对比实验建议

### 毕业论文实验设计

**实验 1：不同 AI 方法对比**
```bash
# 1. L-BFGS 基准
python main.py --method lbfgs --molecule ethanol --perturb 0.1

# 2. L-BFGS+Simple GPR
# 设置 gpr.type = "simple"
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 3. L-BFGS+Random Forest
# 设置 gpr.type = "random_forest"
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 4. L-BFGS+Gradient GPR
# 设置 gpr.type = "gradient"
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

**记录指标**：
- 收敛所需迭代次数
- 总计算时间
- 最终能量和梯度
- AI 方法找到更优点的次数

**预期结论**：
- 所有方法都能收敛到相同能量
- 随机森林最快，梯度 GPR 最慢
- AI 方法的引入不破坏 L-BFGS 收敛性

**实验 2：不同分子对比**
```bash
# 水分子
python main.py --method hybrid --molecule water --perturb 0.1

# 甲烷
python main.py --method hybrid --molecule methane --perturb 0.1

# 乙醇（已有）
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

## 故障排除

### 问题 1：导入错误

```
ModuleNotFoundError: No module named 'sklearn'
```

**解决**：
```bash
pip install scikit-learn
```

### 问题 2：随机森林训练慢

**原因**：树太多或树太深

**解决**：
```yaml
random_forest:
  n_estimators: 50     # 100→50
  max_depth: 5         # 10→5
```

### 问题 3：AI 方法切换后结果不同

**说明**：这是正常的，因为：
1. 不同 AI 模型的预测能力不同
2. 随机森林有随机性（固定 random_state=42 可复现）
3. 最终都会收敛到相同的 L-BFGS 极小值

## 扩展其他 AI 方法

如需添加新的 AI 方法（如神经网络）：

### 步骤 1：创建模型类

```python
# models/neural_network.py
class NeuralNetworkModel(BaseGPRModel):
    def __init__(self, config, dim):
        ...
    
    def train(self, X, y, gradients=None):
        ...
    
    def predict(self, x):
        ...
    
    def suggest_next_point(self, bounds, y_min):
        ...
```

### 步骤 2：修改 hybrid.py

```python
# optimizers/hybrid.py
elif ai_method == 'neural_network':
    from models.neural_network import NeuralNetworkModel
    self.gpr_model = NeuralNetworkModel(self.config, dim)
    self.ai_method_name = "Neural Network"
```

### 步骤 3：更新配置

```yaml
gpr:
  type: "neural_network"
```

## 总结

现在你可以：
1. ✓ 通过 `gpr.type` 参数选择 AI 方法
2. ✓ 支持 simple/gradient/random_forest 三种方法
3. ✓ 每种方法有独立的配置参数
4. ✓ 输出自动显示使用的 AI 方法名称
5. ✓ 便于对比实验和论文撰写
