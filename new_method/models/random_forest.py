"""
随机森林代理模型
使用 sklearn 的 RandomForestRegressor 实现
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor

from models.gpr_base import BaseGPRModel


class RandomForestModel(BaseGPRModel):
    """
    随机森林模型
    
    优点：
    1. 实现简单，无需调参
    2. 训练和预测速度快
    3. 鲁棒性强，对异常值不敏感
    4. 可通过树间方差提供不确定性估计
    """

    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化随机森林模型

        Args:
            config: 配置字典
            dim: 输入维度
        """
        super().__init__(config)
        self.name = "RandomForestModel"
        self.dim = dim

        # 随机森林参数
        rf_config = config.get('random_forest', {})
        self.n_estimators = rf_config.get('n_estimators', 100)
        self.max_depth = rf_config.get('max_depth', 10)
        self.min_samples_split = rf_config.get('min_samples_split', 2)
        self.min_samples_leaf = rf_config.get('min_samples_leaf', 1)

        # 创建模型
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1  # 并行计算
        )
        
        self.bounds = None

    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """设置变量边界"""
        self.bounds = bounds

    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练随机森林模型

        Args:
            X: 输入坐标 (n_samples, dim)
            y: 能量值 (n_samples,)
            gradients: 梯度（随机森林不使用，忽略）
        """
        if X.shape[0] < 2:
            raise ValueError("至少需要 2 个训练点")

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量

        Args:
            x: 输入坐标

        Returns:
            mean: 预测均值
            variance: 预测方差（基于树间方差）
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        x_reshaped = x.reshape(1, -1)
        
        # 获取所有树的预测
        tree_predictions = np.array([
            tree.predict(x_reshaped)[0] for tree in self.model.estimators_
        ])
        
        mean = np.mean(tree_predictions)
        variance = np.var(tree_predictions)
        
        return mean, variance

    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        数值计算梯度（有限差分）
        
        随机森林不直接提供梯度，使用有限差分近似
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        eps = 1e-5
        gradient = np.zeros(self.dim)
        base_energy, _ = self.predict(x)

        for i in range(self.dim):
            x_plus = x.copy()
            x_plus[i] += eps
            energy_plus, _ = self.predict(x_plus)
            gradient[i] = (energy_plus - base_energy) / eps

        return gradient

    def predict_energy_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        同时预测能量和梯度

        Args:
            x: 输入坐标

        Returns:
            energy: 预测能量
            gradient: 预测梯度
            energy_var: 能量预测方差
        """
        energy, energy_var = self.predict(x)
        gradient = self.predict_gradient(x)
        return energy, gradient, energy_var

    def acquisition_function(self, x: np.ndarray,
                             y_min: float = None) -> float:
        """
        采集函数：预测值 - 不确定性激励
        
        Args:
            x: 输入坐标
            y_min: 当前最小能量

        Returns:
            acquisition_value: 采集函数值
        """
        if y_min is None and len(self.y_train) > 0:
            y_min = min(self.y_train)
        elif y_min is None:
            y_min = 0.0

        mean, variance = self.predict(x)
        uncertainty = np.sqrt(variance)
        
        # 采集函数：预测能量 - 不确定性激励
        # 鼓励探索高不确定性区域
        return mean - 0.5 * uncertainty

    def suggest_next_point(self, bounds: List[Tuple[float, float]],
                           y_min: float = None) -> np.ndarray:
        """
        建议下一个采样点
        
        通过随机采样 + 采集函数评估

        Args:
            bounds: 变量边界
            y_min: 当前最小能量

        Returns:
            x_next: 建议的下一个点
        """
        if bounds is None:
            bounds = self.bounds

        if bounds is None:
            raise ValueError("需要设置边界")

        # 随机生成候选点
        dim = len(bounds)
        n_candidates = 50
        candidates = []
        for _ in range(n_candidates):
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            candidates.append(x)
        
        # 评估每个候选点的采集函数值
        best_x = None
        best_value = np.inf
        for x in candidates:
            value = self.acquisition_function(x, y_min)
            if value < best_value:
                best_value = value
                best_x = x
        
        return best_x

    def get_confidence(self, x: np.ndarray) -> float:
        """
        获取预测置信度（方差的倒数）

        Args:
            x: 输入坐标

        Returns:
            confidence: 置信度
        """
        _, var = self.predict(x)
        return 1.0 / (var + 1e-10)
