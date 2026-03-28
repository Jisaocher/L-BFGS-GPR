"""
梯度增强 GPR 模型
使用 GPy 实现能量和梯度的联合建模
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats import norm

try:
    import GPy
    GPy_AVAILABLE = True
except ImportError:
    GPy_AVAILABLE = False
    print("Warning: GPy not available. GradientGPRModel will not work.")

from models.gpr_base import BaseGPRModel


class GradientGPRModel(BaseGPRModel):
    """
    梯度增强 GPR 模型
    
    联合建模能量和梯度，使用 GPy 的 Coregionalize 核
    """
    
    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化梯度增强 GPR 模型
        
        Args:
            config: 配置字典
            dim: 输入维度
        """
        if not GPy_AVAILABLE:
            raise ImportError("GPy is required for GradientGPRModel")
        
        super().__init__(config)
        self.name = "GradientGPRModel"
        self.dim = dim
        self.output_dim = dim + 1  # 能量 + 梯度分量
        
        # GPy 特定参数
        gpr_config = config.get('gpr', {})
        self.noise_variance = gpr_config.get('noise_variance', 1e-4)
        self.kernel_type = gpr_config.get('kernel_type', 'matern52')
        
        # 边界
        self.bounds = None
    
    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """设置变量边界"""
        self.bounds = bounds
    
    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练梯度增强 GPR 模型
        
        Args:
            X: 输入坐标 (n_samples, dim)
            y: 能量值 (n_samples,)
            gradients: 梯度 (n_samples, dim)
        """
        if X.shape[0] < 2:
            raise ValueError("至少需要 2 个训练点")
        
        n_samples = X.shape[0]
        
        # 构建联合训练数据
        # 每个样本点提供 1 个能量值和 dim 个梯度值
        # 总输出数：n_samples * (1 + dim)
        
        # 输入：重复每个样本点 (1+dim) 次
        X_train = np.repeat(X, self.output_dim, axis=0)
        
        # 输出索引
        output_indices = np.tile(np.arange(self.output_dim), n_samples)
        
        # 输出值：能量和梯度
        if gradients is not None:
            Y_train = np.zeros(n_samples * self.output_dim)
            for i in range(n_samples):
                Y_train[i * self.output_dim] = y[i]  # 能量
                Y_train[i * self.output_dim + 1:] = gradients[i]  # 梯度
        else:
            Y_train = np.repeat(y, self.output_dim)
        
        # 添加输出索引作为额外输入维度
        X_train_full = np.column_stack([X_train, output_indices])
        Y_train = Y_train.reshape(-1, 1)
        
        # 构建核函数
        # Matern52 核 × Coregionalize 核
        input_kernel = GPy.kern.Matern52(self.dim, ARD=True)
        coreg = GPy.kern.Coregionalize(input_dim=1, output_dim=self.output_dim, rank=5)
        kernel = input_kernel * coreg
        
        # 创建模型
        self.model = GPy.models.GPRegression(X_train_full, Y_train, kernel)
        self.model.Gaussian_noise.variance = self.noise_variance
        
        # 优化超参数
        try:
            self.model.optimize(messages=False, max_iters=200)
            self.is_trained = True
        except Exception as e:
            print(f"Warning: Model optimization failed: {e}")
            self.is_trained = False
    
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量
        
        Args:
            x: 输入坐标 (dim,)
        
        Returns:
            mean: 预测均值
            variance: 预测方差
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 准备输入
        x_input = np.column_stack([x.reshape(1, -1), [0]])  # 输出索引 0 表示能量
        
        # 预测
        mean, var = self.model.predict(x_input)
        
        return mean[0, 0], var[0, 0]
    
    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度
        
        Args:
            x: 输入坐标 (dim,)
        
        Returns:
            gradient: 预测梯度 (dim,)
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 对每个梯度分量进行预测
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_input = np.column_stack([x.reshape(1, -1), [i + 1]])
            mean, _ = self.model.predict(x_input)
            gradient[i] = mean[0, 0]
        
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
        复合采集函数：EI - λ·||∇E_pred||
        
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
        
        # 预测能量和梯度
        energy, gradient, var = self.predict_energy_gradient(x)
        sigma = np.sqrt(var) if var > 0 else 1e-6
        
        # 计算 EI
        if sigma > 1e-10:
            gamma = (y_min - energy - self.xi) / sigma
            ei = (y_min - energy - self.xi) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        else:
            ei = max(0, y_min - energy - self.xi)
        
        # 梯度惩罚
        grad_norm = np.linalg.norm(gradient)
        
        # 复合采集函数
        return ei - self.lambda_grad * grad_norm
    
    def suggest_next_point(self, bounds: List[Tuple[float, float]],
                           y_min: float = None) -> np.ndarray:
        """
        建议下一个采样点
        
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
        
        return self.optimize_acquisition(bounds, n_restarts=5)
    
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


class SimpleGPRModel(BaseGPRModel):
    """
    简单 GPR 模型（仅能量，无梯度）
    
    使用 scikit-learn 实现，作为对比基线
    """
    
    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化简单 GPR 模型
        
        Args:
            config: 配置字典
            dim: 输入维度
        """
        super().__init__(config)
        self.name = "SimpleGPRModel"
        self.dim = dim
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        
        self.gpr_class = GaussianProcessRegressor
        self.kernel_class = lambda: ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(1e-3)
    
    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练简单 GPR 模型
        
        Args:
            X: 输入坐标 (n_samples, dim)
            y: 能量值 (n_samples,)
            gradients: 梯度（忽略）
        """
        if X.shape[0] < 2:
            raise ValueError("至少需要 2 个训练点")
        
        kernel = self.kernel_class()
        self.model = self.gpr_class(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """预测能量"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        x_reshaped = x.reshape(1, -1)
        mean, std = self.model.predict(x_reshaped, return_std=True)
        return mean[0], std[0] ** 2
    
    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        数值计算梯度
        
        由于简单 GPR 不直接建模梯度，使用有限差分近似
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
    
    def acquisition_function(self, x: np.ndarray, y_min: float = None) -> float:
        """EI 采集函数"""
        if y_min is None and len(self.y_train) > 0:
            y_min = min(self.y_train)
        elif y_min is None:
            y_min = 0.0
        
        mean, var = self.predict(x)
        sigma = np.sqrt(var) if var > 0 else 1e-6
        
        if sigma > 1e-10:
            gamma = (y_min - mean - self.xi) / sigma
            ei = (y_min - mean - self.xi) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        else:
            ei = max(0, y_min - mean - self.xi)
        
        return ei
