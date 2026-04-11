"""
梯度预测 GPR 模型
直接预测梯度向量，引导优化向梯度为零的方向进行

核心思想：
- 输入：原子坐标 R (dim 维)
- 输出：梯度向量 ∇E(R) (dim 维)
- 目标：找到使 ||∇E(R)|| 最小的 R
"""
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from models.gpr_base import BaseGPRModel

# 过滤 sklearn GPR 的收敛警告（这些警告不影响功能）
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')


class GradientPredictingGPR(BaseGPRModel):
    """
    直接预测梯度的 GPR 模型
    
    与能量预测 GPR 的区别：
    - 能量预测 GPR：训练目标是能量 y，梯度只用于辅助
    - 梯度预测 GPR：训练目标是梯度向量，直接预测梯度
    
    优势：
    1. 直接建模梯度，预测更准确
    2. 采集函数直接使用预测梯度范数
    3. 更符合分子几何优化的物理目标（找到梯度为零的点）
    """

    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化梯度预测 GPR 模型

        Args:
            config: 配置字典
            dim: 输入维度（3 * n_atoms）
        """
        super().__init__(config)
        self.name = "GradientPredictingGPR"
        self.dim = dim

        # GPR 参数
        gpr_config = config.get('gpr', {})
        self.noise_variance = gpr_config.get('noise_variance', 1e-2)
        
        # 为每个梯度分量创建独立的 GPR 模型
        # 这样比多输出 GPR 更快，且能捕捉每个方向的梯度变化
        self.models = []
        for i in range(dim):
            # 使用合理的核函数参数边界
            # length_scale: 0.01-100 Å，覆盖分子运动的典型尺度
            # noise_level: 1e-4-1e-1，覆盖典型数值噪声范围
            kernel = (
                ConstantKernel(1.0, (1e-1, 1e1)) * 
                Matern(length_scale=np.ones(dim) * 1.0, nu=2.5, 
                       length_scale_bounds=(0.01, 100.0)) + 
                WhiteKernel(1e-3, (1e-4, 1e-1))
            )
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,  # 2 次重启，平衡速度和精度
                random_state=42
            )
            self.models.append(gpr)
        
        self.bounds = None
        self.is_trained = False

    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """设置变量边界"""
        self.bounds = bounds

    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练梯度预测模型

        Args:
            X: 输入坐标 (n_samples, dim)
            y: 能量值 (n_samples,) - 这里不使用，仅用于接口兼容
            gradients: 梯度 (n_samples, dim) ← 这是训练目标！
        """
        if X.shape[0] < 2:
            raise ValueError("至少需要 2 个训练点")
        
        if gradients is None:
            raise ValueError("梯度预测模型必须提供梯度数据")
        
        if gradients.shape != (X.shape[0], self.dim):
            raise ValueError(f"梯度形状不匹配：期望 {gradients.shape}, 得到 {(X.shape[0], self.dim)}")

        # 为每个梯度分量训练一个 GPR 模型
        # 第 i 个模型预测第 i 个梯度分量
        for i in range(self.dim):
            try:
                self.models[i].fit(X, gradients[:, i])
            except Exception as e:
                print(f"Warning: 训练梯度分量 {i} 失败：{e}")
                # 如果训练失败，使用常数预测
                self.models[i].fit(X, np.zeros(X.shape[0]))
        
        self.is_trained = True

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量（通过梯度积分近似，这里简单返回 0）
        
        注意：这个模型主要用于预测梯度，能量预测是次要的
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 简单实现：返回平均能量（实际应该通过梯度积分）
        return 0.0, 1.0

    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度向量

        Args:
            x: 输入坐标 (dim,)

        Returns:
            gradient: 预测梯度 (dim,)
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        x_reshaped = x.reshape(1, -1)
        
        # 预测每个梯度分量
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            gradient[i] = self.models[i].predict(x_reshaped)[0]
        
        return gradient

    def predict_gradient_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测梯度向量及其不确定性

        Args:
            x: 输入坐标 (dim,)

        Returns:
            gradient: 预测梯度 (dim,)
            uncertainty: 预测不确定性 (dim,)
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        x_reshaped = x.reshape(1, -1)
        
        gradient = np.zeros(self.dim)
        uncertainty = np.zeros(self.dim)
        
        for i in range(self.dim):
            grad_pred, std_pred = self.models[i].predict(x_reshaped, return_std=True)
            gradient[i] = grad_pred[0]
            uncertainty[i] = std_pred[0]
        
        return gradient, uncertainty

    def predict_energy_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        同时预测能量和梯度

        Args:
            x: 输入坐标

        Returns:
            energy: 预测能量（近似）
            gradient: 预测梯度
            energy_var: 能量预测方差
        """
        gradient = self.predict_gradient(x)
        energy, _ = self.predict(x)
        return energy, gradient, 1.0

    def acquisition_function(self, x: np.ndarray,
                             y_min: float = None) -> float:
        """
        采集函数：预测梯度范数 + 不确定性探索
        
        核心思想：
        1. 预测梯度范数 ||∇E_pred(x)||
        2. 梯度范数越小，越接近稳定构型
        3. 加入不确定性探索，鼓励探索高不确定性区域
        
        采集函数：
        Acq(x) = ||∇E_pred(x)|| - ξ·σ(x)
        
        其中：
        - ||∇E_pred(x)||：预测梯度范数（希望小）
        - σ(x)：预测不确定性（鼓励探索）
        - ξ：探索参数

        Args:
            x: 输入坐标
            y_min: 当前最小能量（不使用）

        Returns:
            acquisition_value: 采集函数值
        """
        predicted_gradient, uncertainty = self.predict_gradient_with_uncertainty(x)
        predicted_grad_norm = np.linalg.norm(predicted_gradient)
        avg_uncertainty = np.mean(uncertainty)
        
        # 采集函数：梯度范数 - 探索项
        # 我们希望最小化这个值
        # 梯度范数小 → 接近稳定构型
        # 不确定性大 → 鼓励探索
        xi = self.xi  # 探索参数（从配置读取）
        acquisition = predicted_grad_norm - xi * avg_uncertainty
        
        return acquisition

    def suggest_next_point(self, bounds: List[Tuple[float, float]],
                           y_min: float = None) -> np.ndarray:
        """
        建议下一个采样点
        
        通过优化采集函数找到梯度最小且不确定性高的区域

        Args:
            bounds: 变量边界
            y_min: 当前最小能量（不使用）

        Returns:
            x_next: 建议的下一个点
        """
        if bounds is None:
            bounds = self.bounds

        if bounds is None:
            raise ValueError("需要设置边界")

        # 随机采样 + 采集函数评估
        dim = len(bounds)
        n_candidates = 50  # 候选点数量
        candidates = []
        for _ in range(n_candidates):
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            candidates.append(x)
        
        # 评估每个候选点的采集函数值
        best_x = None
        best_acq_value = np.inf
        
        for x in candidates:
            acq_value = self.acquisition_function(x, y_min)
            if acq_value < best_acq_value:
                best_acq_value = acq_value
                best_x = x
        
        return best_x

    def get_confidence(self, x: np.ndarray) -> float:
        """
        获取预测置信度（梯度预测方差的倒数）
        """
        if not self.is_trained:
            return 0.0
        
        x_reshaped = x.reshape(1, -1)
        variances = []
        for i in range(self.dim):
            _, std = self.models[i].predict(x_reshaped, return_std=True)
            variances.append(std[0] ** 2)
        
        avg_variance = np.mean(variances)
        return 1.0 / (avg_variance + 1e-10)
