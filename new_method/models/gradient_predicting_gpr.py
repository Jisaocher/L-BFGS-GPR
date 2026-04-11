"""
梯度预测 GPR 模型 - 学习优化器策略

核心思想：
- 输入：[坐标 (27 维) + 梯度 (27 维)] = 54 维
- 输出：新坐标 (27 维)
- 目标：学习"从当前状态预测梯度更小的新坐标"

这本质上是在学习 L-BFGS 的优化策略！
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
    梯度预测 GPR 模型 - 学习优化策略
    
    与旧版本的区别：
    - 旧版本：输入坐标 → 预测梯度（27 个独立 GPR）
    - 新版本：输入 [坐标 + 梯度] → 预测新坐标（一个 GPR 学习优化策略）
    """

    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化梯度预测 GPR 模型

        Args:
            config: 配置字典
            dim: 输入维度（3 * n_atoms = 27）
        """
        super().__init__(config)
        self.name = "GradientPredictingGPR"
        self.dim = dim  # 坐标维度（27）
        self.input_dim = dim * 2  # 输入维度（54 = 坐标 + 梯度）
        
        # AI 训练配置
        ai_config = config.get('ai_training', {})
        self.i_best_steps = ai_config.get('i_best_steps', 5)
        self.j_recent_steps = ai_config.get('j_recent_steps', 10)
        
        # AI 预测配置
        pred_config = config.get('ai_prediction', {})
        self.max_displacement = pred_config.get('max_displacement', 0.3)
        self.min_displacement = pred_config.get('min_displacement', 0.01)

        # 创建 GPR 模型（预测坐标更新）
        # 使用各向同性核，减少参数数量
        kernel = (
            ConstantKernel(1.0, (1e-1, 1e1)) * 
            Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 100.0)) + 
            WhiteKernel(1e-3, (1e-4, 1e-1))
        )
        
        # 为每个坐标分量创建一个 GPR 模型
        self.models = []
        for i in range(dim):
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=42
            )
            self.models.append(gpr)
        
        self.bounds = None
        self.is_trained = False
        
        # 存储训练数据（用于增量更新）
        self.X_train = []  # 输入：[坐标，梯度] (54 维)
        self.y_train = []  # 输出：新坐标 (27 维)

    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """设置变量边界"""
        self.bounds = bounds

    def add_training_data(self, coords: np.ndarray, gradient: np.ndarray, 
                         next_coords: np.ndarray) -> None:
        """
        添加训练数据

        Args:
            coords: 当前坐标 (dim,)
            gradient: 当前梯度 (dim,)
            next_coords: 下一步坐标 (dim,)
        """
        # 构建输入：[坐标，梯度] (54 维)
        input_vec = np.concatenate([coords.flatten(), gradient.flatten()])
        self.X_train.append(input_vec)
        self.y_train.append(next_coords.flatten())

    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练模型

        Args:
            X: 当前轮次的坐标数据 (n_samples, dim) - 不使用，实际数据在 self.X_train
            y: 能量值（不使用）
            gradients: 梯度数据 (n_samples, dim) - 不使用
        """
        if len(self.X_train) < 2:
            print("Warning: 训练数据不足，跳过训练")
            return

        # 转换为 numpy 数组
        try:
            X_train = np.array(self.X_train)  # (n_samples, 54)
            y_train = np.array(self.y_train)  # (n_samples, 27)
            
            # 确保形状正确
            if X_train.ndim != 2 or X_train.shape[1] != self.input_dim:
                print(f"Warning: X_train 形状不正确：{X_train.shape}, 期望：(n, {self.input_dim})")
                return
            
            if y_train.ndim != 2 or y_train.shape[1] != self.dim:
                print(f"Warning: y_train 形状不正确：{y_train.shape}, 期望：(n, {self.dim})")
                return
            
            # 为每个坐标分量训练一个 GPR 模型
            for i in range(self.dim):
                try:
                    self.models[i].fit(X_train, y_train[:, i])
                except Exception as e:
                    print(f"Warning: 训练坐标分量 {i} 失败：{e}")
                    # 如果训练失败，使用常数预测
                    self.models[i].fit(X_train, np.zeros(X_train.shape[0]))

            self.is_trained = True
            print(f"GPR 模型训练完成，使用 {len(self.X_train)} 个训练点")
            
        except Exception as e:
            print(f"Error: 训练失败：{e}")
            print(f"X_train length: {len(self.X_train)}, y_train length: {len(self.y_train)}")

    def predict_next_coords(self, coords: np.ndarray, gradient: np.ndarray,
                           apply_displacement_limit: bool = True) -> np.ndarray:
        """
        预测下一步坐标

        Args:
            coords: 当前坐标 (dim,)
            gradient: 当前梯度 (dim,)
            apply_displacement_limit: 是否应用位移限制

        Returns:
            next_coords: 预测的新坐标 (dim,)
        """
        if not self.is_trained:
            print("Warning: 模型未训练，返回当前坐标")
            return coords.copy()
        
        # 构建输入：[坐标，梯度]
        input_vec = np.concatenate([coords, gradient]).reshape(1, -1)
        
        # 预测每个坐标分量
        next_coords = np.zeros(self.dim)
        for i in range(self.dim):
            next_coords[i] = self.models[i].predict(input_vec)[0]
        
        # 应用位移限制
        if apply_displacement_limit:
            displacement = next_coords - coords
            disp_norm = np.linalg.norm(displacement)
            
            # 限制最大位移
            if disp_norm > self.max_displacement:
                displacement = displacement * (self.max_displacement / disp_norm)
            
            # 限制最小位移（避免预测点不变）
            elif disp_norm < self.min_displacement:
                if disp_norm > 1e-10:
                    displacement = displacement * (self.min_displacement / disp_norm)
                else:
                    # 如果预测完全不变，沿负梯度方向移动一小步
                    displacement = -0.01 * gradient / (np.linalg.norm(gradient) + 1e-10)
            
            next_coords = coords + displacement
        
        return next_coords

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量（占位符，不使用）
        """
        return 0.0, 1.0

    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度（占位符，不使用）
        """
        return np.zeros(self.dim)

    def acquisition_function(self, x: np.ndarray, y_min: float = None) -> float:
        """
        采集函数（不使用，因为预测已经是坐标）
        """
        return 0.0

    def suggest_next_point(self, bounds: List[Tuple[float, float]],
                           y_min: float = None) -> np.ndarray:
        """
        建议下一个采样点（不使用，由 hybrid.py 直接调用 predict_next_coords）
        """
        raise NotImplementedError("GradientPredictingGPR 使用 predict_next_coords 直接预测坐标")

    def clear_data(self) -> None:
        """清除所有训练数据"""
        self.X_train = []
        self.y_train = []
        self.is_trained = False

    def n_training_points(self) -> int:
        """获取训练点数"""
        return len(self.X_train)
