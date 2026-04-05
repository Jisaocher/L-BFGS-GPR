"""
3D 分子结构可视化
使用 matplotlib 和 ASE 进行分子结构展示
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Optional, Tuple, Dict
from core.molecule import Molecule

# 导入 zhplot 以支持中文显示
try:
    import zhplot
    zhplot.matplotlib_chineseize()
except (ImportError, AttributeError):
    # 如果 zhplot 不可用，尝试其他中文支持方式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 
                                        'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 原子颜色和半径（CPK 配色）
ATOM_COLORS = {
    'H': 'white', 'He': 'pink',
    'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green',
    'P': 'orange', 'S': 'yellow', 'Cl': 'green', 'Br': 'darkred',
    'I': 'purple', 'Na': 'purple', 'K': 'purple',
    'Fe': 'orange', 'Cu': 'brown', 'Zn': 'gray',
    'Ca': 'green', 'Mg': 'green', 'Al': 'gray',
    'Si': 'brown', 'Ti': 'gray', 'Pt': 'gray'
}

ATOM_RADII = {
    'H': 0.25, 'He': 0.28,
    'C': 0.60, 'N': 0.55, 'O': 0.50, 'F': 0.50,
    'P': 0.90, 'S': 0.90, 'Cl': 0.85, 'Br': 0.95,
    'I': 1.10, 'Na': 1.40, 'K': 1.60,
    'Fe': 1.20, 'Cu': 1.30, 'Zn': 1.30,
    'Ca': 1.70, 'Mg': 1.40, 'Al': 1.25,
    'Si': 1.10, 'Ti': 1.40, 'Pt': 1.40
}

# 默认值
DEFAULT_COLOR = 'gray'
DEFAULT_RADIUS = 0.7


class MoleculeVisualizer3D:
    """
    3D 分子结构可视化器
    """
    
    def __init__(self, font_size: int = 14, figure_size: Tuple[int, int] = (10, 8),
                 dpi: int = 300, show_atom_labels: bool = True):
        """
        初始化可视化器
        
        Args:
            font_size: 字体大小
            figure_size: 图形尺寸
            dpi: 分辨率
            show_atom_labels: 是否显示原子标签
        """
        self.font_size = font_size
        self.figure_size = figure_size
        self.dpi = dpi
        self.show_atom_labels = show_atom_labels
        
        plt.rcParams['font.size'] = font_size
    
    def get_atom_color(self, symbol: str) -> str:
        """获取原子颜色"""
        return ATOM_COLORS.get(symbol, DEFAULT_COLOR)
    
    def get_atom_radius(self, symbol: str) -> float:
        """获取原子半径"""
        return ATOM_RADII.get(symbol, DEFAULT_RADIUS)
    
    def _get_bonds(self, coords: np.ndarray, atom_symbols: List[str],
                   tolerance: float = 0.4) -> List[Tuple[int, int]]:
        """
        检测化学键
        
        Args:
            coords: 原子坐标
            atom_symbols: 原子符号
            tolerance: 键长容忍度
        
        Returns:
            bonds: 键列表 [(atom1_idx, atom2_idx), ...]
        """
        bonds = []
        n_atoms = len(atom_symbols)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # 基于共价半径判断是否成键
                r1 = self.get_atom_radius(atom_symbols[i])
                r2 = self.get_atom_radius(atom_symbols[j])
                bond_threshold = (r1 + r2) * 1.3 + tolerance
                
                if dist < bond_threshold:
                    bonds.append((i, j))
        
        return bonds
    
    def visualize(self, molecule: Molecule, title: str = None,
                  show: bool = False, save_path: str = None,
                  elevation: float = 20, azimuth: float = 45,
                  axis_on: bool = False) -> plt.Figure:
        """
        可视化分子 3D 结构
        
        Args:
            molecule: 分子对象
            title: 标题
            show: 是否显示
            save_path: 保存路径
            elevation: 仰角
            azimuth: 方位角
            axis_on: 是否显示坐标轴
        
        Returns:
            fig: 图形对象
        """
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        coords = molecule.coords
        atom_symbols = molecule.atom_symbols
        
        # 绘制化学键
        bonds = self._get_bonds(coords, atom_symbols)
        if bonds:
            bond_lines = []
            for i, j in bonds:
                bond_lines.append([coords[i], coords[j]])
            
            segments = Line3DCollection(bond_lines, colors='gray', 
                                        linewidths=1.5, alpha=0.7)
            ax.add_collection(segments)
        
        # 绘制原子
        for i, symbol in enumerate(atom_symbols):
            color = self.get_atom_color(symbol)
            radius = self.get_atom_radius(symbol) * 10  # 放大用于显示
            
            # 处理白色原子的边缘
            if color == 'white':
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                          c=color, edgecolors='gray', s=radius**2 * 100,
                          alpha=0.9, label=symbol)
            else:
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                          c=color, s=radius**2 * 100, alpha=0.9, label=symbol)
            
            # 原子标签
            if self.show_atom_labels:
                ax.text(coords[i, 0] + 0.1, coords[i, 1] + 0.1, coords[i, 2] + 0.1,
                       symbol, fontsize=self.font_size, ha='center', va='center')
        
        # 设置视角
        ax.view_init(elev=elevation, azim=azimuth)
        
        # 设置坐标轴
        if not axis_on:
            ax.set_axis_off()
        else:
            ax.set_xlabel('X (Å)', fontsize=self.font_size)
            ax.set_ylabel('Y (Å)', fontsize=self.font_size)
            ax.set_zlabel('Z (Å)', fontsize=self.font_size)
        
        # 标题
        if title:
            ax.set_title(title, fontsize=self.font_size, pad=20)
        
        # 自动调整视角范围
        self._set_equal_aspect(ax, coords)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_comparison(self, molecules: List[Molecule], 
                             titles: List[str] = None,
                             save_path: str = None,
                             show: bool = False) -> plt.Figure:
        """
        并排比较多个分子结构
        
        Args:
            molecules: 分子列表
            titles: 标题列表
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            fig: 图形对象
        """
        n_mols = len(molecules)
        fig = plt.figure(figsize=(self.figure_size[0] * n_mols, self.figure_size[1]),
                        dpi=self.dpi)
        
        for idx, mol in enumerate(molecules):
            ax = fig.add_subplot(1, n_mols, idx + 1, projection='3d')
            
            coords = mol.coords
            atom_symbols = mol.atom_symbols
            
            # 绘制化学键
            bonds = self._get_bonds(coords, atom_symbols)
            if bonds:
                bond_lines = []
                for i, j in bonds:
                    bond_lines.append([coords[i], coords[j]])
                
                segments = Line3DCollection(bond_lines, colors='gray',
                                           linewidths=1.5, alpha=0.7)
                ax.add_collection(segments)
            
            # 绘制原子
            for i, symbol in enumerate(atom_symbols):
                color = self.get_atom_color(symbol)
                radius = self.get_atom_radius(symbol) * 10
                
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                          c=color, s=radius**2 * 100, alpha=0.9)
                
                if self.show_atom_labels:
                    ax.text(coords[i, 0] + 0.1, coords[i, 1] + 0.1, coords[i, 2] + 0.1,
                           symbol, fontsize=self.font_size)
            
            ax.view_init(elev=20, azim=45)
            ax.set_axis_off()
            self._set_equal_aspect(ax, coords)
            
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=self.font_size, pad=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def _set_equal_aspect(self, ax: Axes3D, coords: np.ndarray) -> None:
        """设置等比例坐标轴"""
        max_range = np.max(np.ptp(coords, axis=0)) / 2
        mid = np.mean(coords, axis=0)
        
        ax.set_xlim(mid[0] - max_range * 0.8, mid[0] + max_range * 0.8)
        ax.set_ylim(mid[1] - max_range * 0.8, mid[1] + max_range * 0.8)
        ax.set_zlim(mid[2] - max_range * 0.8, mid[2] + max_range * 0.8)
    
    def visualize_trajectory(self, molecules: List[Molecule],
                            title: str = "优化轨迹",
                            save_path: str = None,
                            show: bool = False) -> plt.Figure:
        """
        可视化优化轨迹
        
        Args:
            molecules: 分子结构列表（按时间顺序）
            title: 标题
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            fig: 图形对象
        """
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用第一个分子的结构作为参考
        coords = molecules[0].coords
        atom_symbols = molecules[0].atom_symbols
        
        # 绘制化学键
        bonds = self._get_bonds(coords, atom_symbols)
        if bonds:
            bond_lines = []
            for i, j in bonds:
                bond_lines.append([coords[i], coords[j]])
            
            segments = Line3DCollection(bond_lines, colors='lightgray',
                                       linewidths=1, alpha=0.5)
            ax.add_collection(segments)
        
        # 绘制原子（半透明背景）
        for i, symbol in enumerate(atom_symbols):
            color = self.get_atom_color(symbol)
            radius = self.get_atom_radius(symbol) * 10
            
            ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                      c=color, s=radius**2 * 100, alpha=0.2)
        
        # 绘制轨迹
        colors = plt.cm.viridis(np.linspace(0, 1, len(molecules)))
        
        for idx, mol in enumerate(molecules):
            alpha = 0.3 + 0.7 * (idx / len(molecules))  # 逐渐变深
            
            for i, symbol in enumerate(atom_symbols):
                ax.scatter(mol.coords[i, 0], mol.coords[i, 1], mol.coords[i, 2],
                          c=[colors[idx]], s=50, alpha=alpha, 
                          edgecolors='none')
        
        # 突出显示初始和最终结构
        # 初始
        for i, symbol in enumerate(atom_symbols):
            ax.scatter(molecules[0].coords[i, 0], molecules[0].coords[i, 1],
                      molecules[0].coords[i, 2],
                      c='red', s=100, marker='o', label='Initial',
                      alpha=0.8)
        
        # 最终
        for i, symbol in enumerate(atom_symbols):
            ax.scatter(molecules[-1].coords[i, 0], molecules[-1].coords[i, 1],
                      molecules[-1].coords[i, 2],
                      c='green', s=100, marker='s', label='Final',
                      alpha=0.8)
        
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off()
        self._set_equal_aspect(ax, coords)
        ax.set_title(title, fontsize=self.font_size, pad=20)
        
        # 添加图例
        ax.legend(loc='upper left', fontsize=self.font_size - 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


def create_structure_visualization(molecule: Molecule, save_path: str,
                                   **kwargs) -> str:
    """
    便捷函数：创建分子结构可视化并保存
    
    Args:
        molecule: 分子对象
        save_path: 保存路径
        **kwargs: 传递给 MoleculeVisualizer3D 的参数
    
    Returns:
        save_path: 保存的文件路径
    """
    vis = MoleculeVisualizer3D()
    vis.visualize(molecule, save_path=save_path, **kwargs)
    return save_path
