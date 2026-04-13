from visualization.structure3d import MoleculeVisualizer3D
vis = MoleculeVisualizer3D(
    font_size=14,
    figure_size=tuple([800, 600]),
    dpi=300,
    ai_method=None  # 传递 AI 方法类型
)
vis.visualize_from_xyz('/mnt/e/wsl_dir/L-BFGS-GPR/new_method/output/structures/lbfgs_final.xyz', save_path='/mnt/e/wsl_dir/L-BFGS-GPR/new_method/output/structures/lbfgs_final.html')