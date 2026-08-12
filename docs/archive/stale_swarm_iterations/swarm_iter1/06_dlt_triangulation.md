# DLT 三角化多视角 3D 关键点

## TL;DR

Direct Linear Transform（DLT）三角化是利用多个已标定相机的 2D 关键点恢复 3D 点的最简方法。它不需要训练，实现简单，应作为多视角融合的第一版 baseline。

## 关键结论

- 输入：N 个相机的投影矩阵 `P_i = K_i [R_i | t_i]` 和对应 2D 点 `(u_i, v_i)`。
- 输出：世界坐标系下的 3D 点 `X`。
- 原理：每个视角提供两个线性方程 `x_i × P_i X = 0`，堆叠后 SVD 求解。
- 优点：无参、确定性强、对遮挡/缺失视角天然鲁棒（只要视图数 ≥ 2）。
- 缺点：假设 2D 点无噪声时精确；噪声大时需要加权/鲁棒版本。

## 最小可运行 NumPy 代码

```python
import numpy as np

def triangulate_dlt(points_2d, proj_matrices):
    """
    points_2d: (N, 2)
    proj_matrices: (N, 3, 4)
    returns X: (3,)
    """
    A = []
    for (u, v), P in zip(points_2d, proj_matrices):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]
```

## 扩展：置信度加权 DLT

对每个视角引入置信度 `w_i`（来自 2D 检测器），将对应方程乘以 `sqrt(w_i)`，可抑制低质量视角。

## 参考

- Hartley & Zisserman. *Multiple View Geometry in Computer Vision*. 2004.
- Iskakov et al. *Learnable Triangulation of Human Pose*. ICCV 2019.
- https://github.com/microsoft/Graph-based-Multi-view-3D-Human-Pose-Estimation
