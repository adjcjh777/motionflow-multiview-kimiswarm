# Shelf / Campus 数据集：标定、真值、下载与评估

## TL;DR
Shelf 与 Campus 是多视角 3D 人体姿态估计的两个经典基准。
- **Campus**：3 摄像机、3 人、约 2000 帧，室外场景。
- **Shelf**：5 摄像机、5 人、约 3200 帧，室内货架场景。
两者均提供摄像机内外参、2D/3D 骨架真值；评估以 MPJPE 为主，多目标场景需先匹配再算指标。

## 关键结论

### 1. 数据集概况
| 数据集 | 场景 | 摄像机数 | 人数 | 帧数 | 分辨率 |
|--------|------|----------|------|------|--------|
| Campus | 室外 | 3 | 3 | ~2000 | 640×480 |
| Shelf  | 室内货架 | 5 | 5 | ~3200 | 1024×768 |

骨架通常采用类 COCO / H36M 的 17/18 关节定义。

### 2. 标定
- 标定信息一般以 `.npy` 或 `.json` 存于 `calibration/`、`cameras/` 或 `camera/` 目录。
- 每相机至少包含：
  - 内参 `K`（3×3）
  - 旋转 `R`（3×3）
  - 平移 `t`（3×1）
  - 可选畸变 `dist`

最小读取示例：
```python
import numpy as np

cam = np.load('calibration/camera0.npy', allow_pickle=True).item()
K, R, t = cam['K'], cam['R'], cam['t']

# 投影 3D -> 2D
X_h = np.hstack([X, np.ones((X.shape[0], 1))])      # (J, 4)
P = K @ np.hstack([R, t])                            # (3, 4)
x_h = (P @ X_h.T).T
x = x_h[:, :2] / x_h[:, 2:3]
```

### 3. 真值（GT）
- 3D 骨架在世界坐标系下，单位通常为 **mm** 或 **m**。
- 2D 真值由 3D 点经 `x = K(RX + t)` 投影得到。
- 遮挡/出界帧的对应关节可能缺失，读取时需过滤 `nan` 或空列表。

### 4. 下载镜像
- 原始发布页（TUM）：http://campar.in.tum.de/Chair/CampusShelfDataset
- VoxelPose 仓库（含预处理脚本/链接）：https://github.com/microsoft/voxelpose
- 社区备份（需自行验证完整性）：
  - https://github.com/HaiboTang/Campus-Shelf-Dataset
  - 部分中文镜像以网盘形式分发，提取码见原仓库说明

> 注：TUM 原页面链接不稳定，优先从 VoxelPose 仓库的 `README` 与 release 中找下载地址。

### 5. 评估
- **MPJPE**（Mean Per Joint Position Error，单位 mm）：
  \[\text{MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \| \hat{p}_j - p_j \|_2\]
- **Recall@500mm**：误差小于 500 mm 的关节比例。
- 多目标场景：先用匈牙利算法或最近邻在 3D 空间匹配预测骨架与 GT 骨架，再逐人算 MPJPE。

最小评估示例：
```python
import numpy as np
from scipy.spatial.procrustes import procrustes

# pred, gt: (J, 3)
mpjpe = np.mean(np.linalg.norm(pred - gt, axis=-1))

# PA-MPJPE：先 Procrustes 对齐
_, gt_aligned, _ = procrustes(gt, pred)
pa_mpjpe = np.mean(np.linalg.norm(pred - gt_aligned, axis=-1))
```

## 参考链接
- Campus/Shelf dataset page: http://campar.in.tum.de/Chair/CampusShelfDataset
- VoxelPose: https://github.com/microsoft/voxelpose
- "Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views", ECCV 2020
