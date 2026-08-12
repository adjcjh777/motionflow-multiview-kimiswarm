# 多视角相机标定最小流程：Checkerboard + COLMAP

## TL;DR
多视角标定的核心是得到每个相机的内参 `K` 与畸变系数，以及相机之间/每帧位姿的外参 `R,t`。离线阶段用棋盘格 + OpenCV 做单目内参标定；重建阶段用 COLMAP 自动估计外参并可选地 refine 内参。最终可输出每个视点的投影矩阵 `P = K [R | t]`。

## 关键结论
- **内参**：打印棋盘格 + OpenCV `findChessboardCorners` / `calibrateCamera`，即可获得 `K` 与 `distCoeffs`。
- **外参**：COLMAP 特征匹配 + SfM 自动给出每帧的 `images.txt`（四元数 + 平移）。
- **先验内参**：若 COLMAP 自动标定不稳定，可在 `feature_extractor` 后传入相机参数，或在 `mapper` 中固定内参。
- **统一坐标系**：多视角场景下，各相机/视角最好同时拍摄同一标定板或共享足够重叠区域，便于后续对齐。
- **最小输出**：`intrinsics.json`（或 COLMAP `cameras.txt`）+ 每帧的 `R,t`，用于构建 `P = K[R|t]`。

## 最小可运行示例

### 1. 棋盘格内参标定（OpenCV）
```python
import cv2, glob, numpy as np

w, h = 9, 6                # 棋盘格内角点数
square_size = 25.0         # mm，按实际打印尺寸修改
obj = np.zeros((w*h, 3), np.float32)
obj[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * square_size

obj_pts, img_pts = [], []
for p in glob.glob("calib/*.jpg"):
    gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        obj_pts.append(obj)
        img_pts.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, gray.shape[::-1], None, None)
print("K=\n", K)
print("dist=", dist)
```

### 2. 外参估计（COLMAP）
```bash
# 假设图片已放在 images/
colmap feature_extractor --database_path db.db --image_path images
colmap exhaustive_matcher --database_path db.db
mkdir -p sparse
colmap mapper --database_path db.db --image_path images --output_path sparse
# 结果：sparse/0/cameras.txt, images.txt, points3D.txt
```

### 3. 从 COLMAP 读取位姿并构造 P 矩阵
```python
import numpy as np

# images.txt 格式：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
# 读取每行后：
#   R = qvec_to_rotmat([qw,qx,qy,qz])   # 四元数 -> 旋转矩阵
#   t = np.array([tx, ty, tz])
# 投影矩阵：P = K @ np.hstack([R, t.reshape(3,1)])
```

## 参考链接
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- COLMAP Tutorial: https://colmap.github.io/tutorial.html
- COLMAP Camera Models: https://colmap.github.io/cameras.html
