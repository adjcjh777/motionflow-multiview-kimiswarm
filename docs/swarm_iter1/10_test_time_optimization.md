# Test-time Optimization / Bundle Adjustment for Multiview Pose

## TL;DR
Test-time optimization (TTO) for multiview pose uses bundle adjustment (BA) to refine 2D-reprojected or lifted 3D keypoints across calibrated cameras *after* a model has produced initial pose estimates. It minimizes the geometric reprojection error over camera extrinsics and/or human joint positions, turning noisy single-view predictions into temporally and view-consistent 3D poses.

## 关键结论
- **何时需要**：单目 pose networks 在不同视角存在遮挡、透视畸变和深度歧义；多视角 BA 在测试阶段消除这些不一致，无需重新训练模型。
- **最小目标函数**：

      min_{X_j, R_i, t_i} sum_{i,j} v_ij * rho( ||pi(K_i (R_i X_j + t_i)) - x_ij||^2 )

  其中 X_j 为 3D 关节，R_i、t_i 为各相机外参，K_i 为内参，pi 为透视投影，v_ij 为可见性掩码，rho 可选 Huber/GM 鲁棒核。
- **常用实现**：用 scipy.optimize.least_squares 或 ceres/g2o 做非线性最小二乘；线性三角测量提供初值。
- **扩展**：可加入时间平滑项 lambda * ||X_t - X_{t-1}||^2 和骨骼长度约束，提升稳定性。
- **取舍**：BA 增加推理延迟，不适合实时性极强的场景；但在离线/准离线多视角人体姿态任务中性价比高。

## 参考链接 / 最小代码

- **Bundler / COLMAP**: Schonberger & Frahm, Structure-from-Motion Revisited (COLMAP). 通用 SfM/BA 框架。
- **Human pose 应用**：Rhodin et al., Learning Monocular 3D Human Pose from Unsupervised Silhouettes; Pavllo et al., 3D Human Pose Estimation in Video with Temporal Convolutions 常与多视角 BA 后处理联用。
- **Python 最小可运行示例**（需 pip install numpy scipy，无大型依赖）：

    import numpy as np
    from scipy.optimize import least_squares

    # 相机内参（示例）
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    # 两个相机的外参
    R1, t1 = np.eye(3), np.zeros(3)
    R2 = np.array([[1, 0, 0], [0, 0.9, -0.44], [0, 0.44, 0.9]])
    t2 = np.array([0.1, 0, 0])
    Rs, ts = [R1, R2], [t1, t2]

    # 生成一个 3D 点并投影到两个视角，加少量噪声
    X_true = np.array([1.0, 2.0, 5.0])
    pts2d = []
    for R, t in zip(Rs, ts):
        x = R @ X_true + t
        x = x / x[2]
        proj = (K @ x)[:2] + np.random.randn(2) * 0.5
        pts2d.append(proj)
    pts2d = np.array(pts2d)

    def project(X):
        pts = []
        for R, t in zip(Rs, ts):
            x = R @ X + t
            x = x / x[2]
            pts.append((K @ x)[:2])
        return np.concatenate(pts)

    # 用线性三角测量给初值，再用 Levenberg-Marquardt 优化
    A = np.vstack([K @ R for R in Rs])
    b = np.hstack([p for p in pts2d])
    X0 = np.linalg.lstsq(A, b, rcond=None)[0][:3]

    res = least_squares(lambda X: project(X) - pts2d.ravel(), X0, method='lm')
    print('Optimized 3D point:', res.x)
    print('True 3D point:', X_true)

## 一句话总结
测试时 BA/TTO 是一种轻量后处理：用多视角几何一致性优化，把单目网络的“每个视角各自为政”变成“跨视角一致的 3D 姿态”。
