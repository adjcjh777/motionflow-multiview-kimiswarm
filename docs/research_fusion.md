# Multi-view Pose Fusion: Research Note

> Compare simple ways to fuse per-view 2D/3D human poses into one 3D space, and pick a minimal method for the MotionFlow multi-view extension.

## 1. Problem statement

Per frame we have:

* `V` calibrated cameras with intrinsics `K_v` and extrinsics `[R_v | t_v]`;
* per-view 2D pose `x_v ∈ R^{J×2}` and confidence `w_v ∈ [0,1]^J`.

Output: a single 3D skeleton `X ∈ R^{J×3}` in a shared world frame.

First milestone assumptions: static synchronized cameras, known calibration, one person per scene, with missing joints handled by zero/low confidence.

## 2. Candidate methods

### 2.1 Direct Linear Transform (DLT) triangulation

DLT builds a linear system from the projection equations of each view and solves for the 3D point that best satisfies all rays in a least-squares sense.

* **Pros:** deterministic, parameter-free, fast (`SVD` / `numpy.linalg.lstsq`), easy to verify.
* **Cons:** treats every view equally, so an occluded/noisy view biases the result; cannot correct systematic detector bias.
* **References:** Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2003); [Wikipedia – Triangulation](https://en.wikipedia.org/wiki/Triangulation_(computer_vision)); EasyMocap triangulation helpers.

### 2.2 Confidence-weighted fusion

Weight each view by its 2D confidence inside the DLT system, then refine by minimizing the weighted reprojection error. Two common forms:

1. **Weighted DLT:** stack `w_v · (cross terms)`, letting reliable views dominate.
2. **Weighted reprojection refinement:** run a few Gauss-Newton/LM iterations on `Σ w_v · ρ²(x_v, π_v(X))`.

* **Pros:** still lightweight; naturally handles missing/occluded views; improves accuracy when confidence correlates with true error.
* **Cons:** needs a calibrated confidence score; fails if all views are poor.
* **References:** EasyMocap epipolar triangulation; Dong et al., *Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views*, IEEE T-PAMI 2021 ([project page](https://zju3dv.github.io/mvpose/)); OpenPose 3D branch.

### 2.3 Light attention / transformer fusion

Treat each per-view 2D keypoint as a token (camera ID + joint ID), then use a small transformer (1–2 layers, 4 heads, ≤256 dim) to predict an attention-weighted 3D pose.

* **Pros:** learns to suppress occluded views and adapt to variable view counts.
* **Cons:** needs training data and ground-truth 3D labels; more code, parameters, and tuning.
* **References:** MTF-Transformer, *Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation* ([arXiv search](https://arxiv.org/search/?query=%22Adaptive+Multi-view+and+Temporal+Fusing+Transformer%22&searchtype=all)); VTP-style volumetric transformers; METRO/FastMETRO encoders.

## 3. Recommendation

**Start with confidence-weighted DLT triangulation.**

* It is the shortest path from per-view 2D poses to a fused 3D skeleton: ~50–100 lines of NumPy, no training data, no GPU model design.
* It uses camera calibration and per-joint confidence scores that most 2D detectors already output.
* It degrades gracefully when cameras or joints are missing (`w → 0`).
* It gives a strong baseline for measuring any later attention-based upgrade.

## 4. Minimal implementation sketch

```
fusion/
  triangulate.py   # DLT + confidence-weighted least squares
  utils.py         # projection / reprojection helpers
```

Core pseudo-code:

```python
import numpy as np

def triangulate_dlt(points_2d, confidences, cameras):
    J = points_2d.shape[1]
    X = np.zeros((J, 3))
    for j in range(J):
        A = []
        for v, p in enumerate(points_2d[:, j]):
            if confidences[v, j] <= 0:
                continue
            P = cameras[v]['P']
            w = confidences[v, j]
            A.append(w * np.array([p[1]*P[2] - P[1], P[0] - p[0]*P[2]]))
        A = np.vstack(A) if A else np.zeros((2, 4))
        _, _, Vt = np.linalg.svd(A)
        X[j] = Vt[-1, :3] / Vt[-1, 3]
    return X
```

After DLT, run a few Gauss-Newton or LM iterations to minimize the weighted reprojection error.

## 5. When to add attention fusion

* When MPJPE/PA-MPJPE on validation plateaus and learned view weighting clearly beats fixed confidence.
* Keep the transformer tiny (≤2 layers, ≤256 hidden dim) and train only the fusion head.

## 6. References

1. Hartley, R. and Zisserman, A., *Multiple View Geometry in Computer Vision*, 2nd ed., CUP, 2003.
2. Wikipedia, “Triangulation (computer vision).” https://en.wikipedia.org/wiki/Triangulation_(computer_vision)
3. ZJU-3DV, EasyMocap. https://github.com/zju3dv/EasyMocap
4. Dong, J., et al., “Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views,” IEEE T-PAMI, 2021. https://zju3dv.github.io/mvpose/
5. Cao, Z., et al., “OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,” arXiv:1812.08008, 2018. https://github.com/CMU-Perceptual-Computing-Lab/openpose
6. Shuai, H., et al., “Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation.” https://arxiv.org/search/?query=%22Adaptive+Multi-view+and+Temporal+Fusing+Transformer%22&searchtype=all
