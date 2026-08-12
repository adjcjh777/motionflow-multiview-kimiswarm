# Calibration-Error Robustness for Iter11+ (ICRA/CVPR 2027)

## 1. Current state

The latest fusion model, `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`), treats the camera calibration `(K, R, t)` as ground truth. It computes ray directions from `K, R, t`, encodes a flattened `(K, R, t)` vector as a camera embedding, triangulates with weighted DLT, and refines with a differentiable Gauss-Newton step. The current best MPI-INF-3DHP validation MPJPE is ~11.17 mm from the cross-view residual model; the advanced model is still being tuned.

Existing training augmentation in `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py` only perturbs 2D keypoints (pixel noise, dropout, outliers). The robustness evaluation in `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` likewise tests 2D noise, occlusion, and outliers, but **never perturbs the camera calibration itself**. Real capture rigs have small calibration errors (focal-length drift, principal-point offsets, rig-alignment errors, temporal drift) that degrade triangulation without touching 2D detections.

## 2. Goal

Make the next-generation model robust to realistic calibration errors **without re-calibration at test time**, while preserving or improving clean-scene MPJPE. This is a strong, self-contained contribution for ICRA/CVPR 2027.

## 3. Proposed improvements

### 3.1 Training-time calibration jitter (highest priority)

Add a `perturb_cameras` transform that perturbs intrinsics and extrinsics before the forward pass:

- **Extrinsics**: small axis-angle noise on `R`, Gaussian noise on `t`.
- **Intrinsics**: perturb focal lengths, principal point, and optionally skew.
- **Scale-aware**: noise in real units (mm/degrees/pixels), sampled per batch.
- **Conservative**: keep jitter small enough to avoid destroying clean accuracy.

This is the cheapest change and should be tried first.

### 3.2 Learned calibration refinement head

Add a lightweight per-view network that predicts residual corrections `ΔR` and `Δt` (optionally `ΔK`) from per-view features, then uses the corrected cameras in DLT/Gauss-Newton. The head is supervised implicitly through the 3D loss and optionally a reprojection loss. A minimal version predicts a 6-DoF residual per camera (3 for rotation, 3 for translation) from pooled features and applies it before triangulation. Add this only after validating jitter.

### 3.3 Robust triangulation under calibration uncertainty

Extend the differentiable Gauss-Newton step with a camera-uncertainty regularizer, or sample multiple perturbed calibrations to obtain an ensemble 3D estimate. Repurpose the existing uncertainty head to down-weight views whose reprojection error is inconsistent with the current calibration.

## 4. Implementation plan

1. Implement `perturb_cameras` in a shared location, e.g. `motionflow_mv/calibration/perturb.py`.
2. Wire it into the training script (`experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`) via an augmentation flag.
3. Train baselines **with** and **without** jitter.
4. If jitter helps, optionally add the learned calibration refinement head.
5. Add a calibration-perturbation sweep to `experiments/eval_residual_robustness_mpiinf3dhp_v1.py`.

### Code snippet: camera perturbation

```python
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def perturb_cameras(K, R, t,
                    focal_std_px=10.0,
                    center_std_px=5.0,
                    rot_std_deg=0.5,
                    trans_std_mm=10.0):
    Kp = K.clone()
    Rp = R.clone()
    tp = t.clone()

    if focal_std_px > 0:
        Kp[:, :, 0, 0] += torch.randn_like(Kp[:, :, 0, 0]) * focal_std_px
        Kp[:, :, 1, 1] += torch.randn_like(Kp[:, :, 1, 1]) * focal_std_px
    if center_std_px > 0:
        Kp[:, :, 0, 2] += torch.randn_like(Kp[:, :, 0, 2]) * center_std_px
        Kp[:, :, 1, 2] += torch.randn_like(Kp[:, :, 1, 2]) * center_std_px

    if rot_std_deg > 0:
        aa = torch.randn_like(tp) * np.deg2rad(rot_std_deg)
        for b in range(aa.shape[0]):
            for v in range(aa.shape[1]):
                delta = R.from_rotvec(aa[b, v].cpu().numpy()).as_matrix()
                Rp[b, v] = torch.from_numpy(delta @ Rp[b, v].cpu().numpy())

    if trans_std_mm > 0:
        tp = tp + torch.randn_like(tp) * (trans_std_mm / 1000.0)

    return Kp, Rp, tp
```

### Code snippet: training integration

```python
for xb, yb, K, R, t in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    K, R, t = K.to(device), R.to(device), t.to(device)
    xb = augment_clip(xb)

    if args.calib_jitter:
        K, R, t = perturb_cameras(K, R, t,
                                  focal_std_px=args.focal_std_px,
                                  center_std_px=args.center_std_px,
                                  rot_std_deg=args.rot_std_deg,
                                  trans_std_mm=args.trans_std_mm)

    pred, _, _, nll_loss = model(xb, K=K, R=R, t=t)
    loss = criterion(pred, yb) + nll_loss
    ...
```

## 5. Experiments to run

| Experiment | Description |
|------------|-------------|
| Baseline | Current augmentations only. |
| +Ext. jitter | Rotation/translation jitter only. |
| +Int. jitter | Focal-length/principal-point jitter only. |
| +Full jitter | Combine extrinsic and intrinsic jitter. |
| +Refinement head | Add learned `ΔR, Δt` on top of full jitter. |

**Calibration robustness evaluation:** add a `perturb_calibration` sweep to `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` with levels:

- Rotation noise: 0.1°, 0.5°, 1.0°
- Translation noise: 1, 5, 10 mm
- Focal-length error: 1%, 2%, 5%
- Principal-point shift: 2, 5, 10 px

Run on MPI-INF-3DHP validation (`s_02_seq_01_v14_multiview_m.npz`).

## 6. Metrics to track

- **Primary**: MPJPE, PA-MPJPE, PCK@50/100/150 mm, AUC.
- **Robustness**: MPJPE under each calibration perturbation level.
- **Degradation ratio**: `MPJPE_perturbed / MPJPE_clean` per error source.
- **Per-view MPJPE**: identify sensitive cameras/views.
- **Uncertainty calibration**: correlation between predicted `log_var` and per-view reprojection error.

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Jitter harms clean accuracy | Start conservative (0.1° rotation, 1 mm translation); use curriculum. |
| Rotation parameterization is tricky | Use `scipy.spatial.transform.Rotation` in the CPU loader; avoid differentiating through it. |
| Intrinsics jitter changes scale | Keep jitter small relative to focal length. |
| Refinement head overfits | Keep it tiny (one or two MLP layers) and share with existing camera embedding. |
| Scope creep | Treat jitter as the first deliverable; refinement head only after jitter shows benefit. |

## 8. Expected outcomes

- Improved robustness to calibration errors without regressing clean MPJPE.
- A publishable ablation showing graceful degradation under realistic miscalibration.
- Clear next step: if learned refinement beats jitter alone, benchmark on Human3.6M WebBridge and Shelf/Campus real sequences.