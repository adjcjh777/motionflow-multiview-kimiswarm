# Iter11+ Loss-Function Design for MotionFlow-MultiView

**Topic:** concrete, implementable loss-function improvements for the ICRA/CVPR 2027 submission roadmap.  
**Scope:** training pipeline for `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` and related ray-attention fusion models.  
**Date:** 2026-08-04.

---

## 1. Current state

The all-in-one model (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`) and its siblings are trained almost exclusively with a single 3D MSE term, plus an optional fixed-weight reprojection loss and a 2D uncertainty NLL:

```python
pred, _, _, nll_loss = model(xb, K=K, R=R, t=t)
loss = MSE(pred, yb)
loss = loss + args.reproj_weight * L_reproj
loss = loss + nll_loss
```

**Key gaps:** MSE is outlier-sensitive; the predicted uncertainty is not used in the 3D loss; the raw DLT, GN, and residual-MLP stages are supervised only through the final output; and there is no skeletal or temporal regularizer. The current best MPI-INF-3DHP validation MPJPE is ~11.17 mm; the goal is to push below 10 mm.

---

## 2. Proposed improvements

### A. Robust per-joint 3D loss
Replace `nn.MSELoss()` with a per-joint Huber (Smooth-L1) loss. Hard joints (wrists, ankles) are up-weighted with an empirically calibrated `joint_weights` vector to focus capacity on error-prone joints.

### B. Aleatoric 3D uncertainty weighting
Convert predicted per-view log-variance into per-joint precision `τ_j = Σ_v exp(-log_var_vj)` and weight the 3D Huber term by `τ_j / τ_bar`. This makes the uncertainty head directly responsible for 3D accuracy and down-weights occluded joints.

### C. Intermediate multi-stage supervision
Expose `X_dlt` and `X_gn` and supervise all three stages: `L = L_pose(X_final) + λ_dlt L_pose(X_dlt) + λ_gn L_pose(X_gn)`. This stabilizes the triangulation head and prevents the residual MLP from absorbing all corrections.

### D. Skeletal consistency loss
Add a bone-length regularizer using mean training-set bone lengths, plus a left-right symmetry term. This helps occluded joints and adds anatomical plausibility.

### E. Temporal smoothness loss
Penalize acceleration: `mean_t || X_{t-1} - 2X_t + X_{t+1} ||^2`. A supervised variant compares predicted and GT velocities.

### F. Learned multi-task loss balancing
Replace hand-tuned `reproj_weight` and `uncertainty_weight` with Kendall & Gal homoscedastic uncertainty weights, learning `log(sigma^2)` for the 3D, reprojection, and smoothness terms.

---

## 3. Implementation plan

1. Create `motionflow_mv/losses/pose_3d.py` with the functions above, including a `multi_task_pose_loss` wrapper with learnable log-sigma terms.
2. Modify `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py` to return `X_dlt`/`X_gn` during training, replace `nn.MSELoss()`, and log per-component losses.
3. Add CLI flags: `--robust_delta`, `--aleatoric_weight`, `--bone_weight`, `--smooth_weight`, `--intermediate_weight`, `--learned_task_weights`.

---

## 4. Recommended experiments and metrics

### Experiments

| # | Experiment | Baseline / Change |
|---|-----------|------------------|
| 1 | **Baseline** | Current MSE + optional reprojection + uncertainty NLL |
| 2 | **Robust Huber** | A: replace MSE with Huber, δ = 10 mm |
| 3 | **+ Aleatoric 3D** | B: weight 3D loss by per-joint precision |
| 4 | **+ Intermediate** | C: supervise DLT and GN outputs |
| 5 | **+ Skeleton** | D: bone-length + symmetry loss |
| 6 | **+ Smoothness** | E: temporal acceleration loss |
| 7 | **+ Multi-task balance** | F: learn loss weights |

Run each on MPI-INF-3DHP S1 Seq1+2 → S2 Seq1 for 30 epochs with the same clip length (13) and batch size (8) to isolate the effect of the loss.

### Metrics to track

- **MPJPE / PA-MPJPE** in mm (primary).
- **PCK@50/100/150 mm** and **AUC**.
- **Per-joint MPJPE** to confirm gains on wrists/ankles.
- **Reprojection error** in pixels (diagnostic for uncertainty calibration).
- **Bone-length RMSE** vs. training-set mean bones.
- **Temporal jerk**: mean 3rd finite difference of predicted joints (should decrease with smoothness loss).
- **Uncertainty calibration**: rank correlation between predicted per-joint precision and actual 3D error.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Aleatoric weights collapse to near-zero | Medium | Clamp `log_var` range; normalize precision by batch mean; start with small `aleatoric_weight` |
| Temporal smoothness over-blurs motion | Medium | Use a small `smooth_weight`; prefer supervised velocity loss over zero-acceleration prior for fast actions |
| Bone-length prior dominates in scale-variant datasets | Low | Normalize skeletons to a common scale in the WebBridge loader before applying `L_bone` |
| Multi-task weights become unstable | Low | Initialize log-sigma to the hand-tuned values; clip gradients |
| More loss terms slow training | Low/Medium | The new terms are cheap per-sample operations; vectorize in `pose_3d.py` |
| Intermediate supervision requires model changes | Low | Add a `return_intermediates=True` training flag so inference stays unchanged |

---

## 6. Code sketch

```python
# motionflow_mv/losses/pose_3d.py
import torch
import torch.nn.functional as F

def huber_loss(err, delta=10.0):
    abs_err = err.abs()
    quad = torch.min(abs_err, torch.full_like(abs_err, delta))
    return 0.5 * quad ** 2 + delta * (abs_err - quad)

def aleatoric_pose_loss(pred, gt, log_var, delta=10.0):
    precision = torch.exp(-log_var).sum(dim=2)  # (B,T,J)
    precision = precision / precision.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    loss = huber_loss(pred - gt, delta).mean(dim=-1)  # (B,T,J)
    return (loss * precision).mean()

def bone_length_loss(pred, parent, child, mean_lengths):
    bones = pred[..., child, :] - pred[..., parent, :]
    return F.l1_loss(torch.norm(bones, dim=-1), mean_lengths)
```

Training-loop change:

```python
from motionflow_mv.losses.pose_3d import aleatoric_pose_loss, bone_length_loss

pred, log_var, nll, pred_dlt, pred_gn = model(xb, K=K, R=R, t=t, return_intermediates=True)
loss = aleatoric_pose_loss(pred, yb, log_var)
loss += 0.1 * (aleatoric_pose_loss(pred_dlt, yb, log_var) +
               aleatoric_pose_loss(pred_gn, yb, log_var))
loss += 0.01 * bone_length_loss(pred, PARENTS, CHILDREN, MEAN_BONES)
loss += args.reproj_weight * reprojection_loss(pred, xb[..., :2], K, R, t)
loss += nll
```

---

## 7. Summary and next step

The biggest expected gains are from **B (aleatoric 3D weighting)** and **C (intermediate supervision)**, because they directly exploit the uncertainty and triangulation heads that already exist. **D (skeletal consistency)** is a strong paper-quality contribution that also helps with occluded joints, and **F (learned multi-task balancing)** removes the fragility of hand-tuned `reproj_weight`.

**Next action:** implement `motionflow_mv/losses/pose_3d.py`, expose `X_dlt`/`X_gn` from the advanced model during training, and run experiment ladder 1–7 on MPI-INF-3DHP with the metrics above.
