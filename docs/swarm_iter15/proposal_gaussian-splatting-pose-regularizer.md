# Proposal: Gaussian-Splatting Pose Regularizer

**Author:** Iter15 design swarm — agent task "gaussian-splatting-pose-regularizer: differentiable 3D Gaussian rendering as a multi-view pose regularizer"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** Principal-point correction in `motionflow_mv/fusion/principal_point_correction.py`, robust reprojection losses in `motionflow_mv/losses/reprojection_consistency.py`, per-view weighting and DLT triangulation in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`.

---

## 1. Problem

The anchor model triangulates a 3-D pose and refines it with a residual MLP, but the only cross-view geometric constraints come from the DLT triangulation step and an optional scalar reprojection MSE. There is no explicit probabilistic model that ties predicted 3-D joint positions back to the 2-D observations in each view, and there is no mechanism to learn per-joint spatial uncertainty. Consequently, the model can overfit to calibration noise, produce over-confident predictions, and fail when a subset of views is noisy or occluded.

## 2. Hypothesis

**One-sentence hypothesis:** Adding a differentiable 3D Gaussian-splatting regularizer that projects each predicted joint (with a learned per-joint anisotropic covariance) back into every calibrated view and penalizes the negative log-likelihood of the observed 2-D keypoints will improve multi-view geometric consistency, calibration alignment, and robustness to missing or noisy views while preserving the 9.32 mm anchor performance.

## 3. Method

### 3.1 Architecture changes

Create a new model variant that predicts a per-joint 3-D Gaussian covariance in addition to the refined 3-D pose.

**New file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`

- Implement `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat`, a subclass of `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
- Reuse the existing pooled spatio-temporal features `feat_pooled` and the raw triangulated pose `pred_3d_raw` already computed by the anchor.
- Append a lightweight covariance head `self.covariance_head` (2-layer MLP, output 3) that predicts `log_std` per joint along the world x/y/z axes.
- Keep the residual refinement head unchanged; the covariance is predicted in parallel and is only used by the new auxiliary loss.
- Forward signature: returns `(pred_3d, weights, pp_delta, focal_scale, log_std)` when `return_covariance=True`; otherwise falls back to the parent return tuple.

### 3.2 Loss / data changes

**New file:** `motionflow_mv/losses/gaussian_splatting_pose_loss.py`

- Implement `gaussian_splatting_pose_loss(pred_3d, points_2d, K, R, t, log_std, confidences=None, eps=1e-4)`.
- Build per-joint 3-D covariance `Σ_3d = diag(exp(log_std)²)`.
- For each view, compute the Jacobian of the pinhole projection at the predicted 3-D point and project the 3-D Gaussian into the image plane: `Σ_2d = J·Σ_3d·Jᵀ`.
- Evaluate the negative log-likelihood of the observed 2-D keypoints under the projected 2-D Gaussian, weighted by input confidences.
- Add a tiny covariance-trace regularizer to prevent degenerate near-zero covariances.

**Modify:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`

- Add `"splat"` to the `model_type` choices.
- Instantiate the new splat model when `model_type == "splat"`.
- Add `--splat_loss_weight` (default 0.01) and `--splat_trace_weight` (default 1e-4) arguments.
- When the splat loss is enabled, extract `log_std` from the model outputs and add `splat_loss_weight * loss_splat + splat_trace_weight * trace_reg` to the total loss.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/losses/gaussian_splatting_pose_loss.py`
- **Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`
- **Create:** `tests/test_gaussian_splatting_pose_loss.py` (CPU sanity: forward + backward, shape checks, finite gradients)
- **Modify:** `motionflow_mv/losses/__init__.py` (export `gaussian_splatting_pose_loss`)
- **Modify:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` (model type, arg parsing, loss term)

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small MPI-INF-3DHP split, identical to the factorized ST+PP smoke so numbers are comparable.

| Setting | Value |
|---|---|
| Train | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (500 random clips) |
| Val | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| Clip length | 13 |
| Batch size | 4 |
| Model dims | `d=32`, `residual_hidden=64`, `n_st_layers=2`, `principal_point_hidden=64` |
| Optimizer | Adam, lr=1e-3 |
| Loss | MPJPE + `splat_loss_weight=0.01` + `splat_trace_weight=1e-4` |
| Epochs | 5 |

**Pass/fail criteria:**

- **Pass:** training completes with no NaNs / crashes and val MPJPE ≤ 60 mm.
- **Pass:** the splat loss returns a finite scalar and produces finite gradients on the CPU test.
- **Pass:** projected 2-D covariance matrices are positive-definite for every joint/view/frame.
- **Fail:** val MPJPE > 80 mm, any NaN/Inf, or the splat loss produces non-positive-definite covariances.

## 5. Evaluation Plan

If the smoke passes, evaluate with the standard harness:

- **Clean metrics:** `experiments/eval_full_metrics.py --model splat --checkpoint <smoke_checkpoint> --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`. Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
- **Calibration alignment:** after artificially perturbing `K` (±10 px principal point, ±2% focal length), compare reprojection error and MPJPE degradation against the anchor. Pass if the splat model degrades less.
- **View-robustness (CPU, 20-clip smoke):** run `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` with `view_dropout_0.2` and `joint_dropout_0.2`; pass if degradation relative to clean is ≤ 5 percentage points better than the anchor.
- **Uncertainty quality (diagnostic):** for held-out samples, report correlation between predicted per-joint `trace(Σ_2d)` and actual reprojection error; pass if positive correlation is ≥ 0.3.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples) | RTX 4090 | ~20–30 min |
| CPU sanity tests (`tests/test_gaussian_splatting_pose_loss.py`) | CPU | < 1 min |
| Full training (20–50 epochs, full split) | RTX 4090 / A800-D | ~5–9 h on RTX 4090 |
| Clean eval + robustness matrix | RTX 4090 or CPU | ~10–20 min |

The covariance head adds fewer than 5k parameters and the splat loss is O(V·J) per frame; memory and throughput are essentially identical to the PP baseline.

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| Splat loss dominates early training and destabilizes the pose branch. | Medium | Start with `splat_loss_weight=0.001` and ramp up; if NaNs appear, disable the splat term and train with MPJPE only. |
| Predicted covariances collapse or explode (non-PD). | Low | Clamp `log_std` to `[-3, 1]` (≈ 5 cm – 2.7 m); add the trace regularizer; if still unstable, fix covariance to a global learnable scalar. |
| No clean improvement over 9.32 mm anchor. | Medium-High | The splat term is a low-risk geometric prior; if clean MPJPE does not improve within 0.3 mm, keep the new model only if robustness metrics improve. Otherwise, fall back to the anchor and publish the loss as an auxiliary ablation. |
| Splat projection Jacobian is expensive on variable views. | Low | The Jacobian is a closed-form matrix multiply; if it becomes a bottleneck, precompute per-view Jacobians once per forward pass. |
| Camera perturbations expose unmodelled radial distortion. | Low | The current pinhole model is sufficient for the dataset; if distortion becomes visible, extend the Jacobian with a radial-distortion term in a follow-up. |

---

## Summary

Introduce a differentiable Gaussian-splatting regularizer that projects predicted 3-D joint Gaussians into each calibrated view and scores the 2-D observations by their likelihood. A tiny per-joint covariance head is added to the anchor model; the change is a single new model file and a single new loss, validated with a 5-epoch MPI-INF-3DHP smoke before any full run.
