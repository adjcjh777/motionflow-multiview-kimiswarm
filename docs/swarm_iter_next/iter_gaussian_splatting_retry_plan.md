# Gaussian Splatting Retry — Next Iteration Plan

**Date:** 2026-08-06  
**Author:** gaussian_splatting_retry agent  
**Baseline anchor:** 8.75 mm MPJPE (RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint, MPI-INF-3DHP clean)  
**Status:** Skeleton model + loss exist; full GPU run has **not** been executed.  
**Primary artifacts:**
- `motionflow_mv/losses/gaussian_splatting_pose_loss.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`
- `experiments/train_splat_pp_smoke_mpiinf3dhp.py`
- `experiments/train_splat_pp_full_mpiinf3dhp.py`
- `tests/test_gaussian_splatting_pose_loss.py`

---

## 1. Motivation (why this direction can beat the 8.75 mm anchor)

The current anchor model already achieves strong 3-D accuracy by combining:
1. Learned principal-point/intrinsic correction (`PrincipalPointCorrection` in `motionflow_mv/fusion/principal_point_correction.py`).
2. Spatio-temporal (time × view) attention over ray-aware embeddings.
3. A residual refinement head that adds a learned per-joint correction on top of weighted DLT triangulation.

However, the failure analysis in `docs/swarm_iter_next/failure_analysis_crossview_pp.md` shows that residual corrections are large (mean 50.57 mm) and worst joints include hands, wrists, elbows and shoulders — joints that are often poorly localized in 2-D and therefore receive noisy or low-confidence observations. The model currently predicts a single 3-D position per joint with no explicit uncertainty, so it cannot down-weight inconsistent views in a probabilistically principled way, nor can it regularize the solution to agree geometrically with all calibrated views.

Gaussian splatting retry addresses this by:

1. **Per-joint 3-D uncertainty in world space.** Instead of treating each joint as a point, we represent it as an anisotropic 3-D Gaussian with a diagonal covariance predicted per joint. Projecting this Gaussian into each view gives a 2-D Gaussian that naturally encodes how much each predicted 2-D location may deviate from the observed keypoint.
2. **Cross-view geometric consistency loss.** The negative log-likelihood (NLL) of the observed 2-D keypoints under the projected 2-D Gaussians forces the predicted 3-D pose to reproject consistently into every calibrated view. This is stronger than a simple 2-D reprojection MSE because it couples all views through a shared 3-D covariance.
3. **Down-weighting of noisy observations in 3-D.** When a view gives a contradictory keypoint, the NLL cost grows, and the model can learn to increase the predicted 3-D uncertainty (via `log_std`) to account for it. In effect, the covariance head learns a per-joint confidence in 3-D space, complementary to the existing per-view weight head.
4. **Differentiable and lightweight.** The loss is fully differentiable and the only extra module is a small MLP (`covariance_head`). The existing spatio-temporal features are reused, so the model stays within the same parameter budget as the PP-residual anchor.

The hypothesis is that, by adding a geometrically grounded probabilistic regularizer, we can reduce the large residual corrections on ambiguous joints (hands/wrists) and therefore push MPJPE below the current 8.75 mm anchor on MPI-INF-3DHP clean. The 20-agent review (`docs/swarm_iter_next/20_agent_direction_review.md`) currently ranks this as P2 because the loss module was missing; it now exists, so a proper full training run is the natural next step.

---

## 2. Architecture

### 2.1 High-level block diagram

```
Input 2-D keypoints + confidences (B, T, V, J, 3)
           │
           ▼
┌─────────────────────────┐
│ PrincipalPointCorrection│  <-- motionflow_mv/fusion/principal_point_correction.py
│ (learned PP/focal fix)  │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ _extract_frame_features │  <-- ray_attention_temporal_crossview_model.py
│ (obs + ray + camera emb)│
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ Spatio-temporal         │  <-- st_transformer (time × view attention)
│ Transformer             │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ weight_head + weighted  │  <-- _triangulate_weighted_dlt
│ DLT triangulation       │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ residual_mlp            │  <-- delta correction
└─────────────────────────┘
           │
           ▼
    pred_3d (refined)
           │
           ▼
┌─────────────────────────┐
│ covariance_head         │  <-- NEW: predicts log_std (B, T, J, 3)
└─────────────────────────
           │
           ▼
   gaussian_splatting_pose_loss
   (projects 3-D Gaussians to 2-D, NLL)
```

### 2.2 Inputs and outputs

**Model:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat`
(`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`).

Inputs (same as the PP-residual anchor):
- `x`: `(B, T, V, J, 3)` — pixel coordinates `(x, y)` and confidence `c`.
- `K`: `(B, V, 3, 3)` intrinsic matrices.
- `R`: `(B, V, 3, 3)` rotation matrices.
- `t`: `(B, V, 3)` translation vectors.

Outputs (when `return_covariance=True`):
- `pred_3d`: `(B, T, J, 3)` refined world 3-D joints.
- `weights`: `(B, T, V, J)` predicted per-view per-joint weights.
- `log_std`: `(B, T, J, 3)` log standard deviations for the per-joint anisotropic 3-D Gaussian.

### 2.3 Key equations

**3-D covariance.**

For each joint, the model predicts `log_std_{j,k} ∈ ℝ³`. The standard deviations are clamped and squared to form a diagonal covariance:

```
std_{j,k} = clamp(exp(log_std_{j,k}), 0.01, 10.0)
Σ_{3d}^{j} = diag(std_{j,1}², std_{j,2}², std_{j,3}²)
```

**Projection to 2-D.**

For a camera `(K, R, t)` and a 3-D point `X`, the projected mean is:

```
X_cam = R X + t
x = X_cam[0] / X_cam[2]
y = X_cam[1] / X_cam[2]
[u, v]ᵀ = K @ [x, y, 1]ᵀ
```

The Jacobian `J_uv = ∂(u,v)/∂X` is computed via the camera coordinate Jacobian and the rotation matrix:

```
J_uv = [[fx/z,    0,   -fx·x/z²],
        [   0, fy/z,   -fy·y/z²]]
J = J_uv · R
```

The 2-D covariance is:

```
Σ_{2d} = J · Σ_{3d} · Jᵀ + ε I₂
```

**Negative log-likelihood loss.**

For observed 2-D keypoint `p_2d` and projected mean `μ_2d`, with Mahalanobis distance:

```
r = p_2d - μ_2d
mahalanobis² = rᵀ Σ_{2d}^{-1} r
logdet = log det(Σ_{2d})
L_splat = 0.5 · (logdet + mahalanobis²)
```

The per-observation weights are `c_{v,j}` (confidences). The final loss is the confidence-weighted average over `(B, T, V, J)`. A small trace regularizer on `std` prevents collapse:

```
L_total = L_splat + λ_trace · mean(sum_k std_k)
```

In the trainer, the splat loss is added with weight `splat_loss_weight`:

```
L = L_MSE + λ_splat · L_splat
```

---

## 3. Code changes needed

### 3.1 Already implemented (no action needed)

| File | What it provides |
|------|------------------|
| `motionflow_mv/losses/gaussian_splatting_pose_loss.py` | `gaussian_splatting_pose_loss(...)` and `gaussian_splatting_render_error(...)` — fully differentiable NLL and diagnostic function. |
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py` | `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat` — subclass of the PP-residual anchor with a `covariance_head` that predicts `log_std`. |
| `tests/test_gaussian_splatting_pose_loss.py` | CPU tests for forward/backward and shape checks. |
| `experiments/train_splat_pp_smoke_mpiinf3dhp.py` | 5-epoch smoke trainer. |
| `experiments/train_splat_pp_full_mpiinf3dhp.py` | 20-epoch full trainer. |

### 3.2 Recommended additions / modifications

1. **Wire the splat model into `eval_full_metrics.py`**
   - File: `experiments/eval_full_metrics.py`
   - Action: add `"splat_pp"` to `MODEL_CLASSES` (already imported; only the dict entry is missing) and ensure the `build_model` function passes the correct kwargs.
   - Why: so a trained `splat_pp` checkpoint can be evaluated with the standard benchmark protocol.

2. **Add a multi-seed / multi-dataset benchmark manifest**
   - File: `docs/swarm_iter_next/iter_gaussian_splatting_retry_manifest.md` (new)
   - Action: document the exact seeds, training args, and evaluation commands used.
   - Why: reproducibility for ICRA/CVPR 2027 review.

3. **Optional: covariance regularization tuning**
   - File: `motionflow_mv/losses/gaussian_splatting_pose_loss.py`
   - Action: expose `trace_weight` and `eps` in the trainer CLI; currently `trace_weight=1e-4` is hard-coded.
   - Why: the 8.75 mm anchor was tuned with a fixed reprojection/MSE trade-off; the splat term may need per-hyperparameter sweeps.

4. **Optional: integrate with variable-view inference**
   - File: `motionflow_mv/fusion/variable_view_inference.py`
   - Action: verify that `VariableViewInferenceWrapper` correctly handles the extra `log_std` output of the splat model.
   - Why: variable-view robustness is a P0 direction; the splat model should not break it.

5. **Optional: failure-analysis dashboard for splat**
   - File: new script under `experiments/`
   - Action: compute per-joint `gaussian_splatting_render_error` on the MPI-INF-3DHP test set and compare to the anchor.
   - Why: validate the hypothesis that the learned 3-D uncertainty is largest exactly on the worst joints (hands/wrists).

---

## 4. Training & evaluation protocol

### 4.1 Datasets

Use the existing WebBridge/MPI-INF-3DHP preprocessed `.npz` files:

- **Train:**
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz`
- **Val:**
  - `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`

This matches the anchor trainer (`experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`).

### 4.2 Training

Run the smoke trainer first:

```bash
python experiments/train_splat_pp_smoke_mpiinf3dhp.py
```

Then the full run:

```bash
python experiments/train_splat_pp_full_mpiinf3dhp.py
```

Key hyperparameters from `train_splat_pp_full_mpiinf3dhp.py`:

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `clip_len` | 13 | Same temporal window as anchor. |
| `d` | 64 | Same hidden dimension as anchor. |
| `residual_hidden` | 128 | Same residual MLP as anchor. |
| `n_st_layers` | 2 | Same spatio-temporal depth as anchor. |
| `pp_loss_weight` | 0.2 | Supervise principal-point correction. |
| `splat_loss_weight` | 0.01 | Auxiliary Gaussian-splatting term (start conservative). |
| `reproj_weight` | 0.0 | Disable redundant MSE reprojection so the splat NLL is the dominant geometric term. |
| `cam_aug_pp` | 5.0 px | Principal-point perturbation to force PP head learning. |
| `cam_aug_focal` | 0.01 | Focal-length perturbation. |
| `cam_aug_schedule` | `intrinsics_curriculum` | Ramp PP/focal aug over 5 epochs. |
| `pp_pretrain_epochs` | 3 | Pre-train PP head before end-to-end. |
| `epochs` | 20 | Full training. |

### 4.3 Loss

The total loss in the trainer is:

```
L = MSE(pred_3d, gt_3d)
  + λ_pp · MSE(pp_delta, -true_pp_delta)
  + λ_splat · gaussian_splatting_pose_loss(pred_3d, points_2d, K, R, t, log_std, confidences)
```

### 4.4 Metrics

Evaluate with `experiments/eval_full_metrics.py` (after wiring `splat_pp`) or directly with `BenchmarkProtocol`:

- MPJPE (mm)
- PA-MPJPE (mm)
- PCK@50/100/150 mm
- PCK-AUC
- Per-joint MPJPE breakdown
- Per-view reprojection error (diagnostic)
- Covariance head diagnostics: mean predicted `std` per joint, correlation with final MPJPE

### 4.5 Baseline to compare

Run the standard PP-residual anchor with the same hyperparameters (except `splat_loss_weight=0.0`) so the comparison is fair:

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
          data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
          data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
          data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
    --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
    --pp_pretrain_epochs 3 --epochs 20 \
    --output outputs/pp_anchor_for_splat_compare.pth
```

The target is to beat **MPJPE < 8.75 mm** on MPI-INF-3DHP clean.

---

## 5. Expected gains and risks

### 5.1 Expected gains

| Source of gain | Mechanism | Estimate |
|----------------|-----------|----------|
| Reduced hand/wrist error | 3-D covariance regularizer down-weights geometrically inconsistent 2-D detections | 0.2–0.5 mm MPJPE |
| Better PP/focal stability | Splat NLL provides a second cross-view signal that constrains intrinsics | 0.1–0.3 mm |
| More calibrated confidences | Per-joint `log_std` correlates with true error, enabling post-hoc thresholding | Metric / robustness improvement |
| Low compute overhead | Only `covariance_head` added; loss is a few extra matrix ops per clip | <5% training slowdown |

**Realistic combined target:** 8.20–8.50 mm MPJPE on MPI-INF-3DHP clean if the hypothesis holds.

### 5.2 Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `splat_loss_weight` too high and drowns MSE | Medium | Sweep `0.001, 0.005, 0.01, 0.02` on smoke data; start with 0.01. |
| Cholesky instability in `gaussian_splatting_pose_loss` | Low | `_robust_cholesky` already handles jitter/eigendecomposition fallback. Monitor for NaN. |
| Covariance head underfits or collapses to isotropic blobs | Medium | Initialize `covariance_head` last layer to small values (near log(0.05) ≈ -3) so initial std ≈ 5 cm. |
| Gains are smaller than Bayesian Triangulation, which is already running | High | This is an *alternative candidate*, not a replacement. Run both and compare on the same validation split. |
| Conflicts with running GPU run | N/A | Do **not** start this training while the Bayesian Tri run is active on the RTX 4090. |

---

## 6. Next steps (what a follow-up implementer should do first)

1. **Run the smoke test.**
   ```bash
   python experiments/train_splat_pp_smoke_mpiinf3dhp.py
   ```
   Verify it completes without NaN and that `outputs/splat_pp_smoke.pth` is produced.

2. **Wire `splat_pp` into the standard evaluator.**
   - Add `"splat_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat` to `MODEL_CLASSES` in `experiments/eval_full_metrics.py`.
   - Run a CPU-only eval on the smoke checkpoint to confirm the forward pass and metric reporting work.

3. **Hyperparameter sweep on smoke.**
   - Vary `splat_loss_weight ∈ {0.005, 0.01, 0.02}`.
   - Vary `trace_weight ∈ {1e-5, 1e-4, 1e-3}`.
   - Pick the setting that gives the best validation MPJPE after 5 epochs.

4. **Launch the full 20-epoch run once the RTX 4090 is free.**
   ```bash
   python experiments/train_splat_pp_full_mpiinf3dhp.py
   ```

5. **Evaluate and compare to the anchor.**
   - Run `eval_full_metrics.py` for both `splat_pp` and the PP-residual anchor on `s_02_seq_01_v14_multiview_m.npz`.
   - Report MPJPE, PA-MPJPE, PCK, AUC, and per-joint breakdown.
   - If `splat_pp` beats 8.75 mm, run a multi-seed (3 seeds) confirmation.

6. **If promising, add variable-view and robustness tests.**
   - Use `motionflow_mv/fusion/variable_view_inference.py` to test `V=2,3,4`.
   - Use `motionflow_mv/eval/robustness.py` (or add Gaussian noise/outliers) to check whether the covariance head improves robustness.

7. **Write a short report.**
   - Save to `docs/swarm_iter_next/iter_gaussian_splatting_retry_report.md` with the final metrics, ablation table, and a decision on whether to merge the splat loss into the main anchor.
