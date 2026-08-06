# Proposal: Reprojection-Consistency Loss for Learned Intrinsic Correction

**Agent task:** Reprojection-consistency loss: add differentiable 2D reprojection term

---

## 1. Problem

The learned principal-point/focal correction in `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` is driven almost entirely by the downstream 3-D MSE loss, so the correction head can under- or over-compensate for intrinsic errors without ever being directly constrained by 2-D observation consistency.

## 2. Hypothesis

Adding a robust, differentiable 2-D reprojection-consistency term—computed on both the raw triangulated 3-D pose and the refined 3-D pose—will pull the corrected intrinsics and the final prediction toward true multi-view geometric consistency, improving clean accuracy and robustness to `cx/cy`/focal perturbations at no extra model capacity.

## 3. Method

### 3.1 New loss module

Create `motionflow_mv/losses/reprojection_consistency.py` with two functions:

- `reprojection_error(pred_3d, points_2d, K, R, t, confidences=None, mask=None, eps=1e-6)` — returns per-sample, per-view, per-joint reprojection error in pixels.
- `robust_reprojection_loss(..., loss_type="charbonnier", alpha=1.0)` — wraps the error in a robust loss (Charbonnier by default, Huber optional) and supports per-joint confidence and view/joint dropout masks.

Key differences from the existing `motionflow_mv/losses/reprojection.py`:

1. Returns the raw per-joint error tensor (useful for gating and diagnostics).
2. Supports robust norms to reduce sensitivity to 2-D outliers.
3. Accepts an explicit validity mask for occluded/dropped joints.
4. Projects the predicted 3-D joints using the *corrected* intrinsics output by `PrincipalPointCorrection`, not the raw calibration.

### 3.2 Model changes

Modify `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`:

- Return the raw triangulated 3-D pose `pred_3d_raw` alongside the refined `pred_3d` when `return_raw=True` is passed.
- Exact edit: after line `pred_3d = pred_3d_raw + delta`, store `raw_3d = pred_3d_raw.view(B, T, J, 3)` and return it as an extra tuple element when `self.return_raw` is true.

No other architecture changes are needed.

### 3.3 Training script changes

Modify `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`:

- Add arguments:
  - `--reproj_raw_weight` (float, default `0.05`)
  - `--reproj_refined_weight` (float, default `0.1`)
  - `--reproj_robust` (flag, use Charbonnier instead of MSE)
  - `--reproj_mask_dropout` (flag, mask zero-confidence joints)
- In the training loop, after the model forward:
  - If `args.reproj_raw_weight > 0`, compute `robust_reprojection_loss(raw_3d, xb[..., :2], K, R, t, confidences=conf, ...)` and add with weight `reproj_raw_weight`.
  - If `args.reproj_refined_weight > 0`, compute the same loss on `pred` (the refined output) and add with weight `reproj_refined_weight`.
- Keep the existing `--reproj_weight` path on the refined output for backward compatibility.

### 3.4 New smoke trainer

Create `experiments/train_reprojection_consistency_pp_smoke_mpiinf3dhp.py`:

- Thin wrapper around the modified PP trainer above.
- Runs 5 epochs on a 500-sample train split with `d=32`, `residual_hidden=64`, and enables both raw and refined reprojection losses.

## 4. Smoke-test plan

Run `experiments/train_reprojection_consistency_pp_smoke_mpiinf3dhp.py` for **5 epochs** on a small MPI-INF-3DHP split:

- Train: `s_01_seq_01_v14_multiview_m.npz` (500 random clips)
- Val: `s_02_seq_01_v14_multiview_m.npz`
- Config: `d=32`, `residual_hidden=64`, `clip_len=13`, `batch_size=8`, `reproj_raw_weight=0.05`, `reproj_refined_weight=0.1`

**Pass criteria:**

- No NaNs or runtime crashes.
- Val MPJPE ≤ 60 mm after 5 epochs (smoke threshold; the factorized ST+PP smoke hit 57.68 mm under similar conditions).
- Reprojection loss tensor is finite and decreases monotonically for the first 3 epochs.
- Corrected intrinsics remain within `max_offset` / `max_focal_scale` bounds.

**Fail criteria:**

- Val MPJPE > 80 mm or any NaN/Inf.
- Reprojection loss does not decrease after the first epoch.
- Final 3-D MPJPE is worse than a baseline run with `--reproj_weight 0` by > 5 mm.

## 5. Evaluation plan

Use the existing evaluation harness:

- `experiments/compare_sota_baselines.py` — run on the smoke checkpoint to report MPJPE/PA-MPJPE vs. the 9.32 mm anchor.
- `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` — run a 20-clip robustness matrix on the `cxcy_3px`, `cxcy_5px`, `focal_1pct`, and `focal_2pct` axes to verify that reprojection consistency improves intrinsic-corruption robustness.
- Add a short diagnostic in `experiments/analyze_failures_crossview_pp.py` to report mean per-view reprojection error before/after correction; pass if the corrected reprojection error is lower.

Metrics to report:

- Clean val MPJPE / PA-MPJPE
- Robustness matrix MPJPE on `cxcy_3px`, `cxcy_5px`, `focal_1pct`, `focal_2pct`
- Mean per-view reprojection error (pixels) before and after intrinsic correction

## 6. Estimated GPU/CPU cost on RTX 4090

- **Smoke (5 epochs, 500 samples):** ~12–18 minutes on RTX 4090.
- **Full 30-epoch run (if smoke passes):** ~4–6 hours on RTX 4090.
- **Robustness evaluation:** CPU-only, ~5–10 minutes for the 6-axis matrix on a small subset.

The reprojection loss adds negligible compute (one extra projection per forward pass), so memory and throughput are essentially unchanged.

## 7. Risks & fallback

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Reprojection loss dominates the 3-D MSE early on and distorts the pose scale | Medium | Start with small weights (`0.05`/`0.1`) and ramp with `intrinsics_curriculum`; if it still dominates, disable the raw-pose term and keep only refined-pose reprojection. |
| The loss is sensitive to outliers/2-D augmentation noise | Medium | Use Charbonnier robust loss; if needed, clip reprojection error at 50 px. |
| No improvement over existing `--reproj_weight` path | Medium-High | Fallback: the existing `reprojection_loss` in `motionflow_mv/losses/reprojection.py` is sufficient; this proposal mainly adds robustness and raw-pose supervision. If smoke shows no gain, abandon the new module and instead tune the existing weight. |
| Camera augmentation noise makes the reprojection target itself noisy | Low | Augmentation is already applied before reprojection; the loss is computed on the augmented 2-D points, which is the correct supervised target. |

---

## Summary

Add a robust differentiable 2-D reprojection-consistency loss applied to both raw and refined 3-D predictions, using the corrected intrinsics from `PrincipalPointCorrection`. Smoke-test on the existing PP trainer for 5 epochs; if it passes, run the 6-axis robustness matrix and SOTA comparison before committing to a full 30-epoch run.
