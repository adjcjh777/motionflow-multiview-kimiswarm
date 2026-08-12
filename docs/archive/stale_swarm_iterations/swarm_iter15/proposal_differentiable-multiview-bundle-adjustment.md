# Differentiable Multi-View Bundle Adjustment (iter15 proposal)

## One-sentence hypothesis

Adding a lightweight, differentiable structure-only bundle-adjustment (DBA) refinement on top of the existing learned triangulation lets the network explicitly minimize per-view reprojection error, improving physical alignment and robustness to noisy/perturbed cameras while preserving the temporal/cross-view attention backbone.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — current iter14 anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, clean MPJPE 9.32 mm on MPI-INF-3DHP S2/Seq1).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` — parent with residual refinement head.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` — spatio-temporal attention backbone.
- `motionflow_mv/fusion/ray_attention_model.py` — `_triangulate_weighted_dlt` and ray helpers.
- `motionflow_mv/fusion/principal_point_correction.py` — learned intrinsic correction layer used by the anchor.
- `motionflow_mv/losses/reprojection.py` and `motionflow_mv/losses/reprojection_consistency.py` — reprojection losses already used for auxiliary supervision.
- `motionflow_mv/calibration/perturb.py` — training-time camera perturbations.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — anchor training script.

## Proposed code changes

### 1. New file: `motionflow_mv/fusion/differentiable_bundle_adjustment.py`

Implements `DifferentiableBundleAdjustment`.

- **Input**: initial 3D points `(B, T, J, 3)`, 2D observations `(B, T, V, J, 2)`, intrinsics `K`, rotations `R`, translations `t`, and predicted per-view weights `(B, T, V, J)`.
- **Operation**: performs `n_iters` (default 2) Gauss-Newton/Levenberg-Marquardt steps on the 3D structure to minimize the weighted reprojection residual.
- **Output**: refined 3D points `(B, T, J, 3)`.
- **Key properties**:
  - Structure-only: cameras remain fixed, so the change is a drop-in refinement layer.
  - Analytic 2×3 Jacobians avoid `autograd.functional` overhead.
  - Damping factor is learned or fixed; each LM step is fully differentiable.

### 2. New file: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bundle_adjustment_model.py`

Implements `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBundleAdjustment`.

- Subclasses the iter14 anchor.
- Signature: same `__init__` plus `dba_iters` (default 2) and `dba_damping` (default 1.0).
- Forward flow:
  1. Run anchor forward to obtain `pred_3d`, `weights` (and optionally `pred_3d_raw`, `pp_delta`, `focal_scale`).
  2. Feed `pred_3d` into `DifferentiableBundleAdjustment` together with the 2D observations and (corrected) cameras.
  3. Return the DBA-refined pose, original weights, and optional diagnostics.

### 3. Training integration (no existing file edits required for smoke)

- New training script (optional, can reuse anchor script with `--model_type bundle_adjustment`): `experiments/train_ray_attention_temporal_crossview_residual_principal_point_bundle_adjustment_mpiinf3dhp.py`.
- Losses:
  - Primary 3D MSE on DBA-refined output.
  - Optional auxiliary robust reprojection loss on refined output (`reproj_refined_weight > 0`).
  - Existing PP/focal losses remain unchanged.
- Camera perturbations: reuse existing `perturb_cameras_with_delta` (rot/trans/pp/focal).

## Training/smoke plan

Use the existing MPI-INF-3DHP S2/Seq1 smoke setup (matching anchor).

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --reproj_refined_weight 0.1 \
  --epochs 5 --batch_size 8
```

- Smoke run: 1–2 epochs on a single RTX 4090 (≈10–15 min/epoch at `batch_size=8`, `clip_len=13`).
- Full reproducibility run: 5 epochs (≤5 h on RTX 4090), then evaluate on MPI-INF-3DHP S2/Seq1 and cross-validate on the read-only A800-D server.

## Success metrics

- **Primary**: clean MPJPE on MPI-INF-3DHP S2/Seq1 < 9.32 mm (beat anchor) or within 0.5 mm with lower variance.
- **Robustness axis**: evaluation under amplified camera perturbations (rot std 1.0°, trans std 0.02 m, pp std 10 px, focal std 2%) compared to the anchor; target ≥10% relative improvement in perturbed MPJPE.
- **Reprojection consistency**: mean per-view reprojection error of the refined pose on validation set < 2 px.
- **Risk/fallback**:
  - *Risk*: Gauss-Newton can be numerically sensitive when all views are nearly coplanar or depth is very small.
  - *Fallback*: if DBA diverges, reduce `n_iters` to 1, clamp updates, or fall back to the anchor’s residual-refined output (gating with a learned mixing weight).
  - *Risk*: extra compute per forward pass.
  - *Fallback*: make DBA optional at inference (`use_dba=False`) and train with the anchor checkpoint warm-start.
