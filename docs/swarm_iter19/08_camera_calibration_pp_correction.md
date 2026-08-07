# 08 — Camera Calibration / Principal-Point Post-Processing Correction

## Summary

This subtask covers learned correction of camera intrinsics—principally principal-point (PP) offsets and focal length—inside the multi-view pose pipeline. Imperfect `K` is common in real rigs and can dominate triangulation error. The repo already has `PrincipalPointCorrection`, `IntrinsicCorrection`, and camera-perturbation curricula, but calibration robustness remains a major gap.

## Current state

**What exists and works**

- `motionflow_mv/fusion/principal_point_correction.py` (lines 22–142) implements a bounded, per-view PP correction head: `delta = tanh(out) * max_offset`, then adds the offset to `K[..., 0:1, 2]`.
- `motionflow_mv/fusion/intrinsic_correction.py` (lines 19–106) extends this to also predict a focal-length scale, correcting both PP and focal length in one MLP.
- `motionflow_mv/calibration/camera_perturbation_curriculum.py` provides schedules `flat`, `extrinsic_curriculum`, `intrinsics_curriculum`, `extended_curriculum`, and `extended_intrinsics_curriculum`.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` exposes `--pp_loss_weight`, `--cam_aug_pp`, `--cam_aug_focal`, and `--cam_aug_schedule` (lines 278–290, 601–613).
- `tests/test_camera_perturbation_curriculum.py` provides CPU smoke coverage.

**What does not work yet**

- The curriculum-only checkpoint (`outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth`) degrades clean accuracy to **10.76 mm** vs. the 9.32 mm PP baseline and does not fix rotation robustness (`rot_0.5` = 26.92 mm).
- `pp_10px` still fails catastrophically at **2070.33 mm** MPJPE (`outputs/eval_curriculum_robustness_final.json`).
- The best single model (Bayesian Tri v2) still has large gaps: `rot_0.5` = 16.89 mm, `focal_1%` = 19.13 mm (`docs/results_icra_cvpr_2027.md`).
- `OmniMultiViewFusionV2` no-graph ablation is training but currently shows ~44–46 mm validation after the freeze phase—far from ready.

## Key findings

1. **PP correction helps small offsets but saturates.** `cxcy_3px` and `cxcy_5px` are handled well, but `pp_10px` is catastrophic. The default `max_offset=20` should be sufficient, so the failure is likely triangulation instability when the predicted correction is wrong.
2. **Focal-length robustness is poor.** `focal_1%` jumps to 19.13 mm on the baseline; the curriculum improves it to 10.66 mm only by sacrificing ~1.5 mm clean accuracy.
3. **Curriculum + view dropout hurt clean accuracy.** The curriculum checkpoint traded clean MPJPE for modest focal robustness and no meaningful rotation gain.
4. **`IntrinsicCorrection` is not yet integrated into the main models.** `principal_point_correction.py` is the one used by `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` and its Bayesian Tri descendants (`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`, line 59).

## Recommendations

1. **Switch main models to `IntrinsicCorrection`.** Replace `PrincipalPointCorrection` with `IntrinsicCorrection` so focal and PP corrections share features and can be supervised jointly via the existing `focal_loss_weight` path.
2. **Enable and tune focal correction.** Set `--focal_max_scale 0.05` and `--focal_loss_weight 0.1–0.2`; the loss path is already implemented at lines 608–613.
3. **Add a bounded rotation/extrinsic correction head or stronger extrinsic curriculum.** Rotation is the largest remaining gap. Either add a small SO(3) residual head or ramp `cam_aug_rot` to 2° with a 5-epoch warmup.
4. **Bound corrections and add a reprojection guard.** For `pp_10px`, clamp corrections or fall back to the uncorrected `K` when reprojection error explodes.
5. **Warm-start from the 8.35 mm ensemble and pre-train the intrinsic head.** Use the existing `--pp_pretrain_epochs` flag (line 507) to freeze the encoder and train only the correction head for 3–5 epochs before end-to-end fine-tuning.
6. **Add a CPU calibration smoke benchmark.** Evaluate a grid of PP 1–10 px, focal 0.5–2%, and rot 0.1–1° on the smoke dataset before GPU time.

## Open questions

- Does joint focal+PP correction improve `focal_1%` without degrading clean accuracy?
- Is the `pp_10px` failure due to correction-head saturation, triangulation degeneracy, or noisy large-offset targets?
- Can a stronger extrinsic perturbation curriculum push `rot_0.5` below 12 mm, or is an explicit rotation correction head required?
- Does the no-graph OmniMultiViewFusionV2 ablation recover after unfreezing, or does the intrinsic head need re-initialization from the 8.35 mm ensemble member?
