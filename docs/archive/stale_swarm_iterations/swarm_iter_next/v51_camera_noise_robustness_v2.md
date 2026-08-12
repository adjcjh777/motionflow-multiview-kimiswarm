# v51 Camera Noise Robustness v2

## Motivation

The v46/v48 pipeline assumes calibration-quality camera parameters, yet real-world deployment (hand-held rigs, 3DPW actual mode, sports broadcasts) faces calibration drift and extrinsic noise. In sparse-view settings, triangulation lacks redundancy, so small camera errors become large 3D joint offsets. This module adds synthetic calibration-noise augmentation during training and a lightweight, identity-initialized per-camera correction head learned from reprojection residuals. It extends the v50 "Camera Calibration Noise Robustness" idea with bounded corrections and per-camera reliability, making it a natural v51 step after the v50 SEFH self-evolution loop.

## Proposed module

**Module / file path**: `CameraNoiseRobustnessV2` → `motionflow_mv/fusion/camera_noise_robustness_v51.py`

**Architecture**: A per-camera two-layer MLP consumes the current 3D pose estimate, the noisy camera matrices (R, t, K), and the per-view reprojection residual, then predicts a bounded 6-DoF extrinsic correction (axis-angle δr and translation δt). The corrections are added to the input cameras before triangulation and are clamped to a small magnitude. An optional per-camera reliability scalar multiplies into the v46 view-reliability path. The correction head is zero-initialized and warm-started from the best v46/v48 checkpoint, so the baseline is preserved at startup.

**Key design choices**
- Identity-initialized correction head (default zero correction).
- Bounded output to prevent over-correction.
- Synthetic noise schedule for camera R/t during training.
- Optional intrinsic correction disabled by default.

## New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_camera_noise_robustness_v2` | bool | `False` |
| `v51_cnr_v2_hidden` | int | `64` |
| `v51_cnr_v2_num_layers` | int | `2` |
| `v51_cnr_v2_dropout` | float | `0.1` |
| `v51_cnr_v2_max_rotation_correction` | float | `0.05` |
| `v51_cnr_v2_max_translation_correction` | float | `0.10` |
| `v51_cnr_v2_synthetic_rotation_noise_deg` | float | `0.5` |
| `v51_cnr_v2_synthetic_translation_noise_m` | float | `0.05` |
| `v51_cnr_v2_noise_schedule` | str | `"constant"` |
| `v51_cnr_v2_use_intrinsic_correction` | bool | `False` |
| `v51_cnr_v2_reproj_loss_weight` | float | `0.01` |
| `v51_cnr_v2_correction_reg_weight` | float | `0.001` |

## Loss term

```
L_cnr = v51_cnr_v2_reproj_loss_weight * Huber(reproj_residual, δ)
      + v51_cnr_v2_correction_reg_weight * ||correction||_2
```

The main supervised 3D pose loss is unchanged; the camera correction is trained only through reprojection residuals, making it easy to graft onto the v46/v48 graph.

## Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full` on standard validation.
- Noisy-camera robustness: `MPJPE@2_with_noise`, measured after injecting synthetic noise and comparing corrected vs. uncorrected outputs.
- Mean absolute correction and `Spearman(correction, injected_noise)` to confirm the head responds to real noise.

## Expected MPJPE impact

- `MPJPE@2` under injected calibration noise: **−3 to −6 mm** vs. uncorrected v46.
- `MPJPE@3/4`: **−1 to −3 mm**.
- Full-view `val_MPJPE`: within **±0.5 mm** of baseline (identity default).
- 3DPW actual: modest gain due to noisier real-world calibration.

## Main risk and mitigation

**Risk**: The head may hallucinate camera corrections that distort geometry. **Mitigation**: identity-at-init, clamped corrections, freeze base model for the first epoch, and begin with `v51_cnr_v2_reproj_loss_weight=0.001`. Ablate with/without injected noise to confirm the head is noise-driven.

## Paper-story fit

This extends the self-evolution narrative from "the model critiques its own views" to "the model repairs its own cameras." It directly supports the cross-domain claim (studio to in-the-wild) and strengthens sparse-view robustness, where camera errors have no redundancy to hide behind.
