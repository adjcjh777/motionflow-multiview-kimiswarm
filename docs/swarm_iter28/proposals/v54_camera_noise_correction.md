# v54 Camera Noise Correction (CNC)

## Motivation

v52 Uncertainty-Weighted Triangulation and v53 Physical-Space Calibration refine 3-D poses, but they treat the camera matrices and 2-D observations as trusted inputs. In practice, calibrated rigs exhibit small extrinsic drift, intrinsics can be off by a few pixels, and 2-D keypoint detectors produce per-joint jitter. These errors propagate through DLT triangulation and degrade the downstream physical-space alignment. v54 closes the loop by learning a lightweight, identity-at-init correction head that refines the inputs fed into v52/v53. It builds directly on the v52/v53 foundation: v54 cleans the camera/observation space, then v52 re-triangulates with better weights and v53 calibrates against physical invariants.

## Architecture

`CameraNoiseCorrectionV54` is inserted after the initial triangulation block (v25/v45) and before v52 UWT. It consumes the initial 3-D estimate `pred_3d_init`, the raw 2-D keypoints, the camera parameters, and the feature tokens, and predicts small bounded corrections. The corrected quantities then flow through v52 and v53 as usual.

Per-view, per-joint input features are built from:

- the feature token `f_{v,j} \in \mathbb{R}^{d}` (`d = model feature dim`);
- reprojection residual `r_{v,j} = \|p_{v,j} - \Pi(K_v, R_v, t_v, X_j)\|_2`;
- log-residual `\log(r_{v,j} + \epsilon)`;
- ray direction from camera center through the 2-D point;
- camera center `c_v = -R_v^{\top} t_v`.

Two lightweight MLPs share these features:

1. **2-D keypoint correction head** predicts per-joint 2-D offsets:
   \[
   \Delta p_{v,j} = \tanh(\text{MLP}_{2d}(f_{v,j}, r_{v,j}, \log r_{v,j}, \text{ray}_{v,j}, c_v)) \cdot p_{\max}
   \]
   where `p_max = v54_cnc_max_2d_offset_px`.

2. **Camera correction head** pools features over joints per view and predicts:
   - focal scale `\Delta f_v \in [-f_{\max}, f_{\max}]`;
   - principal-point offset `\Delta o_v \in \mathbb{R}^2` bounded by `o_max`;
   - rotation correction axis-angle `\Delta \omega_v` with magnitude bounded by `v54_cnc_max_rot_deg`;
   - translation offset `\Delta t_v` bounded by `v54_cnc_max_t_mm`.

Applied corrections:

\[
K'_v = K_v \cdot \text{diag}(1 + \Delta f_v, 1 + \Delta f_v, 1), \quad o'_v = o_v + \Delta o_v
\]
\[
R'_v = \exp([\Delta \omega_v]_\times) R_v, \quad t'_v = t_v + \Delta t_v
\]

All final MLP layers are zero-initialized and a global residual gate `g = \sigma(v54_cnc_residual_gate_init)` is applied so that, at initialization, `K' = K`, `R' = R`, `t' = t`, and `p' = p`. This makes v54 warm-startable from any existing v52/v53 checkpoint.

## Inputs and Outputs

Inputs:

- `points_2d`: `(B, T, V, J, 2)` raw 2-D keypoints.
- `K`: `(B, T, V, 3, 3)` intrinsics.
- `R`: `(B, T, V, 3, 3)` rotations.
- `t`: `(B, T, V, 3)` translations.
- `features`: `(B, T, V, J, d)` feature tokens.
- `pred_3d_init`: `(B, T, J, 3)` initial 3-D estimate from v25/v45.
- `view_mask`: `(B, T, V)` bool mask.

Outputs:

- `points_2d_corr`: `(B, T, V, J, 2)` corrected 2-D keypoints.
- `K_corr`, `R_corr`, `t_corr`: corrected camera parameters, same shapes as inputs.
- `cnc_loss`: scalar auxiliary loss combining reprojection consistency and correction magnitude regularization.

## Config Flags

```yaml
use_v54_camera_noise_correction: false
v54_cnc_hidden: 64
v54_cnc_n_layers: 2
v54_cnc_correct_2d: true
v54_cnc_correct_intrinsics: true
v54_cnc_correct_extrinsics: true
v54_cnc_max_2d_offset_px: 10.0
v54_cnc_max_focal_scale: 0.05
v54_cnc_max_pp_offset_px: 10.0
v54_cnc_max_rot_deg: 2.0
v54_cnc_max_t_mm: 50.0
v54_cnc_identity_init: true
v54_cnc_residual_gate_init: -6.0
v54_cnc_loss_weight: 0.1
v54_cnc_reg_weight: 0.01
v54_cnc_warmup_epochs: 0
```

## Expected MPJPE Impact

- On H36M/WebBridge with near-perfect calibration: roughly neutral to +0.3 mm improvement, mainly from reduced 2-D jitter.
- On MPI-INF-3DHP / 3DPW / real-world domain-shifted data: 0.5–1.5 mm reduction by absorbing small camera errors.
- On deliberately noisy or imperfectly calibrated inputs: up to 2.5 mm reduction.

## Risks

See `docs/swarm_iter28/reports/agent_camera_noise_correction_risks.md`.

## Implementation Plan

1. Implement `motionflow_mv/fusion/camera_noise_correction_v54.py` with identity-at-init final layers, bounded corrections, and the shared feature builder.
2. Wire the module into `OmniMultiViewFusionV5.__init__` with the flags above; instantiate it between the initial triangulation block and v52 UWT.
3. In `OmniMultiViewFusionV5.forward`, call `CameraNoiseCorrectionV54` on the raw inputs, pass the corrected `points_2d`, `K`, `R`, `t` to v52 and v53, and store the returned `cnc_loss`.
4. Add `v54_cnc_loss_weight * cnc_loss` to the total loss in `compute_loss`, gated by `v54_cnc_warmup_epochs`.
5. Smoke test with `configs/benchmark_v54_cnc_smoke.yaml`: verify identity-at-init (MPJPE within 0.1 mm of the v52/v53 baseline), then verify that reprojection residuals decrease during training and that full-view MPJPE improves or stays neutral.
