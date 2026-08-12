# v54 Differentiable Bundle Adjustment (DBA)

## 1. Motivation

v52 Uncertainty-Weighted Triangulation reweights views per joint, and v53 Physical-Space Calibration enforces floor and bone-length constraints, but both still treat the camera parameters `K, R, t` as fixed inputs from an offline calibration stage. In the paper pipeline **multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow**, the calibration step is currently one-way: cameras calibrate people, not the other way around. In real captures, small extrinsic drift, lens distortion, and synchronization jitter are common, and they cap the downstream MPJPE even when the pose network is perfect.

v54 Differentiable Bundle Adjustment closes this loop. It places a lightweight, fully differentiable bundle-adjustment block **after** v53 (or v52 when v53 is disabled) and jointly refines the 3-D pose and the camera parameters using the 2-D observations and the v52 per-joint weights. The block is warm-startable/identity-at-init: all camera and pose corrections are initialized to zero, so loading a v53 checkpoint with v54 enabled leaves the pose unchanged before training.

## 2. Architecture

DBA sits in `OmniMultiViewFusionV5` immediately after `PhysicalSpaceCalibrationV53` and before the final output. It contains two sub-networks:

1. **Camera correction head** — predicts a small, per-view correction to intrinsics and extrinsics.
2. **Pose refinement head** — predicts a small, per-joint residual on the 3-D pose, gated to be zero at initialization.

Camera correction is represented as a multiplicative/additive delta around the input cameras:

\[
\begin{aligned}
K_v' &= K_v \cdot \mathrm{diag}(\exp(\Delta f_v), \exp(\Delta f_v), 1) + \begin{bmatrix} 0 & 0 & \Delta c_x^v \\ 0 & 0 & \Delta c_y^v \\ 0 & 0 & 0 \end{bmatrix}, \\
R_v' &= \exp([\Delta r_v]_\times) \, R_v, \\
t_v' &= t_v + \Delta t_v,
\end{aligned}
\]

where \(\Delta f_v \in \mathbb{R}\), \((\Delta c_x^v, \Delta c_y^v) \in \mathbb{R}^2\), \(\Delta r_v \in \mathbb{R}^3\), and \(\Delta t_v \in \mathbb{R}^3\). The pose residual is

\[
X' = X + g \cdot \mathrm{MLP}_{\mathrm{res}}([X, h_{\mathrm{floor}}, h_{\mathrm{bone}}, e_{\mathrm{reproj}}])
\]

with the gate initialized so that \(g = \sigma(g_{\text{init}}) \approx 0\). The DBA loss is a robust reprojection term plus regularizers:

\[
\mathcal{L}_{\text{DBA}} = \frac{1}{BTVJ_{\text{vis}}} \sum_{b,t,v,j} w_{b,t,v,j} \, \rho\bigl(\pi(K_v', R_v', t_v'; X'_j) - x_{b,t,v,j}\bigr) + \lambda_{\text{cam}} \|\Delta c\|^2 + \lambda_{\text{pose}} \|\Delta X\|^2,
\]

where \(\rho\) is a Huber robustifier, \(w_{b,t,v,j}\) are the v52 UWT weights, and \(\pi(\cdot)\) is the perspective projection.

## 3. Inputs / Outputs (tensor shapes)

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Input 3-D pose from v53 (or v52) |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoints |
| `K` | `(B, T, V, 3, 3)` | Intrinsics |
| `R` | `(B, T, V, 3, 3)` | Rotations |
| `t` | `(B, T, V, 3)` | Translations |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view per-joint weights |
| `view_mask` | `(B, T, V)` | Visibility mask |
| `domain_id` | `(B,)` | Domain labels (optional) |
| **Output** `pred_3d_dba` | `(B, T, J, 3)` | Bundle-adjusted 3-D pose |
| **Output** `K_dba` | `(B, T, V, 3, 3)` | Refined intrinsics |
| **Output** `R_dba` | `(B, T, V, 3, 3)` | Refined rotations |
| **Output** `t_dba` | `(B, T, V, 3)` | Refined translation |
| **Output** `dba_loss` | scalar | Auxiliary reprojection + regularization loss |

## 4. Config flags

```
use_v54_differentiable_bundle_adjustment: bool = False
v54_dba_hidden: int = 64
v54_dba_n_layers: int = 2
v54_dba_residual_gate_init: float = -6.0
v54_dba_correct_intrinsics: bool = True
v54_dba_correct_extrinsics: bool = True
v54_dba_use_uwt_weights: bool = True
v54_dba_use_psc_hints: bool = True
v54_dba_huber_delta: float = 5.0
v54_dba_camera_reg_weight: float = 0.1
v54_dba_pose_reg_weight: float = 0.01
v54_dba_loss_weight: float = 0.01
v54_dba_warmup_epochs: int = 0
v54_dba_identity_init: bool = True
```

## 5. Expected MPJPE impact

- **Full-view inference:** 0.4–1.0 mm gain by correcting small calibration biases that survive v53.
- **Sparse/variable-view inference (`MPJPE@2`, `MPJPE@3`):** 1.0–2.5 mm gain, because joint camera/pose refinement reweights the remaining views more accurately than fixed cameras.
- **Warm-start verification:** loading a v53 checkpoint with v54 enabled and no training should change `val_MPJPE@full` by ≤ 0.1 mm.

## 6. Risks

1. **Camera correction overfitting** — learned intrinsics/extrinsics may absorb dataset-specific calibration and hurt cross-dataset transfer.
2. **Joint camera/pose instability** — optimizing both simultaneously can diverge without strong regularization.
3. **Compute cost** — computing per-view Jacobians for the reprojection loss adds memory and runtime.
4. See `docs/swarm_iter28/reports/agent_differentiable_bundle_adjustment_risks.md` for the full register and mitigations.

## 7. 5-step implementation plan

1. **Module:** create `motionflow_mv/fusion/differentiable_bundle_adjustment_v54.py` with `DifferentiableBundleAdjustmentV54`, implementing the camera correction MLP, pose residual MLP, and robust reprojection loss.
2. **Wiring:** in `OmniMultiViewFusionV5.__init__` add the v54 flags and instantiate the module; in `forward` call it after v53 PSC (or after v52 if v53 is disabled).
3. **Loss:** register `v54_dba_loss` in `forward` and add it to the total loss inside `get_loss` with the `v54_dba_warmup_epochs` gate.
4. **Smoke test:** add `configs/benchmark_v54_dba_smoke.yaml`; run on RTX 4090 and verify identity-at-init (Δ MPJPE ≤ 0.1 mm) and one epoch of stable training.
5. **Ablate:** compare `v52+v53` vs `v52+v53+v54` on the A800 queue; report `MPJPE@full` and `MPJPE@2/3/4`, and inspect per-camera correction magnitudes to confirm they remain small.
