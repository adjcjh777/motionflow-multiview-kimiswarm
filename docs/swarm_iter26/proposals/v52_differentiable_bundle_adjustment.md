# v52 Differentiable Bundle Adjustment (DBA)

**Author:** design-swarm agent  
**Module name:** `differentiable_bundle_adjustment_v52`  
**Status:** proposal (implementation not started)  
**Tracking issue:** TBD / swarm_iter26  

## 1. Motivation

The MotionFlow-MultiView pipeline currently extracts per-view 2-D poses, fuses them with geometry-aware transformers (v25, v45, v46), and refines them with temporal / physical heads (v47, v48, v49-Lite, v50/v51).  However, **camera calibration** is still treated as a fixed input rather than a jointly optimized quantity.  Small errors in intrinsics or extrinsics propagate into systematic reprojection residuals and inflated MPJPE, especially on variable-view/sparse-view inputs where the v46/v51 reliability heads must compensate for calibration drift.

The paper narrative is

```
multi-view video -> human pose extraction -> multi-view fusion and calibration -> physical-space alignment -> optimized motionflow pipeline
```

v52 closes the **multi-view fusion and calibration** loop by adding a lightweight, **differentiable bundle-adjustment (DBA)** block that refines both the 3-D pose *and* the camera extrinsics for a short clip.  It is warm-started from the model’s current 3-D estimate and the input cameras, and it is trained end-to-end with the rest of the network through the pose output.  Because the camera corrections are parameterised as small residuals (SE(3) for extrinsics, 2-D principal-point / focal scale for intrinsics), the block is **identity at init** and safe to stack after v25/v45 and before the residual MLP.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

The block is inserted **after** `multiview_geometry_fusion_v25` (and after v33 uncertainty-aware triangulation if enabled) and **before** the residual refinement head.  This is the natural integration point because:

* the per-frame 3-D pose `pred_3d_gn` is available,
* the camera parameters `K_corrected`, `R`, `t` are in scope,
* the v50/v51 per-view reliability / uncertainty buffers can be reused as DBA weights,
* the residual refinement head can still clean up the pose after bundle adjustment.

### 2.2 Forward block

```
Inputs
------
points_2d : (B, T, V, J, 2)    # detected 2-D joints
confidences : (B, T, V, J)     # detection confidence
K           : (B, T, V, 3, 3)  # input intrinsics
R           : (B, T, V, 3, 3)  # input rotation
pred_3d     : (B, T, J, 3)     # current 3-D estimate (e.g. pred_3d_gn)
view_mask   : (B, T, V)        # valid view flag

Outputs
-------
pred_3d_refined : (B, T, J, 3)
K_refined       : (B, T, V, 3, 3)
R_refined       : (B, T, V, 3, 3)
t_refined       : (B, T, V, 3)
dba_loss        : scalar       # reprojection + smoothness regulariser
```

The block is implemented as `DifferentiableBundleAdjustmentV52(nn.Module)`.

### 2.3 Learned initialisation + fixed-step optimisation

Rather than running a costly inner optimisation at every forward pass, the module uses a **hybrid approach**:

1. **Pose/camera encoder** (`DBAInitEncoder`): a tiny MLP that predicts the initial residual camera corrections and a per-joint outlier weight from the current 3-D pose and the 2-D detections.

   ```
   cam_features = concat(
       pred_3d,                       # (B*T*V, J, 3) after broadcast
       projected_points_2d,           # (B*T*V, J, 2)
       reproj_residual,              # (B*T*V, J, 2)
   )  -> (B*T*V, J, 7)
   cam_vec = mean_joints(cam_features)  # (B*T*V, d)
   delta_R, delta_t, delta_K = MLP(cam_vec)  # each view
   ```

2. **K fixed-step Gauss-Newton / Levenberg-Marquardt style refinement**.  Because pure second-order solvers are hard to differentiate through efficiently, we approximate them with `K` unrolled **gradient-descent steps** on a per-frame reprojection energy:

   ```
   for k in 1..K:
       pose_corr, cam_corr = update(E)
       apply pose_corr and cam_corr
   ```

   The update is computed by back-propagating through the reprojection energy, so the module is fully differentiable.

3. **Warm-start / identity at init**.  The encoder outputs are initialised to zero and a residual gate is used:

   ```
   R_refined = R * exp([delta_R * gate])
   t_refined = t + delta_t * gate
   K_refined = K + delta_K * gate
   pred_3d_refined = pred_3d + gate * delta_pose
   ```

   where `gate` is a scalar `v52_dba_init_gate` defaulting to `0.0`, and `delta_*` are zero-initialised linear projections.  At `gate = 0` the block is an exact identity map.

### 2.4 Reprojection energy

For view `v`, joint `j`, time `t`:

```
x_vj = pi(K_v, [R_v | t_v], P_j)     # projected 2-D point
r_vj = w_vj * (x_vj - x_hat_vj)      # weighted residual
E = sum_{v,j} rho(r_vj^T Sigma^{-1} r_vj) + lambda_pose * E_smooth(P) + lambda_cam * ||delta_cam||^2
```

where:

* `w_vj` is the product of the 2-D detection confidence `confidences` and the v50 SEFH reliability (when v50 is enabled),
* `Sigma` is a predicted per-joint observation covariance (2×2) initialised to identity,
* `rho(·)` is a robust Huber kernel (fixed, not learned) to handle outliers,
* `E_smooth(P)` is a lightweight skeleton smoothness term that penalises large changes in bone length relative to the input pose (prevents overfitting to a single view),
* `||delta_cam||^2` regularises camera corrections to stay small.

### 2.5 Output behaviour

The module returns the refined pose and cameras.  `OmniMultiViewFusionV5` uses the refined pose for the downstream residual MLP and, optionally, the refined cameras for the v37/v50 reprojection losses.  The DBA reprojection energy is added to `epi_loss` with weight `v52_dba_loss_weight`.

## 3. Configuration flags

Following the v46–v51 convention, the block is controlled by constructor flags in `OmniMultiViewFusionV5`:

```python
use_differentiable_bundle_adjustment_v52: bool = False
v52_dba_num_steps: int = 3                # K unrolled refinement steps
v52_dba_lr_pose: float = 1e-2             # pose update step size
v52_dba_lr_cam: float = 1e-3             # camera update step size
v52_dba_init_gate: float = 0.0           # warm-start gate (0 -> identity)
v52_dba_loss_weight: float = 0.1         # weight on DBA energy in total loss
v52_dba_pose_reg: float = 0.01           # lambda_pose bone smoothness
v52_dba_cam_reg: float = 0.1             # lambda_cam camera correction regulariser
v52_dba_use_sefh_weights: bool = True    # multiply by v50 reliability if available
v52_dba_update_intrinsics: bool = False  # also refine K (safer to start with False)
```

## 4. Expected MPJPE impact

* **Sparse-view / cross-domain scenarios** (v46, v51): correcting camera extrinsics on-the-fly should reduce the systematic bias that currently forces the reliability heads to down-weight otherwise good views.  Expected improvement: **1–3 mm MPJPE** on `MPJPE@2/3`.
* **Studio datasets with good calibration** (H36M): the identity-at-init design means no regression; improvements should be small (<0.5 mm).
* **WebBridge / 3DPW actual mode**: calibration is noisier; expected improvement **2–4 mm**, especially in camera-space metrics.

## 5. Risks and mitigations

See `docs/swarm_iter26/reports/agent_differentiable_bundle_adjustment_risks.md` for the full risk register.

## 6. 5-step implementation plan

1. **Prototype the standalone module** in `motionflow_mv/fusion/differentiable_bundle_adjustment_v52.py`.  Implement `DifferentiableBundleAdjustmentV52` with the encoder, the unrolled refinement loop, and the reprojection energy.  Keep intrinsics fixed (`v52_dba_update_intrinsics=False`) for the first version.

2. **Wire into `OmniMultiViewFusionV5`**.  Add the v52 flags to the constructor and call the block after `multiview_geometry_fusion_v25`.  Ensure the refined pose is passed to the residual MLP and the refined cameras can be reused by v37/v50 losses.

3. **Add warm-start smoke tests**.  With `v52_dba_init_gate=0.0`, assert that the output pose equals the input pose and that gradients flow.  Gradually increase the gate and confirm the block produces lower reprojection error on synthetic noisy cameras.

4. **Run smoke training**.  Use `configs/benchmark_v52_dba_smoke.yaml` with `v52_dba_num_steps=3`.  Compare against the v51 baseline on a small mixed manifest.  Target: finite loss, no NaNs, val_MPJPE within 5 mm of baseline after 1 epoch.

5. **Scale to full A800 run**.  If smoke passes, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` and report epoch-1 `MPJPE@k` and per-domain metrics.
