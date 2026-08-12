# v53: Differentiable Bundle Adjustment with v52 Uncertainty Weights

**Module name:** `differentiable_bundle_adjustment_v53`
**Status:** proposal (no code yet)
**Depends on:** v25, v45, v46, v47/v49-Lite, v48, v50, v51, v52

## 1. Motivation

v52 Uncertainty-Weighted Triangulation learns per-view/joint precision weights, yet the camera parameters (`K, R, t`) remain fixed inputs. Calibration drift, lens distortion, and rig flexure push real cameras away from stored values; v52 can down-weight a view but cannot correct the camera itself. v53 closes the **multi-view fusion and calibration** loop by adding a lightweight, differentiable bundle-adjustment (DBA) block that jointly refines the 3-D pose and camera parameters, warm-started from the v52 output and zero-initialised so it is **identity at init**.

## 2. Placement in `OmniMultiViewFusionV5`

Inserted **after** `UncertaintyWeightedTriangulationV52` and **before** the v50/v51 downstream heads. At this point `pred_3d_uwt`, the v52 weights `w_uwt`, and the cameras are all in scope, and downstream heads can still clean up the refined pose.

## 3. Inputs / outputs

```text
Inputs
------
features        : (B, T, V, J, d)
points_2d       : (B, T, V, J, 2)
K               : (B, T, V, 3, 3)   input intrinsics
R               : (B, T, V, 3, 3)   input rotation
t               : (B, T, V, 3)      input translation
pred_3d_init    : (B, T, J, 3)      from v52 UWT
uwt_weights     : (B, T, V, J)      from v52
view_mask       : (B, T, V)

Outputs
-------
pred_3d_refined : (B, T, J, 3)
K_refined       : (B, T, V, 3, 3)
R_refined       : (B, T, V, 3, 3)
t_refined       : (B, T, V, 3)
dba_loss        : scalar
```

## 4. Architecture and equations

**Context encoder.** A small MLP pools per-view features, v52 weights, and reprojection residuals, then predicts initial camera and pose corrections:

```
feat_v = mean_joints( concat(features_v, uwt_weights_v, reproj_residual_v, log_reproj_residual_v) )
delta_R, delta_t, delta_K = MLP_camera(feat_v)   # (B, T, V, 3)
delta_pose = MLP_pose(mean_views(feat_v))        # (B, T, J, 3)
```

All final layers are zero-initialised, so all corrections are zero at init.

**Unrolled LM refinement.** A `K`-step Levenberg-Marquardt loop minimises the robust reprojection energy. Each step uses `torch.autograd.grad`, making the block end-to-end differentiable:

```
for k in 1..K:
    E = reprojection_energy(P^k, K^k, R^k, t^k)
    [dP, dK, dR, dt] = torch.autograd.grad(E, [P, K, R, t], create_graph=True)
    P^{k+1}  = P^k  - lr_pose * (dP + lambda_pose * P^k)
    R^{k+1}  = SO3_exp(-lr_cam * dR) @ R^k
    t^{k+1}  = t^k  - lr_cam * dt
```

**Warm-start / identity at init.** A scalar gate `v53_dba_init_gate` (default `0.0`) scales every correction, so the block is an exact identity map at init:

```
R_refined = SO3_exp([delta_R * gate]) @ R
K_refined = K * (1 + delta_K * gate)
t_refined = t + delta_t * gate
P_refined = P_init + gate * delta_pose
```

**Reprojection energy.**

```
x_vj     = pi(K_v, [R_v | t_v], P_j)
r_vj     = w_vj^uwt * (x_vj - x_hat_vj)
rho(s)   = { 0.5*s                         if s <= delta^2
          { delta*sqrt(s) - 0.5*delta^2     otherwise
E_reproj = sum_{v,j} rho( r_vj^T Sigma^{-1} r_vj )
E_bone   = sum_{(j1,j2)} (||P_j1 - P_j2|| - l_j1j2^0)^2
E_dba    = E_reproj + lambda_pose*E_bone + lambda_cam*||delta_cam||^2
```

`w_vj^uwt` are the v52 UWT weights (clamped to `[v53_dba_min_weight, 1.0]`). `Sigma` is a learned 2×2 per-joint observation covariance initialised to identity. `rho` is a fixed Huber robust kernel. `E_bone` uses skeleton bone lengths `l^0` from the input pose to keep the refinement plausible. The energy is added to the total loss with `v53_dba_loss_weight`.

## 5. Configuration flags

```python
use_v53_differentiable_bundle_adjustment: bool = False
v53_dba_num_steps: int = 3
v53_dba_lr_pose: float = 1e-2
v53_dba_lr_cam: float = 1e-3
v53_dba_init_gate: float = 0.0
v53_dba_loss_weight: float = 0.05
v53_dba_pose_reg: float = 0.01
v53_dba_cam_reg: float = 0.1
v53_dba_huber_delta: float = 1.0
v53_dba_update_intrinsics: bool = False
v53_dba_use_uwt_weights: bool = True
v53_dba_min_views: int = 3
v53_dba_warmup_epochs: int = 0
```

## 6. Expected MPJPE impact

* **Sparse-view (v46/v51):** 1–3 mm improvement on `MPJPE@2/3` by fixing systematic camera bias.
* **Cross-domain / WebBridge / 3DPW:** 2–4 mm improvement on per-domain metrics where calibration is noisier.
* **Studio (H36M):** identity-at-init means no regression; gain < 0.5 mm.

## 7. Risks

See `docs/swarm_iter27/reports/agent_differentiable_bundle_adjustment_risks.md`.

## 8. 5-step implementation plan

1. **Prototype** `motionflow_mv/fusion/differentiable_bundle_adjustment_v53.py` with the context encoder, unrolled LM solver, and robust reprojection energy. Keep `v53_dba_update_intrinsics=False`.
2. **Wire into `OmniMultiViewFusionV5`** after v52 UWT and before v50/v51; pass `uwt_weights` into the block.
3. **Warm-start smoke tests:** with `v53_dba_init_gate=0.0`, assert pose/camera outputs equal inputs to `1e-5` and gradients flow.
4. **Run smoke training** with `configs/benchmark_v53_dba_smoke.yaml`; target val_MPJPE within 3 mm of v52 baseline after 1 epoch.
5. **Scale to A800** if smoke passes; add a queue entry in `scripts/launch_v33_a800_queue.py` and report `MPJPE@k` plus learned camera correction magnitudes.
