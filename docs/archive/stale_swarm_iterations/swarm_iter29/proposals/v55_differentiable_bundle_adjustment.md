# v55 Differentiable Bundle Adjustment (DBA)

## 1. Module name and one-line purpose

**Module:** `DifferentiableBundleAdjustmentV55` → `motionflow_mv/fusion/differentiable_bundle_adjustment_v55.py`

**One-line purpose:** Jointly refine the v54-calibrated 3-D pose and the camera intrinsics/extrinsics using v52 uncertainty weights and 2-D reprojection residuals, while remaining identity-at-init so a v54 checkpoint loads unchanged.

## 2. Placement in the OmniMultiViewFusionV5 forward pass

After v54 PSC-v2 and before the final residual MLP / v47/v49 temporal / v50 SEFH heads.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, floor_height, bone_scale, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
v55 DifferentiableBundleAdjustmentV55
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id,
     optional psc2_floor_height / psc2_bone_scale hints)
    → pred_3d_dba, K_dba, R_dba, t_dba, dba_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

v55 closes the loop between camera calibration and pose estimation: cameras refine the pose and the pose refines the cameras, but only after the local physical calibration from v54 is in place.

## 3. Inputs, outputs, and shapes

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | v54-calibrated 3-D pose (`pred_3d_psc2`). |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoint observations. |
| `K` | `(B, T, V, 3, 3)` | Intrinsics from upstream calibration. |
| `R` | `(B, T, V, 3, 3)` | Rotations from upstream calibration. |
| `t` | `(B, T, V, 3)` | Translations from upstream calibration. |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view per-joint triangulation weights. |
| `view_mask` | `(B, T, V)` | Binary visibility over views. |
| `domain_id` | `(B,)` | Domain labels (optional, used for per-domain regularization). |
| `psc2_floor_height` | `(B, T)` or `(B, T, 1)` | Optional v54 floor height hint. |
| `psc2_bone_scale` | `(B, T, n_bones)` | Optional v54 canonical bone-scale hint. |
| **Output** `pred_3d_dba` | `(B, T, J, 3)` | Bundle-adjusted 3-D pose. |
| **Output** `K_dba` | `(B, T, V, 3, 3)` | Refined intrinsics. |
| **Output** `R_dba` | `(B, T, V, 3, 3)` | Refined rotations. |
| **Output** `t_dba` | `(B, T, V, 3)` | Refined translations. |
| **Output** `dba_loss` | `scalar` | Robust reprojection + regularization loss. |

## 4. Architecture

### 4.1 Camera correction head

A per-view MLP maps a compact camera feature vector into small additive/multiplicative corrections:

```
feat_cam = concat(
    K[:, :, v, 0, 0], K[:, :, v, 1, 1], K[:, :, v, 0, 2], K[:, :, v, 1, 2],  # fx, fy, cx, cy
    R[:, :, v].flatten(-2), t[:, :, v]
)
Δ = MLP_cam(feat_cam)  # (B, T, 8)

Δf = Δ[..., 0:1]          # focal log-scale
Δpp = Δ[..., 1:3]         # principal-point offset (pixels)
Δr = Δ[..., 3:6]          # rotation axis-angle (rad)
Δt = Δ[..., 6:8]          # translation correction (m)

K_v' = K_v · diag(exp(Δf), exp(Δf), 1) + [[0,0,Δpp_x], [0,0,Δpp_y], [0,0,0]]
R_v' = exp([Δr]_×) · R_v
t_v' = t_v + Δt
```

The final layer of `MLP_cam` is zero-initialized so `Δ = 0` and `(K_v', R_v', t_v') = (K_v, R_v, t_v)` at initialization.

### 4.2 Pose refinement head

A per-joint MLP predicts a gated residual:

```
feat_pose = concat(
    pred_3d,                    # (B, T, J, 3)
    uwt_weights.mean(dim=2),    # (B, T, J)
    reproj_hint,                # (B, T, J, 1)
    psc2_floor_hint,            # (B, T, J, 1)
    psc2_bone_hint              # (B, T, J, 1)
)
ΔX = MLP_pose(feat_pose)        # (B, T, J, 3)
pred_3d_dba = pred_3d + σ(g_dba) · ΔX
```

`MLP_pose` final layer is zero-initialized and the residual gate logit is initialized to `−6.0` (`σ(−6) ≈ 0.0025`), so `pred_3d_dba == pred_3d` at initialization.

### 4.3 Robust reprojection loss

```
π(K', R', t'; X') projects X' into view (K', R', t')
ρ(e) = 0.5 e^2               if |e| ≤ δ
       δ (|e| − 0.5 δ)       otherwise   (Huber)

L_reproj = Σ w · ρ(π(K'_v, R'_v, t'_v; X'_j) − x_vj) / Σ w
L_cam    = λ_cam · (||Δf||^2 + ||Δpp||^2 + ||Δr||^2 + ||Δt||^2)
L_pose   = λ_pose · ||ΔX||^2
L_dba    = L_reproj + L_cam + L_pose
```

Weights `w` are the v52 UWT weights when `v55_dba_use_uwt_weights=True`; otherwise uniform over visible joints.

### 4.4 Identity-at-init mechanism

- Zero-initialize the final `Linear` layer of `MLP_cam` and `MLP_pose`.
- Initialize the pose residual gate logit to `v55_dba_residual_gate_init = -6.0`.
- Initialize camera correction MLP output biases to zero.
- As a result, at the first forward pass:
  - `K_dba == K`, `R_dba == R`, `t_dba == t`.
  - `pred_3d_dba == pred_3d_psc2`.
  - `dba_loss` is the reprojection loss under the original cameras/pose plus zero-valued regularizers.

## 5. Expected MPJPE impact and main risks

| View setting | Expected impact |
|--------------|-----------------|
| Full views | `−0.4` to `−1.0 mm` by correcting small residual calibration biases. |
| Sparse views (`MPJPE@2`, `@3`) | `−1.0` to `−2.5 mm`; joint camera/pose refinement reweights remaining views more accurately than fixed cameras. |
| 3DPW actual / cross-domain | Moderate gains (`−0.5` to `−1.5 mm`) if extrinsics drift is present. |

**Main risks:**

| Risk | Symptom | Mitigation |
|------|---------|------------|
| Camera correction overfits to dataset calibration | Cross-dataset MPJPE rises, per-camera corrections grow large. | Clamp focal scale to `[0.90, 1.10]`, principal-point offset to `[-20, 20]` px, rotation to `[-2°, 2°]`, translation to `[-5, 5]` cm; regularize with `L_cam`. |
| Joint camera/pose instability | Training diverges, NaN/Inf in pose or cameras. | Use Huber robustifier; damp camera updates with small learning-rate multiplier; gate pose residual with `σ(−6)`. |
| Compute / memory overhead | OOM or >10% step slowdown. | Compute reprojection loss on a subsampled `(V, J)` grid during training; keep MLPs shallow (`n_layers=2`, `hidden=64`). |
| Double-counting v54 physical constraints | v55 reprojection loss pulls feet through floor. | Feed v54 floor/contact and bone-scale hints to the pose head; down-weight reprojection for joints violating v54 floor constraints. |

## 6. Smoke acceptance criteria

On the local RTX 4090 with `configs/benchmark_v55_differentiable_bundle_adjustment_smoke.yaml`:

1. **Identity-at-init:** loading the best v54 checkpoint with v55 enabled and taking no training step changes `val_MPJPE@full` by `< 0.1 mm`.
2. **No regression:** `val_MPJPE@full` stays within `1 mm` of the v54 PSC-v2 baseline after one epoch.
3. **Stability:** no NaN, Inf, or OOM through one full epoch.
4. **Camera sanity:** at least `95%` of refined cameras satisfy
   - `0.95 ≤ exp(Δf) ≤ 1.05`,
   - `||Δpp||_2 ≤ 20` px,
   - `||Δr||_2 ≤ 2°`,
   - `||Δt||_2 ≤ 5` cm.
5. **Sparse-view non-regression:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

**New files:**

- `motionflow_mv/fusion/differentiable_bundle_adjustment_v55.py` — `DifferentiableBundleAdjustmentV55` module.
- `configs/benchmark_v55_differentiable_bundle_adjustment_smoke.yaml` — smoke config copied from v54 PSC-v2 smoke with v55 flags enabled.
- `scripts/run_v55_differentiable_bundle_adjustment_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 checkpoint.
- `tests/test_differentiable_bundle_adjustment_v55.py` — unit tests for identity-at-init, camera correction bounds, gradient flow, and reprojection loss.

**Files to modify:**

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add `use_v55_differentiable_bundle_adjustment` flag and the v55 hyperparameter block in `__init__`.
  - Instantiate `DifferentiableBundleAdjustmentV55` after the v54 PSC-v2 block.
  - Call it in `forward` immediately after v54 and before the final residual MLP / v47/v49 / v50 heads.
  - Add `dba_loss` to the `epi_loss` dictionary with key `v55_dba`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Ensure `domain_id` is forwarded to the model.
  - Aggregate `loss_dict["v55_dba"]` with weight `v55_dba_loss_weight` only after `v55_dba_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py` — add full-run entry `v55_differentiable_bundle_adjustment_on_v54` warm-started from the best v54 checkpoint.

## 8. Config flags and defaults (v55 naming convention)

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_differentiable_bundle_adjustment` | bool | `False` | Master toggle |
| `v55_dba_hidden` | int | `64` | Hidden dimension of camera/pose MLPs |
| `v55_dba_n_layers` | int | `2` | MLP depth |
| `v55_dba_identity_init` | bool | `True` | Zero-init final layers and gate |
| `v55_dba_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` |
| `v55_dba_correct_intrinsics` | bool | `True` | Enable focal/principal-point correction |
| `v55_dba_correct_extrinsics` | bool | `True` | Enable rotation/translation correction |
| `v55_dba_use_uwt_weights` | bool | `True` | Weight reprojection by v52 UWT weights |
| `v55_dba_use_psc2_hints` | bool | `True` | Feed v54 floor/bone hints to pose head |
| `v55_dba_huber_delta` | float | `5.0` | Huber robustifier threshold (pixels) |
| `v55_dba_camera_reg_weight` | float | `0.1` | `L_cam` multiplier |
| `v55_dba_pose_reg_weight` | float | `0.01` | `L_pose` multiplier |
| `v55_dba_loss_weight` | float | `0.01` | Multiplier on total `L_dba` |
| `v55_dba_warmup_epochs` | int | `0` | Epochs before `dba_loss` contributes |
| `v55_dba_min_visible_views` | int | `2` | Skip joints/views with fewer visible views |
