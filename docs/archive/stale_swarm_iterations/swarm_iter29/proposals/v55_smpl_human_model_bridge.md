# v55 SMPL Human Model Bridge

## 1. Module name and one-line purpose

**Module:** `SMPLHumanModelBridgeV55` → `motionflow_mv/fusion/smpl_human_model_bridge_v55.py`

**Purpose:** Add a lightweight, differentiable SMPL body-model bridge after v54 PSC-v2 that predicts SMPL shape/pose parameters from the calibrated 3-D pose and applies a small, gated residual correction back to the joints, enforcing anatomically plausible human shape and joint-angle priors while remaining identity-at-init.

## 2. Placement in the OmniMultiViewFusionV5 forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    
v55 SMPLHumanModelBridgeV55
    (consumes pred_3d_psc2, uwt_weights, view_mask, domain_id)
    → pred_3d_smpl, smpl_loss, pred_shape, pred_pose
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

The bridge sits **immediately after** `PhysicalSpaceCalibrationV2V54` and **before** any final residual MLP or downstream temporal/SEFH heads. It refines the already physically calibrated pose with a weak human-body-model prior, but does not replace the upstream triangulation or physical calibration blocks.

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc2` | `(B, T, J, 3)` | Output of v54 PSC-v2, in meters. `J = 17` for the standard skeleton. |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/per-joint triangulation weights (robustness signal). |
| `view_mask` | `(B, T, V)` | Boolean mask of visible views. |
| `domain_id` | `(B,)` or `(B, T)` | Domain label used to select per-domain bone-length normalization. |
| `pred_camera` (optional) | `(B, T, V, 3, 4)` | Camera matrices, if reprojection loss is enabled. |
| `points_2d` (optional) | `(B, T, V, J, 2)` | 2-D keypoints, if reprojection loss is enabled. |

### Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_smpl` | `(B, T, J, 3)` | SMPL-refined 3-D pose. |
| `smpl_loss` | `()` | Scalar auxiliary loss (pose prior + shape prior + reprojection residual). |
| `pred_shape` | `(B, T, 10)` | Predicted SMPL shape parameters `β`. |
| `pred_pose` | `(B, T, 72)` | Predicted SMPL pose parameters `θ` (axis-angle, 24 joints × 3). |
| `smpl_gate_value` | `()` | Scalar gate diagnostic. |

## 4. Architecture

### 4.1 Core idea

The v54 module enforces local physical constraints (floor, bone length, contact) but has no explicit notion of a global, anatomically valid human body. The SMPL bridge closes that gap by:

1. Mapping the calibrated joints to SMPL parameters via a small MLP.
2. Running the parameters through a frozen, differentiable SMPL layer to obtain a canonical body-model prior.
3. Computing a small, gated residual between the regressed SMPL joints and the input pose.
4. Applying the residual only when the upstream uncertainty justifies it.

The module is deliberately lightweight: it does **not** replace the heavy triangulation backbone, it only adds a soft, trainable human-model prior on top of v54.

### 4.2 Sub-components

**a) Shape/Pose Regressor**

A small MLP that consumes the flattened per-joint positions and per-joint UWT-weighted reliability:

```text
features = concat(
    flatten(pred_3d_psc2),                 # (B, T, J*3)
    flatten(uwt_weights.mean(dim=2)),        # (B, T, J)  per-joint mean reliability
    flatten(uwt_weights.std(dim=2))          # (B, T, J)  per-joint reliability spread
)

hidden = MLP(features, hidden=[v55_smpl_hidden, v55_smpl_hidden], activation="relu")
pred_shape = Linear(hidden, 10)             # SMPL β
pred_pose  = Linear(hidden, 72)             # SMPL θ
```

The final `Linear` layers are **zero-initialized** so that `pred_shape ≈ 0` and `pred_pose ≈ 0` at init, producing the neutral SMPL mesh.

**b) Frozen SMPL Layer**

Use the existing `data/smpl` model files (SMPL v1.0) wrapped in a thin, non-trainable differentiable layer. The layer takes `β` and `θ` and returns the 3-D joint positions `J_smpl` for the 24-joint SMPL skeleton.

To avoid a hard dependency, provide a `use_smpl_layer=True` flag. If the SMPL files are unavailable, the module falls back to a learned neutral skeleton regressor that is also zero-initialized.

**c) Skeleton Alignment / Joint Mapping**

SMPL has 24 joints; our backbone outputs 17. Use a fixed, pre-defined linear map (stored as a constant `(17, 24)` matrix) to project from SMPL joints to the 17-joint space. The map is initialized from the standard SMPL-H36M/MPII joint correspondence and is **not** learned.

```text
J_smpl_17 = M_smpl_to_17 @ J_smpl_24   # (B, T, 17, 3)
```

**d) Gated Residual Head**

A shallow MLP predicts a per-joint residual and a scalar gate:

```text
corr_features = concat(pred_3d_psc2, J_smpl_17, pred_3d_psc2 - J_smpl_17)
residual = Linear(MLP(corr_features), 3)     # (B, T, J, 3)
gate_logit = Scalar(v55_smpl_residual_gate_init)  # init -6.0 → σ(-6.0) ≈ 0.0025
pred_3d_smpl = pred_3d_psc2 + σ(gate_logit) * residual
```

The final `residual` linear layer is zero-initialized, and the gate is initialized to `v55_smpl_residual_gate_init = -6.0`.

**e) Losses**

| Loss | Weight | Description |
|------|--------|-------------|
| `L_pose` | `v55_smpl_pose_weight` (default 0.01) | L2 prior on `pred_pose`, encouraging plausible axis-angle magnitudes. |
| `L_shape` | `v55_smpl_shape_weight` (default 0.001) | L2 prior on `pred_shape` (β ≈ 0 at init). |
| `L_reproj` | `v55_smpl_reproj_weight` (default 0.1) | Optional reprojection residual between `pred_3d_smpl` and 2-D keypoints weighted by UWT weights. |
| `L_residual` | `v55_smpl_residual_weight` (default 0.01) | L2 regularizer on the gated residual to keep corrections small. |

`smpl_loss = L_pose + L_shape + L_reproj + L_residual`.

All loss weights are ramped from zero during the first `v55_smpl_warmup_epochs` epochs.

### 4.3 Identity-at-init mechanism

- The shape/pose regressor final layers are zero-initialized → `β = 0, θ = 0` → neutral SMPL pose.
- The residual head final layer is zero-initialized → `residual = 0`.
- The residual gate is initialized to `logit = -6.0` → `σ(logit) ≈ 0.0025`, so the correction is negligible.
- The SMPL-to-17-joint mapping is a fixed constant matrix.
- Therefore, loading a v54 checkpoint with v55 enabled changes the output by less than `1e-4` m per joint before any training step.

## 5. Expected MPJPE impact and main risks

### Expected impact

| View setting | Expected change |
|-------------|-----------------|
| Full views | `−0.5 to −1.5 mm` on clean H36M/WebBridge; larger on 3DPW actual mode (`−1.0 to −2.5 mm`). |
| Sparse `@2/3` | `−1.0 to −3.0 mm` because the SMPL prior helps resolve ambiguous sparse-view configurations. |
| Smoke | `−0.5 to −2.0 mm` versus v54 PSC-v2 baseline, assuming SMPL layer is available. |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| SMPL layer unavailable or slow | Import errors, OOM, or CPU bottleneck. | Provide `use_smpl_layer=False` fallback; use a tiny learned neutral skeleton regressor. Load SMPL on demand and cache vertices. |
| Shape/pose prior over-constrains unusual poses | Val MPJPE rises for extreme motions (sports, sitting). | Keep pose/shape loss weights small (`0.01` / `0.001`); residual gate starts closed; losses ramp from zero. |
| Joint mapping misalignment | Elbows/knees drift because SMPL-to-17 mapping is approximate. | Use standard, verified SMPL-H36M correspondences; map via learned offset only after identity gate opens. |
| v54 checkpoint regress | Output changes by `>0.1 mm` at init. | Unit test `‖pred_3d_smpl - pred_3d_psc2‖_∞ < 1e-4`; zero-init all final layers and gate. |
| Extra memory from SMPL layer | RTX 4090 OOM in smoke. | Run SMPL on CPU or use the fallback in smoke; keep batch size small. |

## 6. Smoke acceptance criteria

- **Identity-at-init:** loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE@full` by `< 0.1 mm`.
- **No NaN/Inf/OOM:** at least one full epoch completes on the RTX 4090 smoke config.
- **MPJPE guard:** `val_MPJPE@full` is within `1 mm` of the v54 PSC-v2 baseline after one epoch.
- **Shape/pose sanity:** predicted `β` and `θ` magnitudes are finite; `‖β‖₂ < 10` and `‖θ‖₂ < 10π` for `95%` of frames.
- **Sparse-view improvement (or no regression):** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/smpl_human_model_bridge_v55.py` — the `SMPLHumanModelBridgeV55` module.
- `tests/test_smpl_human_model_bridge_v55.py` — unit tests for identity-at-init, shape/pose sanity, and gradient flow.
- `configs/benchmark_v55_smpl_human_model_bridge_smoke.yaml` — smoke config copied from `benchmark_v54_psc_v2_smoke.yaml` with v55 flags enabled.
- `scripts/run_v55_smpl_human_model_bridge_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 checkpoint.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flag `use_v55_smpl_human_model_bridge` and instantiate `SMPLHumanModelBridgeV55` when enabled.
  - Insert the call after the v54 PSC-v2 block and before the final residual MLP / v47/v49 temporal / v50 SEFH heads.
  - Add `smpl_loss` to the `epi_loss` dictionary with key `v55_smpl`.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Forward `domain_id` to the model if not already done.
  - Aggregate `loss_dict["v55_smpl"]` with weight `v55_smpl_loss_weight`, honoring `v55_smpl_warmup_epochs`.

- `scripts/launch_v33_a800_queue.py`
  - Add an A800 queue entry `v55_smpl_human_model_bridge_on_v54` warm-starting from the best v54 checkpoint.

### Config flags and defaults

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v55_smpl_human_model_bridge` | bool | `False` | Master toggle |
| `v55_smpl_hidden` | int | `64` | Regressor / correction MLP hidden dim |
| `v55_smpl_n_layers` | int | `2` | Regressor MLP depth |
| `v55_smpl_num_domains` | int | `8` | Number of domains (for optional per-domain bone normalization) |
| `v55_smpl_identity_init` | bool | `True` | Zero-initialize final regressor/residual layers |
| `v55_smpl_residual_gate_init` | float | `-6.0` | Gate logit at init |
| `v55_smpl_use_smpl_layer` | bool | `True` | Use real SMPL layer; if false, use learned neutral fallback |
| `v55_smpl_loss_weight` | float | `1.0` | Multiplier on total `smpl_loss` |
| `v55_smpl_pose_weight` | float | `0.01` | Weight of pose prior |
| `v55_smpl_shape_weight` | float | `0.001` | Weight of shape prior |
| `v55_smpl_reproj_weight` | float | `0.1` | Weight of optional reprojection residual |
| `v55_smpl_residual_weight` | float | `0.01` | Weight of residual L2 regularizer |
| `v55_smpl_warmup_epochs` | int | `0` | Epochs before `smpl_loss` contributes to total loss |

## Notes

- Do not implement any code; this proposal is for design review and agent assignment only.
- The SMPL body-model prior should be treated as a **soft regularizer**, not a hard projection. Keep the gate closed at initialization so the v54 baseline is preserved.
- This module naturally follows v54 PSC-v2: once local physical calibration is strong, adding a global human-shape prior can correct remaining anatomically implausible configurations without disturbing the proven pipeline.
