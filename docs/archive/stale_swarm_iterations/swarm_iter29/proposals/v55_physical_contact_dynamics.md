# v55 Physical-Contact Dynamics (PCD)

## 1. Module name and one-line purpose

**Module:** `PhysicalContactDynamicsV55` → `motionflow_mv/fusion/physical_contact_dynamics_v55.py`

A lightweight, identity-at-init contact-dynamics head that learns per-joint contact states from the v54-calibrated pose and enforces zero-velocity / no-penetration constraints at contact, improving foot-floor stability and limb self-contact plausibility while leaving the upstream calibration stack untouched at initialization.

## 2. Placement in the OmniMultiViewFusionV5 forward pass

PCD sits **after v54 PSC-v2** and **before the final residual MLP / v47/v49 temporal / v50 SEFH heads**:

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init
    ↓
v52 UWT → pred_3d_uwt
    ↓
v53 PSC → pred_3d_psc
    ↓
v54 PSC-v2 → pred_3d_psc2, psc2_loss
    ↓
v55 Physical-Contact Dynamics
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask)
    → pred_3d_pcd, contact_logits, floor_height_pcd, pcd_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_in` | `(B, T, J, 3)` | v54 PSC-v2 output pose, meters. |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/per-joint triangulation weights. |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints. |
| `K, R, t` | `(B, V, 3, 3)`, `(B, V, 3, 3)`, `(B, V, 3)` | Camera intrinsics and extrinsics. |
| `view_mask` | `(B, T, V)` | Binary visible-view mask. |
| `domain_id` | `(B,)` or `None` | Domain label for per-domain contact priors. |

**Outputs:**

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_out` | `(B, T, J, 3)` | Contact-corrected pose. |
| `contact_logits` | `(B, T, J, K)` | Contact-state logits for `K` contact classes (e.g., `K=3`: none / floor / self). |
| `floor_height` | `(B, T, 1)` | Estimated per-frame floor height from UWT-weighted foot joints. |
| `pcd_loss` | scalar | Auxiliary contact-dynamics loss. |

## 4. Architecture

### 4.1 Contact-state head
A 2-layer MLP per joint:

```text
features_j = concat(
    pred_3d_in,
    velocity(pred_3d_in),           # (B, T, J, 3)
    acceleration(pred_3d_in),       # (B, T, J, 3)
    height_above_floor(pred_3d_in), # (B, T, J, 1)
    uwt_mean_weights                # (B, T, J, 1)
)
contact_logits = MLP(features_j; hidden=v55_pcd_hidden, out=J*K)  → (B, T, J, K)
```

The **final layer is zero-initialized**, so all contact logits are `0` at init and the contact class is uniform / non-committal.

### 4.2 Floor-height estimator
Same UWT-weighted foot-joint estimator as v54: estimate a per-frame `floor_height`, but keep it independent of the v54 floor head so the two can ablate cleanly.

### 4.3 Contact-conditioned correction
A small 1-layer graph refiner along the kinematic parent-child edges computes a per-joint residual:

```text
residual = σ(gate) · tanh( MLP( concat(contact_probs, pred_3d_in, floor_distance) ) )
```

* `gate` is initialized to `v55_pcd_residual_gate_init = -6.0` → `σ(gate) ≈ 0.0025`.
* The MLP output projection is zero-initialized, so `residual ≈ 0` at init.
* The correction is applied to `pred_3d_in`.

### 4.4 Losses

| Loss | Formula / purpose | Weight |
|---|---|---|
| `L_contact` | Cross-entropy between predicted contact logits and a soft pseudo-label derived from foot height + velocity. | `v55_pcd_contact_weight` |
| `L_zero_vel` | For joints with `P(contact) > 0.5`, penalize `||velocity_j||²`. | `v55_pcd_zero_vel_weight` |
| `L_no_pen` | Penalize joints below the estimated floor when `P(contact=0)` is high. | `v55_pcd_no_pen_weight` |
| `L_reproj` | 2-D reprojection consistency of the corrected pose (reuses v52 weights). | `v55_pcd_reproj_weight` |
| `L_temporal` | Acceleration smoothness on the corrected pose, gated by `1 - P(contact)`. | `v55_pcd_temporal_weight` |

Total: `pcd_loss = v55_pcd_loss_weight * (L_contact + L_zero_vel + L_no_pen + L_reproj + L_temporal)`.

### 4.5 Identity-at-init mechanism

1. Final contact MLP layer → zero init.
2. Correction residual MLP → zero init.
3. Residual gate → `logit = -6.0`.
4. Soft contact pseudo-labels are computed without learnable parameters.

Thus a v54 checkpoint loaded with v55 enabled produces `pred_3d_out ≈ pred_3d_in` and `pcd_loss ≈ 0`.

## 5. Expected MPJPE impact and main risks

* **Full views:** identity `< 0.1 mm`; smoke `−0.3 to −0.8 mm`; full `−0.5 to −1.2 mm`.
* **Sparse views (`@2/3`):** larger relative gain `−1.0 to −2.5 mm` because contact constraints stabilise noisy triangulation.
* **3DPW / in-the-wild:** larger gains on ground-contact-heavy sequences.

| Risk | Symptom | Mitigation |
|---|---|---|
| Over-constrains non-upright motion | MPJPE rises on jumps/sitting. | Soft losses; contact label velocity threshold; gate initialized to near-zero. |
| Conflicts with v54 contact loss | Redundant penalty; double-counting floor. | Make v55 losses optional via flags; ablate with v54 `use_contact=False`. |
| Mis-classified self-contact | Hands pulled to torso / other limbs. | Limit self-contact to a small predefined joint-pair set; use low `v55_pcd_self_contact_weight`. |
| Identity-at-init regression | v54 checkpoint changes `>0.1 mm`. | Unit test on `||pred_3d_out - pred_3d_in||_∞ < 1e-4`. |

## 6. Smoke acceptance criteria

1. `val_MPJPE@full` within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
2. No NaN / Inf / OOM through at least one full epoch.
3. Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
4. Contact-state sanity: estimated `floor_height` is finite and foot-joint contact probabilities are `>0.5` for at least `80%` of frames where feet are near the floor.
5. Zero-velocity loss is zero when no foot joint is predicted to be in contact.
6. `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

### New files
* `motionflow_mv/fusion/physical_contact_dynamics_v55.py` — module implementation.
* `configs/benchmark_v55_physical_contact_dynamics_smoke.yaml` — smoke config.
* `scripts/run_v55_physical_contact_dynamics_smoke_local_4090.sh` — smoke launch script.
* `tests/test_physical_contact_dynamics_v55.py` — unit tests (identity, contact-label sanity, gradient flow).

### Files to modify
* `motionflow_mv/fusion/omniview_fusion_v5.py`
  * Add constructor flag block `use_v55_physical_contact_dynamics`.
  * Instantiate `PhysicalContactDynamicsV55` when enabled.
  * Insert call after v54 PSC-v2 and before the final heads.
  * Add `pcd_loss` to the `epi_loss` dict.
* `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  * Aggregate `loss_dict["v55_pcd"]` with `v55_pcd_loss_weight` and a warmup guard.
* `scripts/launch_v33_a800_queue.py`
  * Add A800 queue entry `v55_physical_contact_dynamics_on_v54`.
* `AGENTS.md`
  * Add the v55 conventions section with flags, defaults, and workflow.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_physical_contact_dynamics` | bool | `False` | Master toggle |
| `v55_pcd_hidden` | int | `64` | Contact / correction MLP hidden dim |
| `v55_pcd_n_layers` | int | `2` | Contact MLP depth |
| `v55_pcd_num_contact_classes` | int | `3` | none / floor / self |
| `v55_pcd_identity_init` | bool | `True` | Zero-init final layers and gate |
| `v55_pcd_residual_gate_init` | float | `-6.0` | Gate logit at init |
| `v55_pcd_use_floor` | bool | `True` | Enable v55 floor-height estimator |
| `v55_pcd_use_self_contact` | bool | `True` | Enable predefined self-contact pairs |
| `v55_pcd_loss_weight` | float | `1.0` | Multiplier on total `pcd_loss` |
| `v55_pcd_contact_weight` | float | `0.05` | Weight of cross-entropy contact loss |
| `v55_pcd_zero_vel_weight` | float | `0.01` | Weight of zero-velocity at contact |
| `v55_pcd_no_pen_weight` | float | `0.01` | Weight of no-penetration loss |
| `v55_pcd_reproj_weight` | float | `0.1` | Weight of reprojection term |
| `v55_pcd_temporal_weight` | float | `0.01` | Weight of temporal smoothness |
| `v55_pcd_contact_velocity_thresh` | float | `0.3` | m/s threshold for soft contact label |
| `v55_pcd_min_visible_views` | int | `2` | Skip joints with fewer visible views |
| `v55_pcd_warmup_epochs` | int | `0` | Epochs before `pcd_loss` contributes |
