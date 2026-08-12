# v55 Bone-Length-Aware Fusion (BLAF)

## 1. Module name and one-line purpose

- **Module:** `BoneLengthAwareFusionV55`
- **File:** `motionflow_mv/fusion/bone_length_aware_fusion_v55.py`
- **One-line purpose:** A per-bone canonical-length prior, gated residual refiner, and auxiliary loss that sits on top of v54 PSC-v2 and pushes the triangulated 3-D pose toward anatomically plausible bone lengths while remaining identity-at-init.

## 2. Placement in `OmniMultiViewFusionV5` forward pass

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
    ↓
v55 BoneLengthAwareFusionV55
    (consumes pred_3d_psc2, uwt_weights, K, R, t, view_mask, domain_id)
    → pred_3d_blaf, blaf_loss, bone_scale_blaf
    
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

BLAF refines the pose *after* v54 has applied its local physical calibration. It does not replace v54; it tightens the bone-length consistency of the v54 output before downstream temporal/SEFH heads consume it.

## 3. Inputs, outputs, and shapes

**Inputs**

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | Calibrated 3-D pose from v54. |
| `uwt_weights` | `(B, T, V, J)` | v52 uncertainty weights per view/joint. |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints. |
| `K` | `(B, T, V, 3, 3)` | Camera intrinsics. |
| `R` | `(B, T, V, 3, 3)` | Camera rotations. |
| `t` | `(B, T, V, 3)` | Camera translation. |
| `view_mask` | `(B, T, V)` | Valid view mask. |
| `domain_id` | `(B,)` or `(B, T)` | Domain index for per-domain canonical lengths. |
| `skeleton_edges` | `(num_bones, 2)` | Parent-child bone topology (static register buffer). |

**Outputs**

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_blaf` | `(B, T, J, 3)` | Bone-length-aware 3-D pose. |
| `blaf_loss` | `scalar` | Auxiliary bone-length and reprojection loss. |
| `bone_scale_blaf` | `(B, T, num_bones)` | Learned per-bone log-scale offsets. |

## 4. Architecture

### 4.1 Per-domain canonical bone-length head

- An embedding table `E ∈ R^{num_domains × num_bones}` initialized to zero (log-scale offsets).
- `domain_id` selects the row; output is reshaped to `(B, T, num_bones)` to allow per-frame domain tags.
- The final projection from `E` to per-bone log-scales is zero-initialized so `exp(offset) = 1` at init.

### 4.2 Bone-length residual computation

- Compute current bone lengths from `pred_3d_psc2` along `skeleton_edges`: `L_cur[b,i] = ||pred[child] - pred[parent]||_2`.
- Target length per bone: `L_tgt = L_cur * exp(offset_bone)` where `offset_bone` comes from the canonical head.
- Per-bone residual direction is along the bone vector; magnitude is `(L_tgt - L_cur)` scaled by a learned per-bone confidence derived from `uwt_weights` aggregated over the two joints and all views.
- Distribute bone corrections to child/parent joints with opposite signs weighted by inverse bone mass (child gets larger share for distal bones to avoid over-pulling parents).

### 4.3 Skeleton-graph refiner (optional GNN)

- A single-layer GNN over the kinematic chain propagates bone-length residuals to joints.
- Node features: `(J, 3)` joint position + `(J,)` aggregated per-joint UWT confidence + `(J,)` bone-length residual magnitude.
- Edge features: bone direction vector, bone length residual.
- Output is a per-joint `(J, 3)` correction delta.
- The final output projection is zero-initialized.
- Fallback: a two-layer per-joint MLP when `use_gnn=False`.

### 4.4 Gated residual update

```python
gate_logit = nn.Parameter(torch.full((), v55_blaf_residual_gate_init))  # default -6.0
pred_3d_blaf = pred_3d_psc2 + torch.sigmoid(gate_logit) * correction
```

At init `σ(-6.0) ≈ 0.0025`, making the module effectively identity.

### 4.5 Losses

| Loss | Weight flag | Description |
|---|---|---|
| `L_bone` | `v55_blaf_bone_weight` | Huber loss between corrected bone lengths and target canonical lengths, masked by visible views. |
| `L_reproj` | `v55_blaf_reproj_weight` | 2-D reprojection consistency of `pred_3d_blaf` against the original keypoints, weighted by UWT weights. |
| `L_floor` | `v55_blaf_floor_weight` | Soft floor loss: penalize foot joints below an estimated floor height from v54 output. |
| `L_temporal` | `v55_blaf_temporal_weight` | Velocity smoothness of the correction term (not the absolute pose). |
| `L_symmetry` | `v55_blaf_symmetry_weight` | Optional left-right bone-length symmetry loss. |

Total auxiliary loss: `L_blaf = v55_blaf_loss_weight * (L_bone + L_reproj + L_floor + L_temporal + L_symmetry)`.

## 5. Expected MPJPE impact and main risks

| View setting | Expected MPJPE impact |
|---|---|
| Full views | `−0.3 to −0.8 mm` |
| Sparse views (`@2/3`) | `−1.5 to −3.0 mm` |

**Main risks**

- **Over-constraining pose with a single canonical skeleton:** Different datasets have different actor proportions; per-domain scales mitigate this but still risk regressing toward mean bone lengths.
  - *Mitigation:* Initialize log-scales to zero; keep loss weight low at start (`v55_blaf_loss_weight` with warmup); allow per-domain scales and a soft Huber loss.
- **Double-counting v54 PSC-v2 bone constraints:** v54 already has a bone-scale head. BLAF must not pull in the same direction too strongly.
  - *Mitigation:* Gate residual with `−6.0`; keep `v55_blaf_bone_weight` modest (default `0.05`); run ablation disabling the v54 bone loss and compare.
- **Identity-at-init failure:** v54 checkpoint could regress if BLAF output layers are not zero-initialized.
  - *Mitigation:* Zero-initialize all final projections and the gate logit; add unit test `||pred_blaf - pred_psc2||_∞ < 1e-4` at init.
- **Sparse-view instability:** With few views, bone-length corrections can amplify triangulation noise.
  - *Mitigation:* Mask bones with invisible joints; weight corrections by UWT confidence; clamp correction magnitude to a small fraction of bone length during warmup.

## 6. Smoke acceptance criteria

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- No NaN/Inf/OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Bone-length sanity: `95%` of corrected bone lengths satisfy `exp(offset) ∈ [0.5, 2.0]`.
- Reprojection sanity: mean 2-D reprojection error of `pred_3d_blaf` is within `0.5 px` of v54 baseline.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

**New files**

- `motionflow_mv/fusion/bone_length_aware_fusion_v55.py` — `BoneLengthAwareFusionV55` module.
- `configs/benchmark_v55_blaf_smoke.yaml` — smoke config (copied from v54 smoke, v55 flags enabled).
- `scripts/run_v55_blaf_smoke_local_4090.sh` — smoke launch script warm-starting from the best v54 checkpoint.
- `tests/test_bone_length_aware_fusion_v55.py` — unit tests for identity-at-init, per-domain canonical lengths, bone-length sanity, and gradient flow.

**Files to modify**

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flags under the v55 block.
  - Instantiate `BoneLengthAwareFusionV55` when `use_v55_bone_length_aware_fusion=True`.
  - Insert the module call immediately after the v54 PSC-v2 block.
  - Add `blaf_loss` to the `epi_loss` dictionary.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Ensure `domain_id` is forwarded into the model.
  - Aggregate `loss_dict["v55_blaf"]` into the total loss with weight `v55_blaf_loss_weight` and warmup guard `v55_blaf_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py`
  - Add an A800 full-run entry `v55_bone_length_aware_fusion_on_v54` warm-starting from the best v54 checkpoint.

## 8. Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_bone_length_aware_fusion` | bool | `False` | Master toggle |
| `v55_blaf_hidden` | int | `64` | MLP/GNN hidden dimension |
| `v55_blaf_n_layers` | int | `2` | Refiner MLP depth |
| `v55_blaf_num_domains` | int | `8` | Number of domains for per-domain bone scales |
| `v55_blaf_num_bones` | int | `16` | Number of skeleton bones (derived from topology) |
| `v55_blaf_identity_init` | bool | `True` | Zero-initialize final residual layers and gate |
| `v55_blaf_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_blaf_use_gnn` | bool | `True` | Use skeleton-graph refiner (fallback: per-joint MLP) |
| `v55_blaf_gnn_layers` | int | `1` | Number of GNN layers |
| `v55_blaf_loss_weight` | float | `1.0` | Multiplier on total `L_blaf` |
| `v55_blaf_bone_weight` | float | `0.05` | Weight of `L_bone` |
| `v55_blaf_reproj_weight` | float | `0.1` | Weight of `L_reproj` |
| `v55_blaf_floor_weight` | float | `0.01` | Weight of `L_floor` |
| `v55_blaf_temporal_weight` | float | `0.01` | Weight of `L_temporal` |
| `v55_blaf_symmetry_weight` | float | `0.0` | Weight of optional `L_symmetry` |
| `v55_blaf_min_visible_views` | int | `2` | Skip bones with fewer visible views |
| `v55_blaf_warmup_epochs` | int | `0` | Epochs before `blaf_loss` contributes to total loss |
| `v55_blaf_max_correction_ratio` | float | `0.1` | Clamp correction magnitude to 10% of bone length during warmup |
