# v55 Cross-Dataset Domain Bridge (CDDB)

**Tracking issue:** #TBD  
**Base branch:** `v55-cddb`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## 1. Module name and one-line purpose

**Module:** `CrossDatasetDomainBridgeV55` (`motionflow_mv/fusion/cross_dataset_domain_bridge_v55.py`)

**One-line purpose:** After v54 physically calibrates each domain’s pose, CDDB learns a *dataset-agnostic canonical pose space* and a small gated refiner that removes remaining dataset-specific biases before the final prediction heads, so that H36M, MPI, WebBridge and 3DPW all project into the same kinematic reference frame.

## 2. Position in the OmniMultiViewFusionV5 forward pass

CDDB sits **immediately after v54 PSC-v2** and **before the final residual MLP / v47/v49 temporal / v50 SEFH heads**, so the downstream heads consume a domain-neutral, physically consistent pose.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init
    ↓
v52 UWT  → pred_3d_uwt,  uwt_weights
    ↓
v53 PSC  → pred_3d_psc
    ↓
v54 PSC-v2 → pred_3d_psc2
    ↓
v55 CrossDatasetDomainBridge
    (consumes pred_3d_psc2, uwt_weights, domain_id,
            optional pred_3d_uwt for reprojection anchoring)
    → pred_3d_cddb, cddb_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

Assume batch `(B, T)` frames, `J` joints, `3` coordinates, and `V` views.

**Inputs:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc2` | `(B, T, J, 3)` | Calibrated 3-D pose from v54 PSC-v2. |
| `uwt_weights` | `(B, T, V, J)` or `(B, T, V)` | Per-view/per-joint reliability weights from v52 UWT. |
| `domain_id` | `(B,)` or `(B, T)` | Integer domain labels (H36M=0, MPI=1, WebBridge=2, 3DPW=3, etc.). |
| `pred_3d_uwt` | `(B, T, J, 3)` | Optional raw UWT pose, used as an anchoring signal for the identity-at-init reprojection term. |
| `view_mask` | `(B, T, V)` | Valid-view mask (for masking loss/statistics). |

**Outputs:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_cddb` | `(B, T, J, 3)` | Domain-bridge 3-D pose. |
| `cddb_loss` | scalar | Domain-alignment + soft-physical loss. |
| `domain_emb` | `(B, T, d)` | Optional per-domain latent embedding (returned for diagnostics). |

## 4. Architecture

### 4.1 Per-domain canonical affine

For each domain `d`, learn an affine transform of the pose into a shared canonical space:

```
z_d = scale_d(pred_3d_psc2 - shift_d)
```

* `shift_d`: learned per-domain 3-D centroid offset `(num_domains, 3)`, initialized to zero.
* `scale_d`: learned per-domain log-scale `(num_domains, 3)`, initialized to zero (so `exp(scale)=1`).
* The transform is **joint-shared** (one shift/scale per domain, not per joint), so it only removes coarse dataset-level location/scale biases and cannot warp the skeleton at identity.

### 4.2 Domain-agnostic refiner

A small per-joint MLP (or 1-layer GNN optional fallback) processes the canonical pose plus per-joint reliability-derived features:

```
feat_j = concat(z_j,  log(uwt_weight_j + eps),  pred_3d_psc2_j - pred_3d_uwt_j)
delta_j = MLP(feat_j)   # output layer zero-initialized
gate_j  = σ(logit_gate) # logit_gate initialized to -6.0 → σ≈0.0025
pred_3d_cddb_j = pred_3d_psc2_j + gate_j * tanh(delta_j)
```

* Hidden dim: `v55_cddb_hidden=64` (default).
* Depth: `v55_cddb_n_layers=2` (default).
* Output layer zero-initialized → `delta_j = 0` at init, so `pred_3d_cddb == pred_3d_psc2` regardless of `gate_j`.
* Residual gate initialized to `-6.0` keeps the correction effectively zero even if `tanh(delta)` is slightly non-zero during very first steps.

### 4.3 Losses

All losses are weighted and summed only after `v55_cddb_warmup_epochs`.

**Domain-alignment loss `L_align`** (main):  
Maximum-Mean-Discrepancy (MMD) between the per-domain distributions of canonical pose features. We use a fast linear kernel on the `(B*T, J, 3)` flattened pose, computed only between pairs of domains present in the batch. MMD is purely a loss term; it does not add parameters or change the forward pass at inference.

**Anchor-reprojection loss `L_reproj`** (keeps bridge honest):  
For the original (full-view) configuration, encourage the bridge to preserve reprojection consistency:

```
L_reproj = || Π_cam(pred_3d_cddb) - points_2d ||_2 * uwt_weights
```
This is intentionally a soft term; `v55_cddb_reproj_weight=0.1` (default) so the module cannot drift far from the physically calibrated pose.

**Bone-length preservation `L_bone`** (low-risk regularizer):  
Encourage the bridge to preserve the per-bone lengths of `pred_3d_psc2`:

```
L_bone = mean( | ||bone||_cddb - ||bone||_psc2 | )
```

Weighted by `v55_cddb_bone_weight=0.05`.

**Optional cross-domain consistency `L_xcons`** (if multi-domain batching is enabled):  
For samples from two different domains that are paired by a shared action/pose label (when available), minimize `||pred_cddb_A - pred_cddb_B||_2`. When no pairs exist, this loss is zero. Weight `v55_cddb_xcons_weight=0.01`.

Total:

```
cddb_loss = v55_cddb_loss_weight * (
              v55_cddb_align_weight   * L_align +
               v55_cddb_reproj_weight * L_reproj +
               v55_cddb_bone_weight   * L_bone +
               v55_cddb_xcons_weight  * L_xcons )
```

### 4.4 Identity-at-init mechanism

1. Per-domain affine `shift` zero, `scale` log-zero → `z_d = pred_3d_psc2`.
2. Refiner MLP output layer zero-initialized → `delta = 0`.
3. Gate logit `-6.0` → `σ≈0.0025`, so `pred_3d_cddb - pred_3d_psc2 ≈ 0`.
4. All loss weights can be ramped from zero via `v55_cddb_warmup_epochs`.

Therefore loading a v54 checkpoint with v55 enabled and not training keeps `val_MPJPE` change `< 0.1 mm`.

## 5. Expected MPJPE impact and main risks

### Expected impact

| Scenario | Expected change |
|----------|-----------------|
| In-domain (H36M→H36M, MPI→MPI) | `±0.2 mm` (identity-like). |
| Cross-domain / mixed training | `-1.0 to -2.5 mm` on the weakest domain, `-0.5 to -1.2 mm` on average. |
| Sparse views (`@2`, `@3`) | `-0.8 to -2.0 mm` because domain bias is more harmful when views are few. |
| Full views | `-0.3 to -0.8 mm`. |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Affine overfits domain shift into a pose deformation** | Bone lengths change across domains; MPJPE rises. | Joint-shared affine (cannot alter shape); bone-length loss clamps it. |
| **MMD loss is unstable / negative transfer** | Smoke loss spikes; validation NaN. | Use linear MMD (no exponentiated kernels); loss warmup from zero; `v55_cddb_align_weight=0.01`. |
| **v54 checkpoint regresses at init** | `val_MPJPE` jumps `>0.1 mm` when v55 enabled. | Zero-init all final layers and gate; unit test `||pred_cddb - pred_psc2||_∞ < 1e-4`. |
| **Cross-domain consistency requires labels/pairs** | `L_xcons` is rarely active. | Make it optional and zero when no pairs are available. |
| **Slows training / OOM** | Memory or step time grows. | Module is shallow (`n_layers=2`, `hidden=64`), no sequence-level attention. |

## 6. Smoke acceptance criteria (RTX 4090)

1. **Identity-at-init:** loading a v54 checkpoint with v55 enabled and no training changes `val_MPJPE` by `< 0.1 mm`.
2. **No regression:** `val_MPJPE@full` stays within `1 mm` of the v54 baseline on the same smoke config.
3. **Stability:** no NaN, Inf, or OOM through at least one full epoch.
4. **Domain-shift sanity:** per-domain validation means shift by `< 2 mm` relative to in-domain; no domain collapses.
5. **Sparse-view safety:** `MPJPE@2` and `MPJPE@3` are not worse than v54 baseline.
6. **Loss sanity:** `cddb_loss` is finite and non-increasing after warmup begins.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/cross_dataset_domain_bridge_v55.py` — `CrossDatasetDomainBridgeV55` module.
- `configs/benchmark_v55_cross_dataset_domain_bridge_smoke.yaml` — smoke config.
- `scripts/run_v55_cross_dataset_domain_bridge_smoke_local_4090.sh` — smoke launch script, warm-starting from the best v54 checkpoint.
- `tests/test_cross_dataset_domain_bridge_v55.py` — unit tests for identity-at-init, per-domain affine, MMD finite-ness, and gate behavior.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flag `use_v55_cross_dataset_domain_bridge`.
  - Instantiate `CrossDatasetDomainBridgeV55` when enabled.
  - Insert call after the v54 PSC-v2 block and before final heads.
  - Add `cddb_loss` to the `epi_loss` dict with key `v55_cddb`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Forward `domain_id` to the model (reuses existing v48 plumbing).
  - Aggregate `loss_dict["v55_cddb"]` with `v55_cddb_loss_weight` and warmup guard.
- `scripts/launch_v33_a800_queue.py`
  - Add `v55_cross_dataset_domain_bridge_on_v54` entry warm-starting from the best v54 checkpoint.
- `AGENTS.md`
  - Add a "v55 cross-dataset domain bridge conventions" section with flags, defaults, and workflow after v54 is accepted.

## Config flags and defaults

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v55_cross_dataset_domain_bridge` | bool | `False` | Master toggle. |
| `v55_cddb_hidden` | int | `64` | Refiner MLP hidden dimension. |
| `v55_cddb_n_layers` | int | `2` | Refiner depth. |
| `v55_cddb_num_domains` | int | `8` | Number of domains (must cover H36M/MPI/WebBridge/3DPW). |
| `v55_cddb_identity_init` | bool | `True` | Zero-initialize final layers and gate. |
| `v55_cddb_residual_gate_init` | float | `-6.0` | Gate logit at init. |
| `v55_cddb_loss_weight` | float | `1.0` | Multiplier on total `L_cddb`. |
| `v55_cddb_align_weight` | float | `0.01` | MMD domain-alignment weight. |
| `v55_cddb_reproj_weight` | float | `0.1` | Anchor-reprojection weight. |
| `v55_cddb_bone_weight` | float | `0.05` | Bone-length preservation weight. |
| `v55_cddb_xcons_weight` | float | `0.01` | Cross-domain consistency weight. |
| `v55_cddb_warmup_epochs` | int | `0` | Epochs before `cddb_loss` contributes. |
