# v55: Domain-Conditional Normalization v2 (DCN-v2)

## 1. Module name and one-line purpose

**Module:** `DomainConditionalNormalizationV55` → `motionflow_mv/fusion/domain_conditional_normalization_v55.py`

**One-line purpose:** After v54 PSC-v2 has locally calibrated the 3-D pose, learn a lightweight, per-domain affine normalization of the calibrated pose and v52 uncertainty weights so that downstream heads operate on a single domain-invariant motion representation.

## 2. Placement in `OmniMultiViewFusionV5`

```text
v52 UWT → pred_3d_uwt, uwt_weights
v53 PSC → pred_3d_psc, psc_floor, psc_bone_scale
v54 PSC-v2 → pred_3d_psc2, psc2_floor, psc2_bone_scale
v55 DCN-v2 → pred_3d_dcn, weights_dcn
final residual MLP / v47/v49 temporal / v50 SEFH
```

v55 sits **immediately after v54 PSC-v2** and **before any final residual MLP, temporal, or SEFH head**. It does not replace v54; it removes residual dataset-level affine drift from the physically-calibrated output.

## 3. Inputs, outputs, and shapes

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc2` | `(B, T, J, 3)` | Locally calibrated 3-D pose from v54 PSC-v2 |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/joint precision weights |
| `psc2_floor_height` | `(B, T)` | v54 estimated floor height |
| `psc2_bone_scale` | `(B, T, n_bones)` | v54 per-bone scale ratios |
| `domain_id` | `(B,)` | Integer domain label per clip |
| `view_mask` | `(B, T, V)` | Binary active-view mask |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_dcn` | `(B, T, J, 3)` | Domain-normalized 3-D pose |
| `weights_dcn` | `(B, T, V, J)` | Domain-normalized triangulation weights |
| `dcn_loss` | scalar | Auxiliary regularization loss |

## 4. Architecture

### 4.1 Domain and view-count conditioning

```
z_d      = Embed(domain_id)                       # (B, d_emb)
n_active = view_mask.sum(dim=-1).float() / V      # (B, T)
v_emb    = MLP_view_count(n_active)               # (B, T, d_emb)
z        = z_d.unsqueeze(1) + v_emb                 # (B, T, d_emb)
```

`MLP_view_count` final layer is zero-initialized, so `v_emb = 0` at init.

### 4.2 Domain-conditional pose affine

Joints are partitioned into `g = v55_dcn_num_groups` kinematic groups. A single shared affine (`g = 1`) is the default to keep the module low-risk.

```
h_p       = MLP_pose(concat([z, hints], -1))     # (B, T, g * 3 * 2)
γ_p, β_p  = split(h_p)                            # each (B, T, g, 3)
γ_p       = 1.0 + 0.1 * tanh(γ_p)                 # near 1.0 at init
β_p       = 0.1    * tanh(β_p)                    # near 0.0 at init
pred_3d_dcn = γ_p[..., j, :] * pred_3d_psc2 + β_p[..., j, :]
```

The final layer of `MLP_pose` is zero-initialized, giving `pred_3d_dcn = pred_3d_psc2` at init.

### 4.3 Domain-conditional weight normalization

```
log_w     = log(uwt_weights + ε)                  # (B, T, V, J)
μ_logw    = log_w.mean(dim=(2,3), keepdim=True) # (B, T, 1, 1)
σ_logw    = log_w.std (dim=(2,3), keepdim=True) # (B, T, 1, 1)
w_stats   = concat([μ_logw.squeeze(), σ_logw.squeeze()], -1)  # (B, T, 2)
h_w       = MLP_weight(concat([z, w_stats], -1))  # (B, T, 2)
γ_w, β_w  = split(h_w)                            # each (B, T)
γ_w       = 1.0 + 0.1 * tanh(γ_w)
β_w       = 0.1    * tanh(β_w)
log_w_norm = γ_w * (log_w - μ_logw) / (σ_logw + ε) + β_w
weights_dcn = exp(log_w_norm).clamp(min=v55_dcn_min_weight, max=1.0)
```

The final layer of `MLP_weight` is zero-initialized, so `weights_dcn = uwt_weights` up to the clamp floor at init.

### 4.4 Optional physical hints

```
hints = concat([
    psc2_floor_height.unsqueeze(-1).expand(-1, -1, J),        # (B, T, J)
    psc2_bone_scale.mean(dim=-1, keepdim=True).expand(-1, -1, J),  # (B, T, J)
], dim=-1)  # (B, T, 2J)
```

Controlled by `v55_dcn_use_floor_hint` and `v55_dcn_use_bone_hint`.

### 4.5 Auxiliary loss

```
dcn_loss = λ_pose  * (tanh(γ_p - 1)^2 + tanh(β_p)^2).mean()
       + λ_weight * ((γ_w - 1.0)^2 + tanh(β_w)^2).mean()
```

This penalizes large domain-specific deviations from the identity mapping.

### 4.6 Identity-at-init mechanism

- Final layers of `MLP_pose`, `MLP_weight`, and `MLP_view_count` are zero-initialized.
- Affine scales are reparameterized as `1.0 + 0.1 * tanh(·)`; shifts as `0.1 * tanh(·)`.
- No batch normalization is used; no running statistics are kept.

## 5. Expected MPJPE impact and main risks

**Expected impact:**

- **Identity check:** Enabling v55 on a trained v54 checkpoint should change `val_MPJPE` by less than `0.1 mm` before any gradient step.
- **Mixed-domain val:** `−0.5 to −1.2 mm` reduction by removing residual domain-specific scale/shift bias in the physically-calibrated pose.
- **Sparse-view gains:** Normalizing weight magnitudes per domain should improve `MPJPE@2/3` on cross-dataset evaluation, especially for low-confidence or few-view domains.
- **Physical-space alignment:** A more domain-invariant pose representation makes downstream physical losses (v28/v31/v40) and temporal heads (v47/v49) more stable across datasets.

**Main risks:**

| Risk | Symptom | Mitigation |
|---|---|---|
| **Redundancy with v48 domain FiLM / v54 PSC-v2** | Gains are negligible because earlier modules already remove domain bias. | Keep module small; default `g=1`; make all sub-heads optional via flags; ablate against baseline. |
| **Per-domain scale collapse** | `γ_p` or `γ_w` collapse to a constant, removing useful domain variance. | Bound affine with `tanh`; auxiliary loss keeps deviations from identity small. |
| **Overfitting small domains** | Large swings on rare domains (3DPW actual, AIST++). | Use shared `MLP_pose` / `MLP_weight` with only the final affine layer domain-conditional; avoid per-domain MLPs. |
| **Warm-start drift** | Loading a v54 checkpoint with v55 enabled changes `val_MPJPE` by `>0.1 mm`. | Unit test identity-at-init; zero-init all final layers; clamp weights to avoid numerical drift. |

## 6. Smoke acceptance criteria

Run on RTX 4090 warm-started from the best available v54 PSC-v2 checkpoint:

1. `val_MPJPE@full` is within `1 mm` of the v54 PSC-v2 baseline on the same smoke config.
2. `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
3. No NaN, Inf, or OOM through at least one full epoch.
4. Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
5. Domain-scale sanity: `γ_p` stays in `[0.8, 1.2]` and `β_p` magnitude stays `< 0.2 m` for `≥95%` of frames.
6. Weight-scale sanity: `γ_w` stays in `[0.8, 1.2]` for `≥95%` of frames.

## 7. Required new files and files to modify

**New file:**

- `motionflow_mv/fusion/domain_conditional_normalization_v55.py` — `DomainConditionalNormalizationV55` module.

**Files to modify:**

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add v55 flag block, instantiate module when enabled, insert call after v54 PSC-v2, feed `pred_3d_dcn` and `weights_dcn` to downstream heads, add `dcn_loss` to `epi_loss` dictionary.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — forward `domain_id`, aggregate `loss_dict["v55_dcn"]` with `v55_dcn_loss_weight`, and honor a warmup guard.
- `configs/benchmark_v55_dcn_smoke.yaml` — smoke config copied from v54 PSC-v2 smoke with v55 flags enabled.
- `scripts/run_v55_dcn_smoke_local_4090.sh` — smoke launch script warm-starting from the best available v54 checkpoint.
- `scripts/launch_v33_a800_queue.py` — add A800 full-run entry.
- `tests/test_domain_conditional_normalization_v55.py` — unit tests for identity-at-init, per-domain affine behavior, weight normalization, and gradient flow.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_domain_conditional_normalization` | bool | `False` | Master toggle |
| `v55_dcn_hidden` | int | `64` | Hidden dimension of domain / pose / weight MLPs |
| `v55_dcn_num_groups` | int | `1` | Number of joint groups for pose affine (`1` = global, `J` = per-joint) |
| `v55_dcn_use_view_count` | bool | `True` | Append active-view-count embedding |
| `v55_dcn_use_floor_hint` | bool | `True` | Feed v54 floor height into pose MLP |
| `v55_dcn_use_bone_hint` | bool | `True` | Feed v54 bone-scale into pose MLP |
| `v55_dcn_min_weight` | float | `0.05` | Floor on normalized triangulation weights |
| `v55_dcn_pose_loss_weight` | float | `0.01` | Weight of pose-affine penalty |
| `v55_dcn_weight_loss_weight` | float | `0.01` | Weight of weight-affine penalty |
| `v55_dcn_identity_init` | bool | `True` | Zero-init final MLP layers and affine centers |
| `v55_dcn_warmup_epochs` | int | `0` | Epochs before `dcn_loss` contributes to total loss |
