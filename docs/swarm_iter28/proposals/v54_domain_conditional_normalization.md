# v54: Domain-Conditional Normalization of Calibrated Multi-View Fusion (DCN-CMF)

**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v52 Uncertainty-Weighted Triangulation (#182), v53 Physical-Space Calibration (#183), v48 Domain Generalization (#164)

## Motivation

v52 Uncertainty-Weighted Triangulation (UWT) replaces the fixed v25 triangulation weights with learned per-view, per-joint precision weights. v53 Physical-Space Calibration (PSC) then calibrates the fused 3-D pose against floor-plane and canonical bone-length invariants. Together they produce a geometrically consistent pose, but neither removes residual *dataset-level affine bias* in the calibrated 3-D pose space: studio rigs (H36M, MPI-INF-3DHP) tend to yield tight, metric-scale skeletons, while in-the-wild clips (3DPW, AIST++) exhibit larger variance in global scale, floor-height convention, and joint-position distribution. v48 already conditions feature tokens before triangulation, but it does not normalize the *output* pose and weights after v53.

v54 introduces a **Domain-Conditional Normalization of Calibrated Multi-View Fusion (DCN-CMF)** module that sits **after** v53 PSC and **before** the final residual MLP / temporal heads. It learns per-domain affine transforms for the calibrated pose and the v52 UWT weights, normalizing both to a shared, domain-invariant representation. The module is identity-at-init, so it can be warm-started from any trained v52/v53 checkpoint without changing the baseline MPJPE.

## Proposed Architecture

### Module placement

```text
nfeat (B, T, V, J, d)                              pred_3d_gn (B, T, J, 3)
    │                                                       │
    ▼                                                       ▼
[v52 UWT]───────────────────────────────────────────────────┘
    │ pred_3d_uwt, uwt_weights, uwt_log_precision
    ▼
[v53 PSC]  (floor + bone calibration)
    │ pred_3d_psc, floor_height, bone_scale
    ▼
[v54 DCN-CMF]  ← domain_id + view_mask
    │
    ▼
[final residual MLP / v47/v49 temporal / v50 SEFH]
```

`motionflow_mv/fusion/domain_conditional_normalization_v54.py` exposes `DomainConditionalNormalizationV54`.

### Forward logic

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc` | `(B, T, J, 3)` | Calibrated 3-D pose from v53 PSC |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/joint precision weights |
| `floor_height` | `(B, T)` | v53 estimated floor height |
| `bone_scale` | `(B, T, n_bones)` | v53 per-bone scale ratios |
| `domain_id` | `(B,)` | Integer domain label per clip |
| `view_mask` | `(B, T, V)` | Binary active-view mask |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_dcn` | `(B, T, J, 3)` | Domain-normalized 3-D pose |
| `weights_dcn` | `(B, T, V, J)` | Domain-normalized triangulation weights |
| `dcn_loss` | scalar | Auxiliary regularization loss |

### Equations

Let `d_emb = v54_dcn_hidden`, `J` the number of joints, `V` the number of views, and `g = v54_dcn_num_groups` the number of joint groups used to share pose affine parameters (`g = 1` means a single shared affine; `g = J` means per-joint).

**1. Domain and view-count conditioning**

```
z_d      = Embed(domain_id)                       # (B, d_emb)
n_active = view_mask.sum(dim=-1).float() / V      # (B, T), normalized view count
v_emb    = MLP_view_count(n_active)               # (B, T, d_emb)
z        = z_d.unsqueeze(1) + v_emb                 # (B, T, d_emb)
```

**2. Domain-conditional pose affine**

```
# Group joints into g kinematic groups via a fixed partition.
h_p  = MLP_pose(z)                                # (B, T, g * 3 * 2)
γ_p, β_p = split(h_p)                             # each (B, T, g, 3)
γ_p = 1.0 + 0.1 * tanh(γ_p)                       # stays near 1.0 at init
β_p = 0.1 * tanh(β_p)                             # near 0.0 at init
pred_3d_dcn = γ_p[..., j, :] * pred_3d_psc + β_p[..., j, :]
```

At initialization, the final layer of `MLP_pose` is zero-initialized, so `γ_p = 1.0` and `β_p = 0.0`, giving `pred_3d_dcn = pred_3d_psc`.

**3. Domain-conditional weight normalization**

```
log_w      = log(uwt_weights + ε)                 # (B, T, V, J)
μ_logw     = log_w.mean(dim=(2, 3))               # (B, T)
σ_logw     = log_w.std(dim=(2, 3))                # (B, T)
w_stats    = concat([μ_logw, σ_logw], -1)        # (B, T, 2)
h_w        = MLP_weight(concat([z, w_stats], -1))  # (B, T, 2)
γ_w, β_w   = split(h_w)                           # each (B, T)
γ_w = 1.0 + 0.1 * tanh(γ_w)
β_w = 0.1 * tanh(β_w)
log_w_norm = γ_w * (log_w - μ_logw) / (σ_logw + ε) + β_w
weights_dcn = exp(log_w_norm).clamp(min=v54_dcn_min_weight, max=1.0)
```

Again, the final layer of `MLP_weight` is zero-initialized, so at startup `γ_w = 1.0`, `β_w = 0.0`, and `weights_dcn = uwt_weights` up to the clamp floor.

**4. Optional physical hints**

```
# Floor and bone hints are concatenated to the pose MLP input to keep the
# normalization physically grounded:
hints = concat([
    floor_height.unsqueeze(-1).expand(-1, -1, J),   # (B, T, J)
    bone_scale.mean(dim=-1, keepdim=True).expand(-1, -1, J),  # (B, T, J)
], dim=-1)  # (B, T, 2J)
```

Controlled by `v54_dcn_use_floor_hint` and `v54_dcn_use_bone_hint`.

**5. Auxiliary loss**

```
dcn_loss = λ_pose * (tanh(γ_p)^2 + tanh(β_p)^2).mean()
         + λ_weight * ((γ_w - 1.0)^2 + tanh(β_w)^2).mean()
```

This penalizes large domain-specific deviations from the identity mapping and keeps the learned per-domain scale/shift small and smooth.

## Config Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v54_domain_conditional_normalization` | `bool` | `False` | Enable the v54 DCN-CMF module |
| `v54_dcn_hidden` | `int` | `64` | Hidden dimension of domain / pose / weight MLPs |
| `v54_dcn_num_groups` | `int` | `1` | Number of joint groups for the pose affine (`1` = global, `J` = per-joint) |
| `v54_dcn_use_view_count` | `bool` | `True` | Append active-view-count embedding |
| `v54_dcn_use_floor_hint` | `bool` | `True` | Feed v53 floor height into the pose MLP |
| `v54_dcn_use_bone_hint` | `bool` | `True` | Feed v53 bone-scale into the pose MLP |
| `v54_dcn_min_weight` | `float` | `0.05` | Floor on normalized triangulation weights |
| `v54_dcn_pose_loss_weight` | `float` | `0.01` | Weight of the pose-affine penalty |
| `v54_dcn_weight_loss_weight` | `float` | `0.01` | Weight of the weight-affine penalty |
| `v54_dcn_identity_init` | `bool` | `True` | Zero-init final MLP layers and affine centers |

## Expected MPJPE Impact

- **Identity check:** Enabling v54 on a trained v52/v53 checkpoint should change `val_MPJPE` by less than `0.1 mm` before any gradient step, because all affine transforms are identity at init.
- **Mixed-domain val:** Expect a `0.5–1.2 mm` reduction by removing residual domain-specific scale/shift bias in the calibrated pose.
- **Sparse-view gains:** Normalizing weight magnitudes per domain should improve `MPJPE@2/3` on cross-dataset evaluation, where rare-view or low-confidence domains no longer get dominated by camera-rich studio data.
- **Physical-space alignment:** By producing a more domain-invariant pose representation, the downstream v28/v31/v40 physical losses should be more stable, especially when datasets use different floor or bone-length conventions.

## Risks

See `docs/swarm_iter28/reports/agent_domain_conditional_normalization_risks.md` for full details. Top risks include redundancy with v48 domain FiLM and v53 PSC, per-domain scale collapse, overfitting small domains, and warm-start drift.

## 5-Step Implementation Plan

1. **Create module.** Implement `motionflow_mv/fusion/domain_conditional_normalization_v54.py` with the API above. Enforce identity-at-init by zero-initializing the final layers of `MLP_pose`, `MLP_weight`, and `MLP_view_count`, and initialize the affine scale around `1.0` using `tanh` reparameterization.
2. **Wire into `OmniMultiViewFusionV5`.** Add the v54 flags, instantiate `DomainConditionalNormalizationV54`, and call it immediately after the v53 PSC block (around lines 1811–1834), feeding it `pred_3d_psc`, `uwt_weights_for_v53`, `psc_floor_height`, `psc_bone_scale`, `domain_id`, and `view_mask`. Feed the resulting `pred_3d_dcn` and `weights_dcn` into the final residual MLP / temporal heads.
3. **Update trainer and config.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, accumulate the new `dcn_loss` into the total loss when `use_v54_domain_conditional_normalization=True`, with a warmup guard mirroring v53. Create `configs/benchmark_v54_dcn_smoke.yaml` and a smoke shell script `scripts/run_v54_dcn_smoke_local_4090.sh`.
4. **Smoke test.** Run on RTX 4090 on top of the latest v53 checkpoint. Verify identity-at-init: loading with v54 enabled should yield the same `val_MPJPE` as the parent v53 run before training. Confirm no NaN/OOM and that `MPJPE@2/3` does not regress.
5. **Ablation and A800 queue.** Ablate `v54_dcn_use_view_count`, `v54_dcn_use_floor_hint`, `v54_dcn_use_bone_hint`, and `v54_dcn_num_groups`. If the smoke shows a gain, add an entry to `scripts/launch_v33_a800_queue.py` for a full A800 run and update the v54 status in `docs/swarm_iter28/status.md`.

## Paper Story Fit

v54 completes the multi-view fusion and calibration stage of the MotionFlow pipeline. After v52 learns data-driven triangulation weights and v53 calibrates the fused pose against physical invariants, v54 removes the remaining dataset-level affine bias so that the downstream physical-space alignment and temporal refinement stages operate on a single, domain-invariant motion representation. This directly supports the ICRA/CVPR 2027 paper narrative: multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline.
