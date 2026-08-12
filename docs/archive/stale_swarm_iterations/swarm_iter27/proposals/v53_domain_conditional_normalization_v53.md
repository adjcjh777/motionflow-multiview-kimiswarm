# v53: Domain-Conditional Normalization of UWT Outputs

**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v52 Uncertainty-Weighted Triangulation (#182), v48 domain generalization (#164)

## Motivation

v52 Uncertainty-Weighted Triangulation (UWT) replaces the fixed v25/v45 triangulation weights with learned per-view, per-joint precision weights and a gated residual correction. While this improves multi-view fusion, the *distribution* of those learned weights and the *scale* of the resulting 3-D skeleton still carry strong dataset-specific biases. In the mixed WebBridge training manifest (H36M, MPI-INF-3DHP, AIST++, 3DPW), studio rigs produce tight, high-confidence weight distributions, whereas in-the-wild clips yield broader, lower-magnitude weights and larger absolute pose offsets. A single global weight floor or pose prior cannot optimally handle all regimes.

v53 proposes a **Domain-Conditional Normalization (DCN)** module that sits **after** v52 UWT. It recalibrates the UWT weights and the triangulated pose in a domain-aware manner before physical-space alignment, while remaining identity-at-init so it can be warm-started from any v52 checkpoint.

## Proposed Architecture

### Module placement

```textnfeat (B, T, V, J, d)        pred_3d_v52 (B, T, J, 3)        uwt_weights (B, T, V, J)n    |                                |                              |n    ▼                                ▼                              ▼n[v52 UWT]─────────────────────────────────────────────────────────┘n    │                                                                      n    │ pred_3d_v52, uwt_weights, log_precisionn    ▼n[v53 Domain-Conditional Normalization]  ← domain_id + view_maskn    │                                                                      n    ▼n[ ST transformer / physical-space alignment ]n```

`motionflow_mv/fusion/domain_conditional_normalization_v53.py` exposes `DomainConditionalNormalizationV53`.

### Forward logic

Inputs:

- `pred_3d_v52`: `(B, T, J, 3)` refined 3-D pose from v52 UWT.
- `uwt_weights`: `(B, T, V, J)` v52 precision weights in `[0, 1]`.
- `log_precision`: `(B, T, V, J)` v52 log-precision values (used as features).
- `points_2d`: `(B, T, V, J, 2)`, `K`: `(B, T, V, 3, 3)`, `R`: `(B, T, V, 3, 3)`, `t`: `(B, T, V, 3)` — camera/2-D data needed for optional re-triangulation.
- `domain_id`: `(B,)` integer domain labels.
- `view_mask`: `(B, T, V)` binary mask.

Outputs:

- `pred_3d_out`: `(B, T, J, 3)` domain-normalized 3-D pose.
- `dcn_loss`: scalar auxiliary regularization loss.
- `weights_out`: `(B, T, V, J)` recalibrated triangulation weights.

### Equations

Let `d_emb` be the domain-embedding dimension and `g` the number of joint groups (default `g = J`, i.e. per-joint).

**1. Domain and view-count conditioning**

```
z_d = Embed(domain_id)                              # (B, d_emb)
n_active = view_mask.sum(dim=-1)                    # (B, T)
v_emb    = MLP_view_count(n_active)                 # (B, T, d_emb)
z        = z_d.unsqueeze(1) + v_emb                 # (B, T, d_emb)
```

**2. Weight recalibration**

```
w_stats = [uwt_weights.mean(dim=(2,3)),              # (B, T)
           uwt_weights.std(dim=(2,3))]               # (B, T)
h_w     = MLP_weight(torch.cat([z, w_stats], -1))   # (B, T, V·J)
Δlog_w  = h_w.view(B, T, V, J)                      # final layer zero-init
log_w'  = log_precision + Δlog_w
w'      = sigmoid(log_w').clamp(min=v53_dcn_min_weight, max=1.0)
```

Because `Δlog_w = 0` at initialization, `w' = sigmoid(log_precision) = uwt_weights`.

**3. Re-triangulation with recalibrated weights**

```
P' = weighted_dlt_triangulate(points_2d, K, R, t,
                              weights=w',
                              view_mask=view_mask,
                              damping=v53_dcn_damping)
```

**4. Domain-conditional pose residual**

```
h_p     = MLP_pose(z)                              # (B, T, 2·g·3)
γ, β    = split(h_p)                               # each (B, T, g, 3)
P_out   = P' + residual_gate * (tanh(γ) * P' + β)
```

The final `Linear` layers of `MLP_weight` and `MLP_pose` are zero-initialized, and `residual_gate` is initialized to `0.0`, so `P_out = P'` and `w' = uwt_weights` at startup.

**5. Auxiliary loss**

```
dcn_loss = λ_weight * (Δlog_w ** 2).mean() + λ_pose * (tanh(γ) ** 2 + β ** 2).mean()
```

This penalizes large domain-specific deviations from the v52 baseline.

## Inputs / Outputs Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_v52` | `(B, T, J, 3)` | Pose refined by v52 UWT |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/joint weights |
| `log_precision` | `(B, T, V, J)` | v52 log-precision values |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints |
| `K`, `R`, `t` | `(B, T, V, 3, 3)`, `(B, T, V, 3, 3)`, `(B, T, V, 3)` | Calibrated cameras |
| `domain_id` | `(B,)` | Integer domain label per clip |
| `view_mask` | `(B, T, V)` | Binary active-view mask |
| `pred_3d_out` | `(B, T, J, 3)` | Domain-normalized 3-D pose |
| `weights_out` | `(B, T, V, J)` | Recalibrated triangulation weights |

## Config Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v53_domain_conditional_normalization` | `bool` | `False` | Enable the v53 DCN module |
| `v53_dcn_hidden` | `int` | `64` | Hidden dimension of domain/weight/pose MLPs |
| `v53_dcn_num_groups` | `int` | `J` | Number of joint groups for pose affine (`-1` = per-joint) |
| `v53_dcn_use_view_count` | `bool` | `True` | Append active-view-count embedding |
| `v53_dcn_use_weight_stats` | `bool` | `True` | Feed mean/std of v52 weights to the weight MLP |
| `v53_dcn_min_weight` | `float` | `0.05` | Floor on recalibrated weights |
| `v53_dcn_damping` | `float` | `1e-4` | DLT damping when re-triangulating |
| `v53_dcn_weight_loss_weight` | `float` | `0.01` | Weight of the `Δlog_w` penalty |
| `v53_dcn_pose_loss_weight` | `float` | `0.01` | Weight of the pose-affine penalty |
| `v53_dcn_identity_init` | `bool` | `True` | Zero-init final MLP layers and `residual_gate` |

## Expected MPJPE Impact

- **Primary:** 0.5–1.2 mm reduction on mixed-domain validation by removing the per-domain bias in v52 weights.
- **Sparse views:** Normalizing weight magnitudes per domain should improve `MPJPE@2/3` on cross-dataset evaluation, because rare-view domains no longer get drowned out by camera-rich studio data.
- **Warm-start:** Because of identity initialization, enabling v53 on a trained v48/v50/v51/v52 checkpoint should change `val_MPJPE` by less than 0.1 mm before any gradient step.

## Risks

See `docs/swarm_iter27/reports/agent_domain_conditional_normalization_v53_risks.md` for full details. Top risks include redundancy with v52's internal precision MLP, re-triangulation instability, and overfitting small domains.

## 5-Step Implementation Plan

1. **Create module.** Implement `motionflow_mv/fusion/domain_conditional_normalization_v53.py` with the API above, including explicit identity-at-init tests.
2. **Wire into `OmniMultiViewFusionV5`.** Add the v53 flags, instantiate `DomainConditionalNormalizationV53`, and call it immediately after the v52 UWT block (around lines 1748–1765), feeding it `pred_3d_gn`, `uwt_weights`, `uwt_log_precision`, cameras, `domain_id`, and `view_mask`.
3. **Update trainer and config.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, accumulate the new `dcn_loss` into the total loss when `use_v53_domain_conditional_normalization` is set. Add `configs/benchmark_v53_dcn_smoke.yaml` and a smoke shell script.
4. **Smoke test.** Run on RTX 4090 on top of the latest v52 checkpoint. Verify identity-at-init: loading with v53 enabled should yield the same `val_MPJPE` as the parent v52 run before training. Confirm no NaN/OOM and that `MPJPE@2/3` does not regress.
5. **Ablation and A800 queue.** Ablate `v53_dcn_use_view_count`, `v53_dcn_use_weight_stats`, and `v53_dcn_num_groups`. If the smoke shows a gain, add an entry to `scripts/launch_v33_a800_queue.py` for a full A800 run and update the v53 status in `docs/swarm_iter27/status.md`.

## Paper Story Fit

v53 closes the loop between multi-view fusion/calibration and physical-space alignment. After v52 learns data-driven triangulation weights, v53 normalizes those weights—and the resulting 3-D pose—according to the dataset domain. This makes the downstream physical prior operate on a domain-invariant pose representation, which is central to the optimized motionflow pipeline for ICRA/CVPR 2027.
