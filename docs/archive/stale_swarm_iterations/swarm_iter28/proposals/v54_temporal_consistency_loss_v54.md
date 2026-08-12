# v54 Temporal Consistency Loss

## Summary

Add a differentiable temporal consistency regulariser to the v52/v53 multi-view fusion pipeline. It penalises implausible frame-to-frame jitter and high-frequency acceleration in the triangulated 3D skeleton, weighted by the per-joint/per-view uncertainty that v52 already produces. The module is loss-only, does not change inference architecture, and is identity-at-init because its loss term starts at zero and can be linearly ramped in.

## Motivation

The current pipeline (v25 geometry fusion → v52 uncertainty-weighted triangulation → v53 physical-space calibration) produces per-frame poses independently across time. For video data this leaves frame-to-frame jitter unregularised, which hurts downstream motion quality even when per-frame MPJPE is good. Prior temporal modules such as v47/v49 add extra transformer heads that increase latency and memory. A lightweight loss operating directly on the final 3D pose sequence is cheaper, keeps the inference path unchanged, and naturally exploits the uncertainty estimates already learned in v52.

## Architecture

The module is implemented as `TemporalConsistencyLossV54` and is called after v53 (or v52/v25 if v53 is disabled) in `OmniMultiViewFusionV5.forward`. It computes three terms on the final per-frame 3D pose `P ∈ R^(B×T×J×3)`:

1. **Velocity smoothness** (`v54_tc_use_velocity`): penalises large per-joint velocities between consecutive frames.
2. **Acceleration smoothness** (`v54_tc_use_acceleration`): penalises large second-order differences, encouraging constant-velocity motion.
3. **Uncertainty-aware masking** (`v54_tc_use_uncertainty`): down-weights the loss for joints/time-steps where the UWT/v53 per-joint uncertainty is high.

Optional learned temporal gating is avoided to preserve identity-at-init; instead a scalar `v54_tc_loss_weight` scales the entire term and can be set to `0.0` for an exact no-op.

### Equations

Let `P[b, t, j, :]` be the predicted 3D position for batch element `b`, frame `t`, and joint `j`.

Velocity term:

```
V[b, t, j] = P[b, t, j] - P[b, t-1, j]      for t = 1..T-1
L_vel = (1 / ((T-1) J)) Σ_{b,t,j} || V[b, t, j] ||_2
```

Acceleration term:

```
A[b, t, j] = P[b, t+1, j] - 2 P[b, t, j] + P[b, t-1, j]   for t = 1..T-2
L_acc = (1 / ((T-2) J)) Σ_{b,t,j} || A[b, t, j] ||_2
```

Uncertainty-aware reweighting uses per-joint visibility confidence `γ[b, t, j] ∈ [0, 1]` derived from v52/v53 weights (or set to `1` if those modules are disabled):

```
L_tc = v54_tc_loss_weight * ( β_vel L_vel + β_acc L_acc )
```

where `β_vel = v54_tc_velocity_weight`, `β_acc = v54_tc_acceleration_weight`, and the per-step uncertainty weights `γ` are applied inside the sums above.

## Inputs and Outputs

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Final per-frame 3D pose from v53/v52/v25. |
| `uwt_weights` | `(B, T, V, J)` or `(B, T, J)` | Optional per-view-joint or per-joint weights from v52/v53. |
| `view_mask` | `(B, T, V)` | Boolean mask for missing views; used when reducing UWT weights. |
| `domain_id` | `(B,)` | Domain label; unused by default but reserved for per-domain weighting. |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `loss` | scalar | Temporal consistency loss `L_tc`. |

No output pose is produced; the module only adds `loss` to the total training objective.

## Config Flags

```yaml
use_v54_temporal_consistency_loss: false   # Master toggle
v54_tc_loss_weight: 0.01                  # Global scalar on the loss
v54_tc_velocity_weight: 1.0                 # Weight for the velocity term
v54_tc_acceleration_weight: 1.0             # Weight for the acceleration term
v54_tc_use_velocity: true                   # Enable velocity smoothness
v54_tc_use_acceleration: true             # Enable acceleration smoothness
v54_tc_use_uncertainty: true                # Use v52/v53 weights for masking
v54_tc_uncertainty_min_views: 2             # Minimum visible views to trust weight
v54_tc_warmup_epochs: 0                     # Linear ramp-up start epoch
v54_tc_normalize_by_length: true            # Normalize by number of valid frames
```

## Expected MPJPE Impact

- **Baseline**: v53 full run on A800; MPJPE is the current best reference.
- **Conservative estimate**: `−0.3` to `−0.8` mm reduction in per-frame MPJPE by suppressing outlier frames.
- **Optimistic estimate**: `−1.0` to `−1.5` mm on sequences with fast motion or occlusion, where temporal jitter is largest.
- **Risk of degradation**: If `v54_tc_loss_weight` is too high, the model may oversmooth rapid motions; smoke tests should start at `0.001` and sweep to `0.05`.

## Risks

1. **Over-smoothing of fast actions**: High temporal loss weight can penalise legitimate fast motion.
2. **Coupling with UWT uncertainty**: Bad uncertainty estimates from v52/v53 will bias the temporal loss.
3. **Sequence-length dependence**: Very short clips (`T < 3`) cannot compute the acceleration term.
4. **Conflict with v47/v49 temporal modules**: Redundant temporal regularisation may stack non-linearly and hurt training stability.

See `agent_temporal_consistency_loss_v54_risks.md` for mitigations.

## 5-Step Implementation Plan

1. **Implement `TemporalConsistencyLossV54`** in `motionflow_mv/fusion/temporal_consistency_loss_v54.py`. Keep it pure-functional (no learnable parameters) so it is identity-at-init and warm-startable from any prior checkpoint.
2. **Wire into `OmniMultiViewFusionV5`**. Add the config flags to `__init__`, instantiate the module after v53, and call it with `pred_3d_gn` and `uwt_weights_for_v53`. Accumulate `v54_tc_loss` into `epi_loss` with the standard warmup guard.
3. **Add YAML smoke config** `configs/benchmark_v54_temporal_consistency_loss_smoke.yaml` that enables v52 + v53 + v54 with `v54_tc_loss_weight=0.001`, and a smoke script `scripts/run_v54_temporal_consistency_loss_smoke_local_4090.sh`.
4. **Run smoke tests** on RTX 4090. Verify no NaN/Inf, that loss starts at zero, and that `val_MPJPE@full` is within `0.5` mm of v53. Sweep `v54_tc_loss_weight ∈ {0.001, 0.01, 0.05}`.
5. **Add A800 queue entry** in `scripts/launch_v33_a800_queue.py` (e.g. `v54_temporal_consistency_loss_on_v53`) after v53 smoke completes, then update `AGENTS.md` status tables and the v54 run log.
